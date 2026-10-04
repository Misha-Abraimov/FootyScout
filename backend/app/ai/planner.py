"""Structured planner that never executes FootyScout tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.ai.diagnostics import (
    FailureDiagnostic,
    FailureStage,
    diagnostic_from_exception,
)
from app.ai.methodology_guard import apply_pure_methodology_guard
from app.ai.plan_builder import build_scout_plan
from app.ai.prompts import PLANNER_PROMPT, PLANNER_PROMPT_VERSION
from app.ai.provider import (
    PlannerProvider,
    PlannerProviderError,
    PlannerRefusalError,
    PlannerUsage,
)
from app.ai.provider_schemas import LLMPlannerDecision
from app.ai.request_guard import deterministic_request_decision
from app.ai.schemas import ScoutPlan


class PlannerStatus(str, Enum):
    SUCCESS = "success"
    REFUSED = "refused"
    ERROR = "error"


@dataclass(frozen=True)
class PlannerResult:
    status: PlannerStatus
    provider: str
    model: str
    prompt_version: str
    provider_decision: LLMPlannerDecision | None = None
    plan: ScoutPlan | None = None
    usage: PlannerUsage | None = None
    error_message: str | None = None
    diagnostic: FailureDiagnostic | None = None
    response_received: bool = False
    request_id: str | None = None
    http_status: int | None = None
    methodology_guard_applied: bool = False
    fallback_attempted: bool = False
    fallback_provider: str | None = None
    fallback_model: str | None = None
    fallback_reason: str | None = None
    fallback_success: bool = False
    provider_attempts: int = 0


class ScoutPlanner:
    """Turns one natural-language question into one strict ScoutPlan."""

    def __init__(self, provider: PlannerProvider) -> None:
        self._provider = provider

    @property
    def model(self) -> str:
        return self._provider.model

    @property
    def provider(self) -> str:
        return self._provider.provider

    def plan(self, question: str) -> PlannerResult:
        normalized_question = question.strip()
        if not normalized_question:
            diagnostic = FailureDiagnostic(
                failure_stage=FailureStage.PLANNER_VALIDATION,
                error_type="EmptyQuestionError",
                sanitized_message="A non-empty scouting question is required.",
                planner_status=PlannerStatus.ERROR.value,
                provider=self.provider,
                provider_response_received=False,
            )
            return PlannerResult(
                status=PlannerStatus.ERROR,
                provider=self.provider,
                model=self.model,
                prompt_version=PLANNER_PROMPT_VERSION,
                error_message="A non-empty scouting question is required.",
                diagnostic=diagnostic,
            )
        deterministic_decision = deterministic_request_decision(normalized_question)
        if deterministic_decision is not None:
            try:
                guarded_plan = ScoutPlan.model_validate(
                    build_scout_plan(
                        deterministic_decision,
                        question=normalized_question,
                    )
                )
            except (TypeError, ValueError) as exc:
                diagnostic = _with_planner_status(
                    diagnostic_from_exception(
                        exc,
                        default_stage=FailureStage.PLANNER_VALIDATION,
                        response_received=False,
                        provider=self.provider,
                    ).model_copy(
                        update={"failure_stage": FailureStage.PLANNER_VALIDATION}
                    ),
                    PlannerStatus.ERROR,
                    self.provider,
                )
                return PlannerResult(
                    status=PlannerStatus.ERROR,
                    provider=self.provider,
                    model=self.model,
                    prompt_version=PLANNER_PROMPT_VERSION,
                    provider_decision=deterministic_decision,
                    error_message="The deterministic plan failed planner validation.",
                    diagnostic=diagnostic,
                )
            return PlannerResult(
                status=PlannerStatus.SUCCESS,
                provider=self.provider,
                model=self.model,
                prompt_version=PLANNER_PROMPT_VERSION,
                provider_decision=deterministic_decision,
                plan=guarded_plan,
                response_received=False,
            )
        try:
            response = self._provider.create_plan(
                question=normalized_question,
                system_prompt=PLANNER_PROMPT,
            )
        except PlannerRefusalError as exc:
            diagnostic = _with_planner_status(
                exc.diagnostic
                or diagnostic_from_exception(
                    exc,
                    default_stage=FailureStage.STRUCTURED_PARSE,
                ),
                PlannerStatus.REFUSED,
                self.provider,
            )
            return PlannerResult(
                status=PlannerStatus.REFUSED,
                provider=self.provider,
                model=self.model,
                prompt_version=PLANNER_PROMPT_VERSION,
                error_message="The planning provider declined the request.",
                usage=_diagnostic_usage(diagnostic),
                diagnostic=diagnostic,
                response_received=diagnostic.provider_response_received,
                request_id=diagnostic.request_id,
                http_status=diagnostic.http_status,
                provider_attempts=exc.provider_attempts,
            )
        except PlannerProviderError as exc:
            diagnostic = _with_planner_status(
                exc.diagnostic
                or diagnostic_from_exception(
                    exc,
                    default_stage=FailureStage.PROVIDER,
                ),
                PlannerStatus.ERROR,
                self.provider,
            )
            return PlannerResult(
                status=PlannerStatus.ERROR,
                provider=self.provider,
                model=self.model,
                prompt_version=PLANNER_PROMPT_VERSION,
                error_message="The planning provider request failed.",
                usage=_diagnostic_usage(diagnostic),
                diagnostic=diagnostic,
                response_received=diagnostic.provider_response_received,
                request_id=diagnostic.request_id,
                http_status=diagnostic.http_status,
                provider_attempts=exc.provider_attempts,
            )
        except Exception as exc:  # noqa: BLE001 - provider implementations are untrusted
            diagnostic = _with_planner_status(
                diagnostic_from_exception(
                    exc,
                    default_stage=FailureStage.PROVIDER,
                    provider=self.provider,
                ),
                PlannerStatus.ERROR,
                self.provider,
            )
            return PlannerResult(
                status=PlannerStatus.ERROR,
                provider=self.provider,
                model=self.model,
                prompt_version=PLANNER_PROMPT_VERSION,
                error_message="The planning provider request failed.",
                usage=_diagnostic_usage(diagnostic),
                diagnostic=diagnostic,
                response_received=diagnostic.provider_response_received,
                request_id=diagnostic.request_id,
                http_status=diagnostic.http_status,
                provider_attempts=1,
            )
        guard = apply_pure_methodology_guard(normalized_question, response.decision)
        try:
            validated_plan = ScoutPlan.model_validate(
                build_scout_plan(guard.decision, question=normalized_question)
            )
        except (TypeError, ValueError) as exc:
            return PlannerResult(
                status=PlannerStatus.ERROR,
                provider=self.provider,
                model=self.model,
                prompt_version=PLANNER_PROMPT_VERSION,
                error_message="The structured plan failed planner validation.",
                usage=response.usage,
                provider_decision=response.decision,
                diagnostic=_with_planner_status(
                    diagnostic_from_exception(
                        exc,
                        default_stage=FailureStage.PLANNER_VALIDATION,
                        response_received=response.response_received,
                        provider=self.provider,
                    ).model_copy(
                        update={
                            "failure_stage": FailureStage.PLANNER_VALIDATION,
                            "request_id": response.request_id,
                            "http_status": response.http_status,
                            "input_tokens": response.usage.input_tokens,
                            "cached_input_tokens": response.usage.cached_input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                    ),
                    PlannerStatus.ERROR,
                    self.provider,
                ),
                response_received=response.response_received,
                request_id=response.request_id,
                http_status=response.http_status,
                methodology_guard_applied=guard.applied,
                provider_attempts=response.provider_attempts,
            )
        return PlannerResult(
            status=PlannerStatus.SUCCESS,
            provider=self.provider,
            model=self.model,
            prompt_version=PLANNER_PROMPT_VERSION,
            provider_decision=response.decision,
            plan=validated_plan,
            usage=response.usage,
            response_received=response.response_received,
            request_id=response.request_id,
            http_status=response.http_status,
            methodology_guard_applied=guard.applied,
            fallback_attempted=response.fallback_attempted,
            fallback_provider=response.fallback_provider,
            fallback_model=response.fallback_model,
            fallback_reason=response.fallback_reason,
            fallback_success=response.fallback_success,
            provider_attempts=response.provider_attempts,
        )


def _with_planner_status(
    diagnostic: FailureDiagnostic | None,
    status: PlannerStatus,
    provider: str,
) -> FailureDiagnostic:
    if diagnostic is None:
        diagnostic = FailureDiagnostic(
            failure_stage=FailureStage.PROVIDER,
            error_type=PlannerProviderError.__name__,
            sanitized_message="The planning provider request failed.",
            provider=provider,
            provider_response_received=False,
        )
    return diagnostic.model_copy(
        update={
            "planner_status": status.value,
            "provider": diagnostic.provider or provider,
        }
    )


def _diagnostic_usage(diagnostic: FailureDiagnostic) -> PlannerUsage | None:
    if all(
        value is None
        for value in (
            diagnostic.input_tokens,
            diagnostic.cached_input_tokens,
            diagnostic.output_tokens,
            diagnostic.total_tokens,
        )
    ):
        return None
    return PlannerUsage(
        input_tokens=diagnostic.input_tokens,
        cached_input_tokens=diagnostic.cached_input_tokens,
        output_tokens=diagnostic.output_tokens,
        total_tokens=diagnostic.total_tokens,
    )
