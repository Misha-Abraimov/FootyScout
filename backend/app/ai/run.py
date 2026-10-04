"""Internal Phase 2 orchestration for one structured AI Scout run."""

from __future__ import annotations

from enum import Enum
from time import perf_counter
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.ai.diagnostics import FailureDiagnostic, FailureStage, sanitize_error_message
from app.ai.executor import PlanExecutionStatus, execute_plan
from app.ai.grounding import EvidenceRecord
from app.ai.plan_normalizer import prepare_plan
from app.ai.planner import PlannerStatus, ScoutPlanner
from app.ai.prompts import PLANNER_PROMPT_VERSION
from app.ai.provider import PlannerUsage
from app.ai.provider_schemas import LLMPlannerDecision
from app.ai.schemas import (
    NormalizedPlan,
    PlanPreparationStatus,
    PlayerResolution,
    ScoutIntent,
    ScoutPlan,
    TeamResolution,
)


class AIScoutRunStatus(str, Enum):
    COMPLETED = "completed"
    CLARIFICATION_REQUIRED = "clarification_required"
    NOT_FOUND = "not_found"
    UNSUPPORTED = "unsupported"
    INVALID_PLAN = "invalid_plan"
    PLANNER_REFUSED = "planner_refused"
    PLANNER_ERROR = "planner_error"
    EXECUTION_STOPPED = "execution_stopped"


class UsageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int | None = None
    cached_input_tokens: int | None = Field(default=None, exclude=True)
    output_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_provider(cls, usage: PlannerUsage | None) -> UsageMetadata | None:
        if usage is None:
            return None
        return cls(
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )


class LatencyMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    planning_ms: float = Field(ge=0)
    normalization_ms: float = Field(ge=0)
    execution_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class AIScoutRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    question: str
    prompt_version: str
    provider: str
    model: str
    status: AIScoutRunStatus
    planner_status: PlannerStatus
    provider_decision: LLMPlannerDecision | None = None
    intent: ScoutIntent | None = None
    planned_plan: ScoutPlan | None = None
    normalized_plan: NormalizedPlan | None = None
    clarification_required: bool = False
    clarification_message: str | None = None
    player_resolutions: tuple[PlayerResolution, ...] = ()
    team_resolutions: tuple[TeamResolution, ...] = ()
    evidence: tuple[EvidenceRecord, ...] = ()
    limitations: tuple[str, ...] = ()
    unsupported_reason: str | None = None
    error_message: str | None = None
    failure: FailureDiagnostic | None = None
    usage: UsageMetadata | None = None
    latency: LatencyMetadata


def run_ai_scout(
    session: Session,
    *,
    question: str,
    planner: ScoutPlanner,
) -> AIScoutRunResult:
    """Plan once, validate once, execute once, and return structured evidence."""
    run_id = str(uuid4())
    started = perf_counter()
    planning_started = perf_counter()
    planned = planner.plan(question)
    planning_ms = _elapsed_ms(planning_started)

    if planned.status is not PlannerStatus.SUCCESS or planned.plan is None:
        status = (
            AIScoutRunStatus.PLANNER_REFUSED
            if planned.status is PlannerStatus.REFUSED
            else AIScoutRunStatus.PLANNER_ERROR
        )
        return _run_result(
            run_id=run_id,
            question=question,
            planner=planner,
            planner_status=planned.status,
            status=status,
            started=started,
            planning_ms=planning_ms,
            error_message=planned.error_message,
            provider_decision=planned.provider_decision,
            usage=UsageMetadata.from_provider(planned.usage),
            failure=_run_failure(
                planned.diagnostic,
                planner_status=planned.status,
                run_status=status,
                usage=planned.usage,
                response_received=planned.response_received,
                request_id=planned.request_id,
                http_status=planned.http_status,
            ),
        )

    normalization_started = perf_counter()
    preparation = prepare_plan(session, planned.plan)
    normalization_ms = _elapsed_ms(normalization_started)
    base = {
        "run_id": run_id,
        "question": question,
        "planner": planner,
        "planner_status": planned.status,
        "provider_decision": planned.provider_decision,
        "intent": planned.plan.intent,
        "planned_plan": planned.plan,
        "started": started,
        "planning_ms": planning_ms,
        "normalization_ms": normalization_ms,
        "player_resolutions": tuple(preparation.player_resolutions),
        "team_resolutions": tuple(preparation.team_resolutions),
        "limitations": tuple(preparation.limitations),
        "usage": UsageMetadata.from_provider(planned.usage),
    }
    if preparation.status is not PlanPreparationStatus.READY:
        status_map = {
            PlanPreparationStatus.CLARIFICATION_REQUIRED: (AIScoutRunStatus.CLARIFICATION_REQUIRED),
            PlanPreparationStatus.NOT_FOUND: AIScoutRunStatus.NOT_FOUND,
            PlanPreparationStatus.UNSUPPORTED: AIScoutRunStatus.UNSUPPORTED,
            PlanPreparationStatus.INVALID: AIScoutRunStatus.INVALID_PLAN,
        }
        run_status = status_map[preparation.status]
        stage = (
            FailureStage.NORMALIZATION
            if preparation.status is PlanPreparationStatus.INVALID
            else FailureStage.ENTITY_RESOLUTION
        )
        diagnostic_message = (
            preparation.error_message
            or preparation.unsupported_reason
            or preparation.clarification_message
            or "Plan preparation did not produce an executable plan."
        )
        return _run_result(
            **base,
            status=run_status,
            clarification_required=(
                preparation.status is PlanPreparationStatus.CLARIFICATION_REQUIRED
            ),
            clarification_message=preparation.clarification_message,
            unsupported_reason=preparation.unsupported_reason,
            error_message=preparation.error_message,
            failure=_run_failure(
                FailureDiagnostic(
                    failure_stage=stage,
                    error_type=f"PlanPreparation{preparation.status.value.title()}Error",
                    sanitized_message=sanitize_error_message(diagnostic_message),
                    provider=planner.provider,
                    provider_response_received=planned.response_received,
                    request_id=planned.request_id,
                    http_status=planned.http_status,
                ),
                planner_status=planned.status,
                run_status=run_status,
                usage=planned.usage,
                response_received=planned.response_received,
                request_id=planned.request_id,
                http_status=planned.http_status,
            ),
        )

    assert preparation.normalized_plan is not None
    execution_started = perf_counter()
    execution = execute_plan(
        session,
        preparation.normalized_plan,
        run_id=run_id,
    )
    execution_ms = _elapsed_ms(execution_started)
    run_status = (
        AIScoutRunStatus.COMPLETED
        if execution.status is PlanExecutionStatus.COMPLETED
        else AIScoutRunStatus.EXECUTION_STOPPED
    )
    execution_failure = None
    if run_status is AIScoutRunStatus.EXECUTION_STOPPED:
        execution_failure = _run_failure(
            FailureDiagnostic(
                failure_stage=FailureStage.EXECUTION,
                error_type="ToolExecutionError",
                sanitized_message=(
                    f"Tool execution stopped on {execution.stopped_on_call_id or 'an unknown call'}."
                ),
                provider=planner.provider,
                provider_response_received=planned.response_received,
                request_id=planned.request_id,
                http_status=planned.http_status,
            ),
            planner_status=planned.status,
            run_status=run_status,
            usage=planned.usage,
            response_received=planned.response_received,
            request_id=planned.request_id,
            http_status=planned.http_status,
        )
    return _run_result(
        **base,
        status=run_status,
        normalized_plan=preparation.normalized_plan,
        evidence=execution.evidence.records,
        execution_ms=execution_ms,
        failure=execution_failure,
    )


def _run_result(
    *,
    run_id: str,
    question: str,
    planner: ScoutPlanner,
    planner_status: PlannerStatus,
    status: AIScoutRunStatus,
    started: float,
    planning_ms: float,
    normalization_ms: float = 0.0,
    execution_ms: float = 0.0,
    intent: ScoutIntent | None = None,
    provider_decision: LLMPlannerDecision | None = None,
    planned_plan: ScoutPlan | None = None,
    normalized_plan: NormalizedPlan | None = None,
    clarification_required: bool = False,
    clarification_message: str | None = None,
    player_resolutions: tuple[PlayerResolution, ...] = (),
    team_resolutions: tuple[TeamResolution, ...] = (),
    evidence: tuple[EvidenceRecord, ...] = (),
    limitations: tuple[str, ...] = (),
    unsupported_reason: str | None = None,
    error_message: str | None = None,
    failure: FailureDiagnostic | None = None,
    usage: UsageMetadata | None = None,
) -> AIScoutRunResult:
    return AIScoutRunResult(
        run_id=run_id,
        question=question,
        prompt_version=PLANNER_PROMPT_VERSION,
        provider=planner.provider,
        model=planner.model,
        status=status,
        planner_status=planner_status,
        provider_decision=provider_decision,
        intent=intent,
        planned_plan=planned_plan,
        normalized_plan=normalized_plan,
        clarification_required=clarification_required,
        clarification_message=clarification_message,
        player_resolutions=player_resolutions,
        team_resolutions=team_resolutions,
        evidence=evidence,
        limitations=limitations,
        unsupported_reason=unsupported_reason,
        error_message=error_message,
        failure=failure,
        usage=usage,
        latency=LatencyMetadata(
            planning_ms=planning_ms,
            normalization_ms=normalization_ms,
            execution_ms=execution_ms,
            total_ms=_elapsed_ms(started),
        ),
    )


def _elapsed_ms(started: float) -> float:
    return max(0.0, (perf_counter() - started) * 1000.0)


def _run_failure(
    diagnostic: FailureDiagnostic | None,
    *,
    planner_status: PlannerStatus,
    run_status: AIScoutRunStatus,
    usage: PlannerUsage | None,
    response_received: bool,
    request_id: str | None,
    http_status: int | None,
) -> FailureDiagnostic | None:
    if diagnostic is None:
        return None
    return diagnostic.model_copy(
        update={
            "planner_status": planner_status.value,
            "run_status": run_status.value,
            "provider": diagnostic.provider,
            "provider_response_received": (
                diagnostic.provider_response_received or response_received
            ),
            "request_id": diagnostic.request_id or request_id,
            "http_status": diagnostic.http_status or http_status,
            "input_tokens": (
                diagnostic.input_tokens
                if diagnostic.input_tokens is not None
                else getattr(usage, "input_tokens", None)
            ),
            "output_tokens": (
                diagnostic.output_tokens
                if diagnostic.output_tokens is not None
                else getattr(usage, "output_tokens", None)
            ),
            "total_tokens": (
                diagnostic.total_tokens
                if diagnostic.total_tokens is not None
                else getattr(usage, "total_tokens", None)
            ),
        }
    )
