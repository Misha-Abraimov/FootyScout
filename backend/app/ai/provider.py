"""Small provider boundary for structured AI Scout planning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from anthropic import Anthropic
from anthropic import APIError as AnthropicAPIError
from openai import APIError as OpenAIAPIError
from openai import OpenAI
from pydantic import ValidationError

from app.ai.diagnostics import (
    FailureDiagnostic,
    FailureStage,
    diagnostic_from_exception,
    sanitize_error_message,
)
from app.ai.observability.telemetry import set_current_span_attributes, start_span
from app.ai.provider_schemas import LLMPlannerDecision
from app.ai.reliability import is_transient_provider_error

logger = logging.getLogger(__name__)


class PlannerProviderError(RuntimeError):
    """Safe provider failure without exposing provider payloads or secrets."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: FailureDiagnostic | None = None,
        provider_attempts: int = 1,
    ) -> None:
        super().__init__(sanitize_error_message(message))
        self.diagnostic = diagnostic
        self.provider_attempts = provider_attempts


class PlannerRefusalError(PlannerProviderError):
    """The provider declined to return a structured plan."""


@dataclass(frozen=True)
class PlannerUsage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ProviderPlanResponse:
    decision: LLMPlannerDecision
    usage: PlannerUsage
    response_received: bool = True
    request_id: str | None = None
    http_status: int | None = None
    fallback_attempted: bool = False
    fallback_provider: str | None = None
    fallback_model: str | None = None
    fallback_reason: str | None = None
    fallback_success: bool = False
    # Application-initiated provider SDK requests, including a fallback request.
    # Opaque transport retries performed inside a vendor SDK are not observable here.
    provider_attempts: int = 1


class PlannerProvider(Protocol):
    provider: str
    model: str

    def create_plan(self, *, question: str, system_prompt: str) -> ProviderPlanResponse:
        """Return one already-validated structured plan."""


class FallbackPlannerProvider:
    """Use one compatible fallback only after a transient primary failure."""

    def __init__(self, primary: PlannerProvider, fallback: PlannerProvider) -> None:
        if primary.provider == fallback.provider and primary.model == fallback.model:
            raise ValueError("Planner fallback must differ from the primary provider/model.")
        self._primary = primary
        self._fallback = fallback
        self.provider = primary.provider
        self.model = primary.model

    def create_plan(self, *, question: str, system_prompt: str) -> ProviderPlanResponse:
        try:
            return self._primary.create_plan(question=question, system_prompt=system_prompt)
        except PlannerProviderError as exc:
            if not _is_transient_planner_failure(exc):
                raise
            reason = (
                exc.diagnostic.error_type
                if exc.diagnostic is not None
                else type(exc).__name__
            )
            with start_span(
                "ai_scout.planner_fallback",
                {
                    "ai.fallback.attempted": True,
                    "ai.fallback.provider": self._fallback.provider,
                    "ai.fallback.reason": reason,
                },
            ):
                try:
                    response = self._fallback.create_plan(
                        question=question,
                        system_prompt=system_prompt,
                    )
                except PlannerProviderError as fallback_error:
                    fallback_error.provider_attempts += exc.provider_attempts
                    raise
                set_current_span_attributes({"ai.fallback.success": True})
            return ProviderPlanResponse(
                decision=response.decision,
                usage=response.usage,
                response_received=response.response_received,
                request_id=response.request_id,
                http_status=response.http_status,
                fallback_attempted=True,
                fallback_provider=self._fallback.provider,
                fallback_model=self._fallback.model,
                fallback_reason=reason,
                fallback_success=True,
                provider_attempts=(
                    exc.provider_attempts + response.provider_attempts
                ),
            )


def _is_transient_planner_failure(exc: PlannerProviderError) -> bool:
    diagnostic = exc.diagnostic
    if diagnostic is None:
        return is_transient_provider_error(exc)
    if diagnostic.http_status == 429 or (
        diagnostic.http_status is not None and 500 <= diagnostic.http_status <= 599
    ):
        return True
    lowered = diagnostic.error_type.casefold()
    return any(marker in lowered for marker in ("timeout", "connection", "connecterror"))


class OpenAIPlannerProvider:
    """Official OpenAI Responses API implementation of the planner boundary."""

    provider = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        max_output_tokens: int = 4096,
        reasoning_effort: str = "low",
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when AI_SCOUT_PROVIDER=openai.")
        self._api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def create_plan(self, *, question: str, system_prompt: str) -> ProviderPlanResponse:
        raw_response = None
        try:
            raw_response = self._client.responses.with_raw_response.parse(
                model=self.model,
                max_output_tokens=getattr(self, "max_output_tokens", 4096),
                reasoning={
                    "effort": getattr(self, "reasoning_effort", "low")
                },
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                text_format=LLMPlannerDecision,
            )
            response = raw_response.parse()
        except ValidationError as exc:
            logger.warning(
                "Planner structured output failed validation: %s",
                _safe_planner_validation_shape(exc),
            )
            usage = _openai_raw_response_usage(raw_response)
            raise PlannerProviderError(
                "The provider response failed structured-plan parsing.",
                diagnostic=diagnostic_from_exception(
                    exc,
                    default_stage=FailureStage.STRUCTURED_PARSE,
                    response_received=raw_response is not None,
                    provider=self.provider,
                    known_secrets=(self._api_key,),
                ).model_copy(
                    update={
                        "request_id": _raw_response_request_id(raw_response),
                        "http_status": _raw_response_http_status(raw_response),
                        "input_tokens": usage.input_tokens,
                        "cached_input_tokens": usage.cached_input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                ),
            ) from exc
        except OpenAIAPIError as exc:
            raise PlannerProviderError(
                "The planning provider request failed.",
                diagnostic=diagnostic_from_exception(
                    exc,
                    default_stage=FailureStage.PROVIDER,
                    provider=self.provider,
                    known_secrets=(self._api_key,),
                ),
            ) from exc
        except Exception as exc:
            raise PlannerProviderError(
                "The provider response could not be parsed into a planner decision.",
                diagnostic=diagnostic_from_exception(
                    exc,
                    default_stage=FailureStage.STRUCTURED_PARSE,
                    provider=self.provider,
                    known_secrets=(self._api_key,),
                ),
            ) from exc

        provider_decision = getattr(response, "output_parsed", None)
        if provider_decision is None:
            usage = _usage(response)
            raise PlannerRefusalError(
                "The planning provider did not return a structured plan.",
                diagnostic=FailureDiagnostic(
                    failure_stage=FailureStage.STRUCTURED_PARSE,
                    error_type=PlannerRefusalError.__name__,
                    sanitized_message=(
                        "The OpenAI response did not contain a parsed planner decision."
                    ),
                    provider=self.provider,
                    provider_response_received=True,
                    request_id=getattr(response, "_request_id", None),
                    input_tokens=usage.input_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )
        return ProviderPlanResponse(
            decision=provider_decision,
            usage=_usage(response),
            request_id=getattr(response, "_request_id", None),
        )


class AnthropicPlannerProvider:
    """Official Anthropic Messages API implementation of the planner boundary."""

    provider = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float | None,
        max_retries: int,
        max_tokens: int = 4096,
    ) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when AI_SCOUT_PROVIDER=anthropic.")
        self._api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        if timeout_seconds is None:
            self._client = Anthropic(
                api_key=api_key,
                max_retries=max_retries,
            )
        else:
            self._client = Anthropic(
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )

    def create_plan(self, *, question: str, system_prompt: str) -> ProviderPlanResponse:
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": question}],
                output_format=LLMPlannerDecision,
            )
        except ValidationError as exc:
            raise PlannerProviderError(
                "The provider response failed structured-plan parsing.",
                diagnostic=diagnostic_from_exception(
                    exc,
                    default_stage=FailureStage.STRUCTURED_PARSE,
                    provider=self.provider,
                    known_secrets=(self._api_key,),
                ),
            ) from exc
        except AnthropicAPIError as exc:
            raise PlannerProviderError(
                "The planning provider request failed.",
                diagnostic=diagnostic_from_exception(
                    exc,
                    default_stage=FailureStage.PROVIDER,
                    provider=self.provider,
                    known_secrets=(self._api_key,),
                ),
            ) from exc
        except Exception as exc:
            raise PlannerProviderError(
                "The provider response could not be parsed into a planner decision.",
                diagnostic=diagnostic_from_exception(
                    exc,
                    default_stage=FailureStage.STRUCTURED_PARSE,
                    provider=self.provider,
                    known_secrets=(self._api_key,),
                ),
            ) from exc

        provider_decision = getattr(response, "parsed_output", None)
        usage = _anthropic_usage(response)
        request_id = getattr(response, "_request_id", None)
        if provider_decision is None:
            raise PlannerRefusalError(
                "The planning provider did not return a structured plan.",
                diagnostic=FailureDiagnostic(
                    failure_stage=FailureStage.STRUCTURED_PARSE,
                    error_type=PlannerRefusalError.__name__,
                    sanitized_message=(
                        "The Anthropic response did not contain a parsed planner decision."
                    ),
                    provider=self.provider,
                    provider_response_received=True,
                    request_id=request_id,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )
        return ProviderPlanResponse(
            decision=provider_decision,
            usage=usage,
            request_id=request_id,
        )


def _usage(response: object) -> PlannerUsage:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    return PlannerUsage(
        input_tokens=getattr(usage, "input_tokens", None),
        cached_input_tokens=getattr(input_details, "cached_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _openai_raw_response_usage(raw_response: object | None) -> PlannerUsage:
    """Extract only allowlisted usage fields from a completed raw HTTP response."""
    if raw_response is None:
        return PlannerUsage()
    http_response = getattr(raw_response, "http_response", None)
    json_loader = getattr(http_response, "json", None) or getattr(
        raw_response, "json", None
    )
    if not callable(json_loader):
        return PlannerUsage()
    try:
        payload = json_loader()
    except Exception:  # noqa: BLE001 - telemetry must not mask the parse failure
        return PlannerUsage()
    if not isinstance(payload, dict):
        return PlannerUsage()
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return PlannerUsage()
    details = usage.get("input_tokens_details")
    if not isinstance(details, dict):
        details = {}
    return PlannerUsage(
        input_tokens=_nonnegative_int(usage.get("input_tokens")),
        cached_input_tokens=_nonnegative_int(details.get("cached_tokens")),
        output_tokens=_nonnegative_int(usage.get("output_tokens")),
        total_tokens=_nonnegative_int(usage.get("total_tokens")),
    )


def _raw_response_request_id(raw_response: object | None) -> str | None:
    request_id = getattr(raw_response, "request_id", None)
    return str(request_id) if request_id is not None else None


def _raw_response_http_status(raw_response: object | None) -> int | None:
    status_code = getattr(raw_response, "status_code", None)
    if status_code is None:
        http_response = getattr(raw_response, "http_response", None)
        status_code = getattr(http_response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _anthropic_usage(response: object) -> PlannerUsage:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = (
        input_tokens + output_tokens
        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else None
    )
    return PlannerUsage(
        input_tokens=input_tokens,
        cached_input_tokens=getattr(usage, "cache_read_input_tokens", None),
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _safe_planner_validation_shape(exception: ValidationError) -> dict[str, object]:
    """Return allowlisted planner structure without names or message contents."""
    for error in exception.errors(
        include_url=False,
        include_context=False,
        include_input=True,
    ):
        payload = error.get("input")
        if not isinstance(payload, dict):
            continue
        primary_player_id = payload.get("primary_player_id")
        return {
            "decision": str(payload.get("decision", ""))[:40],
            "intent": str(payload.get("intent", ""))[:80],
            "primary_player_reference": bool(
                str(payload.get("primary_player_name", "")).strip()
                or isinstance(primary_player_id, int)
                and primary_player_id > 0
            ),
            "nonempty_control_fields": tuple(
                field
                for field in (
                    "clarification_message",
                    "unsupported_reason",
                    "limitation",
                )
                if isinstance(payload.get(field), str) and payload[field].strip()
            ),
        }
    return {"structured_input_available": False}
