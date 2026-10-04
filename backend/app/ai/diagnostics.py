"""Sanitized, serializable diagnostics for internal AI Scout evaluation failures."""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class FailureStage(str, Enum):
    PROVIDER = "provider"
    STRUCTURED_PARSE = "structured_parse"
    PLANNER_VALIDATION = "planner_validation"
    NORMALIZATION = "normalization"
    ENTITY_RESOLUTION = "entity_resolution"
    EXECUTION = "execution"


class ValidationIssue(BaseModel):
    """Safe Pydantic issue metadata without input values or provider payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    location: tuple[str, ...] = ()
    error_type: str


class FailureDiagnostic(BaseModel):
    """Safe failure context; never includes requests, headers, or response bodies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str | None = None
    failure_stage: FailureStage
    error_type: str
    sanitized_message: str = Field(max_length=1000)
    planner_status: str | None = None
    run_status: str | None = None
    provider: str | None = None
    provider_response_received: bool
    request_id: str | None = None
    http_status: int | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    validation_issues: tuple[ValidationIssue, ...] = ()


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(openai_api_key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(anthropic_api_key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(api[_ -]?key\s*(?:is|[:=])\s*)[^\s,;]+"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{8,}\b"),
)


def sanitize_error_message(
    message: object,
    *,
    known_secrets: Iterable[str | None] = (),
) -> str:
    """Redact common credential forms, flatten control whitespace, and bound output."""
    sanitized = " ".join(str(message).split())
    for secret in known_secrets:
        if secret:
            sanitized = sanitized.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub(_redact_match, sanitized)
    return sanitized[:1000] or "No diagnostic message was provided."


def diagnostic_from_exception(
    exception: Exception,
    *,
    default_stage: FailureStage,
    response_received: bool | None = None,
    provider: str | None = None,
    known_secrets: Iterable[str | None] = (),
) -> FailureDiagnostic:
    """Extract only safe SDK/Pydantic metadata from an exception."""
    response = getattr(exception, "response", None)
    stage = (
        FailureStage.STRUCTURED_PARSE if isinstance(exception, ValidationError) else default_stage
    )
    request_id = getattr(exception, "request_id", None)
    if request_id is None and response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            request_id = headers.get("request-id") or headers.get("x-request-id")
    http_status = getattr(exception, "status_code", None)
    if http_status is None and response is not None:
        http_status = getattr(response, "status_code", None)
    validation_issues = (
        _validation_issues(exception) if isinstance(exception, ValidationError) else ()
    )
    return FailureDiagnostic(
        failure_stage=stage,
        error_type=type(exception).__name__,
        sanitized_message=sanitize_error_message(exception, known_secrets=known_secrets),
        provider=provider,
        provider_response_received=(
            response is not None if response_received is None else response_received
        ),
        request_id=str(request_id) if request_id is not None else None,
        http_status=http_status if isinstance(http_status, int) else None,
        validation_issues=validation_issues,
    )


def _validation_issues(exception: ValidationError) -> tuple[ValidationIssue, ...]:
    """Extract bounded Pydantic locations/types and deliberately omit values."""
    issues: list[ValidationIssue] = []
    for error in exception.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[:20]:
        location = tuple(str(segment)[:120] for segment in error.get("loc", ()))
        error_type = sanitize_error_message(error.get("type", "validation_error"))[:120]
        issues.append(ValidationIssue(location=location, error_type=error_type))
    return tuple(issues)


def _redact_match(match: re.Match[str]) -> str:
    prefix = match.group(1) if match.lastindex else ""
    return f"{prefix}[REDACTED]"
