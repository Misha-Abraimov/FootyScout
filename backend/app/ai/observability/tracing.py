"""Normalize canonical workflow diagnostics into the stable trace schema."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from app.ai.diagnostics import sanitize_error_message
from app.ai.grounding import EvidenceCategory, EvidenceRecord
from app.ai.observability.models import (
    AIScoutRunTrace,
    CitationRepairTrace,
    EntityResolutionTrace,
    ErrorTrace,
    PlannedToolTrace,
    ToolExecutionTrace,
    ValidationFindingTrace,
)
from app.ai.observability.pricing import EMPTY_PRICING_REGISTRY, PricingRegistry, estimate_cost_usd
from app.ai.schemas import ToolExecutionStatus

if TYPE_CHECKING:
    from app.ai.workflow import AIScoutWorkflowResult

_SENSITIVE_ARGUMENT_KEYS = (
    "api_key",
    "authorization",
    "credential",
    "header",
    "password",
    "prompt",
    "provider_response",
    "raw_response",
    "secret",
    "token",
)
_SAFE_STRING_LIMIT = 160


def build_run_trace(
    result: AIScoutWorkflowResult,
    *,
    case_id: str | None = None,
    environment: str = "development",
    timestamp_utc: datetime | None = None,
    pricing_registry: PricingRegistry = EMPTY_PRICING_REGISTRY,
) -> AIScoutRunTrace:
    """Build one privacy-bounded trace from the canonical completed result."""
    diagnostics = result.diagnostics
    timestamp = timestamp_utc or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    else:
        timestamp = timestamp.astimezone(UTC)

    records = result.evidence.records
    generated_ids = diagnostics.evidence_ids_generated
    supplied_ids = diagnostics.evidence_ids_supplied
    cited_ids = diagnostics.evidence_ids_cited
    categories = Counter(record.evidence_category.value for record in records)
    methodology_records = tuple(
        record for record in records if record.evidence_category is EvidenceCategory.METHODOLOGY
    )
    analytics_records = tuple(
        record for record in records if record.evidence_category is EvidenceCategory.ANALYTICS
    )

    planner_usage = diagnostics.planner_usage
    synthesis_usage = diagnostics.synthesis_usage
    planner_provider_calls = diagnostics.planner_provider_attempts
    synthesis_provider_calls = diagnostics.synthesis_provider_attempts
    total_input = _combined_usage(
        planner_usage.input_tokens if planner_usage else None,
        synthesis_usage.input_tokens if synthesis_usage else None,
        planner_provider_calls=planner_provider_calls,
        synthesis_provider_calls=synthesis_provider_calls,
    )
    total_output = _combined_usage(
        planner_usage.output_tokens if planner_usage else None,
        synthesis_usage.output_tokens if synthesis_usage else None,
        planner_provider_calls=planner_provider_calls,
        synthesis_provider_calls=synthesis_provider_calls,
    )
    total_tokens = _combined_usage(
        planner_usage.total_tokens if planner_usage else None,
        synthesis_usage.total_tokens if synthesis_usage else None,
        planner_provider_calls=planner_provider_calls,
        synthesis_provider_calls=synthesis_provider_calls,
    )

    planner_pricing = pricing_registry.find(diagnostics.provider, diagnostics.model)
    synthesis_pricing = pricing_registry.find(
        diagnostics.synthesis_provider,
        diagnostics.synthesis_model,
    )
    planner_cost = estimate_cost_usd(
        input_tokens=planner_usage.input_tokens if planner_usage else None,
        cached_input_tokens=(
            planner_usage.cached_input_tokens if planner_usage else None
        ),
        output_tokens=planner_usage.output_tokens if planner_usage else None,
        pricing=planner_pricing,
    )
    synthesis_cost = estimate_cost_usd(
        input_tokens=synthesis_usage.input_tokens if synthesis_usage else None,
        cached_input_tokens=(
            synthesis_usage.cached_input_tokens if synthesis_usage else None
        ),
        output_tokens=synthesis_usage.output_tokens if synthesis_usage else None,
        pricing=synthesis_pricing,
    )
    estimated_total = _combined_cost(
        planner_cost,
        synthesis_cost,
        planner_provider_calls=planner_provider_calls,
        synthesis_provider_calls=synthesis_provider_calls,
    )

    errors = _normalized_errors(result)
    provider_failure = any(
        error.error_type
        in {
            "planner_timeout",
            "planner_provider_error",
            "synthesis_timeout",
            "synthesis_provider_error",
            "web_timeout",
            "web_provider_error",
        }
        for error in errors
    )
    workflow_completed = diagnostics.terminal_status.value != "error"
    validation_rules = Counter(
        finding.rule for finding in diagnostics.validation_findings if finding.rule
    )
    timings = diagnostics.stage_timings_ms
    web_query = diagnostics.normalized_search_query
    web_executed = timings.get("web_search") is not None
    llm_provider_calls = planner_provider_calls + synthesis_provider_calls
    web_provider_calls = (
        diagnostics.web_provider_attempts if web_executed else 0
    )
    ai_latencies = tuple(
        value
        for value in (timings.get("planner"), timings.get("synthesis"))
        if value is not None
    )

    return AIScoutRunTrace(
        run_id=result.run_id,
        case_id=case_id,
        timestamp_utc=timestamp,
        environment=environment,
        validation_mode=diagnostics.validation_mode.value,
        question_hash=_hash_text(result.question),
        question_length=len(result.question),
        planner_provider=diagnostics.provider,
        planner_model=diagnostics.model,
        planner_prompt_version=diagnostics.planner_prompt_version,
        synthesis_provider=diagnostics.synthesis_provider,
        synthesis_model=diagnostics.synthesis_model,
        synthesis_prompt_version=diagnostics.synthesis_prompt_version,
        web_search_provider=(
            diagnostics.web_search_provider
            if timings.get("web_search") is not None
            else None
        ),
        planner_fallback_attempted=diagnostics.planner_fallback_attempted,
        planner_fallback_provider=diagnostics.planner_fallback_provider,
        planner_fallback_model=diagnostics.planner_fallback_model,
        planner_fallback_reason=diagnostics.planner_fallback_reason,
        planner_fallback_success=diagnostics.planner_fallback_success,
        primary_intent=diagnostics.planner_primary_intent,
        normalized_intent=diagnostics.normalized_intent,
        terminal_status=diagnostics.terminal_status.value,
        sufficiency_status=result.context.status.value if result.context else None,
        terminal_branch_reason=diagnostics.terminal_branch_reason,
        preparation_status=diagnostics.preparation_status,
        preparation_reason_code=diagnostics.preparation_reason_code,
        entity_resolutions=tuple(
            EntityResolutionTrace.model_validate(entity)
            for entity in diagnostics.resolved_entities
        ),
        tools_planned=(
            diagnostics.pre_normalization_tool_names
            or diagnostics.post_normalization_tool_names
        ),
        tools_executed=diagnostics.tools_executed,
        planned_tool_calls=_planned_calls(diagnostics.pre_normalization_tool_calls),
        normalized_tool_calls=_planned_calls(diagnostics.post_normalization_tool_calls),
        tool_executions=tuple(_tool_trace(record) for record in analytics_records),
        methodology_topics_requested=diagnostics.methodology_topics_requested,
        methodology_topics_retrieved=tuple(
            record.methodology_topic
            for record in methodology_records
            if record.methodology_topic is not None
        ),
        methodology_source_ids=_unique(
            source
            for record in methodology_records
            for source in record.methodology_sources
        ),
        methodology_retrieval_modes=_unique(
            str(record.provenance["retrieval"])
            for record in methodology_records
            if record.provenance.get("retrieval") is not None
        ),
        web_search_required=diagnostics.web_search_required,
        web_search_executed=(
            timings.get("web_search") is not None and diagnostics.web_failure_stage != "disabled"
        ),
        web_search_category=diagnostics.web_search_category,
        web_query_hash=_hash_text(web_query) if web_query else None,
        web_query_length=len(web_query) if web_query else None,
        publication_filter_start=diagnostics.publication_filter_start,
        content_max_age_hours=diagnostics.content_max_age_hours,
        evidence_max_age_hours=diagnostics.evidence_max_age_hours,
        web_result_count=diagnostics.returned_result_count,
        web_selected_result_count=diagnostics.selected_result_count,
        web_selected_domains=diagnostics.selected_domains,
        web_selected_publication_dates=diagnostics.selected_publication_dates,
        web_failure_stage=diagnostics.web_failure_stage,
        web_failure_type=diagnostics.web_error_type,
        degraded_due_to_web_failure=diagnostics.degraded_due_to_web_failure,
        web_provider_attempts=diagnostics.web_provider_attempts,
        web_retry_count=diagnostics.web_retry_count,
        web_rate_limit_events=diagnostics.web_rate_limit_events,
        analytics_evidence_count=categories[EvidenceCategory.ANALYTICS.value],
        methodology_evidence_count=categories[EvidenceCategory.METHODOLOGY.value],
        web_evidence_count=categories[EvidenceCategory.WEB.value],
        evidence_generated_count=len(generated_ids),
        evidence_supplied_count=len(supplied_ids),
        evidence_cited_count=len(cited_ids),
        unused_supplied_evidence_count=len(set(supplied_ids) - set(cited_ids)),
        uncited_generated_evidence_count=len(set(generated_ids) - set(cited_ids)),
        evidence_ids_generated=generated_ids,
        evidence_ids_supplied=supplied_ids,
        evidence_ids_cited=cited_ids,
        validation_outcome=(
            diagnostics.validation_outcome.value if diagnostics.validation_outcome else None
        ),
        validation_findings_count=diagnostics.validation_findings_count,
        validation_warning_count=diagnostics.validation_warning_count,
        validation_repair_count=diagnostics.validation_repair_count,
        validation_block_count=diagnostics.validation_block_count,
        validation_repair_applied=diagnostics.validation_repair_applied,
        validation_repair_strategy=diagnostics.validation_repair_strategy,
        validation_citation_repairs=tuple(
            CitationRepairTrace(
                original_id=repair.original_id,
                canonical_id=repair.canonical_id,
                strategy=repair.strategy,
            )
            for repair in diagnostics.validation_citation_repairs
        ),
        validation_rule_frequencies=dict(sorted(validation_rules.items())),
        validation_findings=tuple(
            ValidationFindingTrace(
                rule=finding.rule,
                error_code=finding.error_code,
                action=finding.action.value,
                line=finding.line,
                evidence_ids=finding.evidence_ids,
                evidence_categories=finding.evidence_categories,
            )
            for finding in diagnostics.validation_findings
        ),
        planner_input_tokens=planner_usage.input_tokens if planner_usage else None,
        planner_cached_input_tokens=(
            planner_usage.cached_input_tokens if planner_usage else None
        ),
        planner_output_tokens=planner_usage.output_tokens if planner_usage else None,
        planner_total_tokens=planner_usage.total_tokens if planner_usage else None,
        synthesis_input_tokens=synthesis_usage.input_tokens if synthesis_usage else None,
        synthesis_cached_input_tokens=(
            synthesis_usage.cached_input_tokens if synthesis_usage else None
        ),
        synthesis_output_tokens=synthesis_usage.output_tokens if synthesis_usage else None,
        synthesis_total_tokens=synthesis_usage.total_tokens if synthesis_usage else None,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_tokens,
        planner_provider_call_count=planner_provider_calls,
        synthesis_provider_call_count=synthesis_provider_calls,
        llm_provider_call_count=llm_provider_calls,
        web_provider_call_count=web_provider_calls,
        provider_call_count=llm_provider_calls + web_provider_calls,
        planner_latency_ms=timings.get("planner"),
        preparation_latency_ms=timings.get("preparation"),
        analytics_latency_ms=timings.get("analytics_execution"),
        methodology_latency_ms=timings.get("methodology_retrieval"),
        current_context_preparation_latency_ms=timings.get(
            "current_context_preparation"
        ),
        web_latency_ms=diagnostics.web_search_latency_ms,
        web_normalization_latency_ms=timings.get("web_normalization"),
        context_assembly_latency_ms=timings.get("context_assembly"),
        synthesis_latency_ms=timings.get("synthesis"),
        validation_latency_ms=timings.get("answer_validation"),
        total_ai_latency_ms=sum(ai_latencies) if ai_latencies else None,
        total_latency_ms=diagnostics.total_ms,
        estimated_planner_cost_usd=planner_cost,
        estimated_synthesis_cost_usd=synthesis_cost,
        estimated_total_cost_usd=estimated_total,
        pricing_version=pricing_registry.version,
        error_count=len(errors),
        error_types=_unique(error.error_type for error in errors),
        errors=errors,
        workflow_completed=workflow_completed,
        provider_failure=provider_failure,
        success=workflow_completed and not provider_failure,
    )


def _tool_trace(record: EvidenceRecord) -> ToolExecutionTrace:
    failed = record.execution_status is not ToolExecutionStatus.SUCCESS
    return ToolExecutionTrace(
        tool_name=record.tool_name.value,
        execution_order=record.execution_order,
        execution_status=record.execution_status.value,
        safe_argument_summary=_safe_value(record.normalized_arguments),
        result_count=_result_count(record.result),
        evidence_ids_produced=(record.evidence_id,),
        error_type=(f"tool_{record.execution_status.value}" if failed else None),
    )


def _planned_calls(calls: tuple[dict[str, Any], ...]) -> tuple[PlannedToolTrace, ...]:
    return tuple(
        PlannedToolTrace(
            tool_name=str(call["tool_name"]),
            selection_order=index,
            safe_argument_summary=_safe_value(call.get("arguments", {})),
        )
        for index, call in enumerate(calls, start=1)
    )


def _safe_value(value: Any, *, key: str | None = None) -> Any:
    if key and any(marker in key.lower() for marker in _SENSITIVE_ARGUMENT_KEYS):
        return "[redacted]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return normalized[:_SAFE_STRING_LIMIT]
    if isinstance(value, dict):
        return {
            str(item_key): _safe_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in list(value)[:20]]
    return str(value)[:_SAFE_STRING_LIMIT]


def _result_count(result: dict[str, Any] | None) -> int | None:
    if result is None:
        return None
    for key in ("items", "players", "recommendations", "results"):
        value = result.get(key)
        if isinstance(value, list):
            return len(value)
    total = result.get("total")
    return total if isinstance(total, int) and total >= 0 else None


def _normalized_errors(result: AIScoutWorkflowResult) -> tuple[ErrorTrace, ...]:
    diagnostics = result.diagnostics
    errors: list[ErrorTrace] = []
    if diagnostics.planner_error_type:
        kind = _planner_error_type(
            diagnostics.planner_failure_stage,
            diagnostics.planner_error_type,
        )
        errors.append(
            ErrorTrace(
                error_type=kind,
                stage=diagnostics.planner_failure_stage or "planner",
                safe_message=None,
                original_code=diagnostics.planner_error_type,
                validation_issues=diagnostics.planner_validation_issues,
            )
        )
    if diagnostics.web_failure_stage and diagnostics.web_failure_stage not in {
        "disabled",
        "no_relevant_results",
        "query_construction",
    }:
        kind = _provider_error_type("web", diagnostics.web_error_type or "web_error")
        errors.append(
            ErrorTrace(
                error_type=kind,
                stage="web",
                safe_message=_safe_message(diagnostics.web_error_message),
                original_code=diagnostics.web_error_type,
            )
        )
    if diagnostics.validation_block_count:
        errors.append(
            ErrorTrace(
                error_type="validation_block",
                stage="validation",
                original_code=diagnostics.validation_error_code,
            )
        )
    for record in result.evidence.records:
        if record.execution_status is not ToolExecutionStatus.SUCCESS:
            errors.append(
                ErrorTrace(
                    error_type="tool_error",
                    stage="analytics",
                    original_code=record.execution_status.value,
                )
            )
    existing_messages = {error.safe_message for error in errors if error.safe_message}
    for message in diagnostics.errors:
        safe_message = _safe_message(message)
        if safe_message in existing_messages:
            continue
        lowered = message.lower()
        if "synthesis failed" in lowered:
            kind = _provider_error_type("synthesis", message)
            stage = "synthesis"
        elif "methodology retrieval failed" in lowered:
            kind = "methodology_error"
            stage = "methodology"
        elif "web search failed" in lowered:
            if any(error.stage == "web" for error in errors):
                continue
            kind = _provider_error_type("web", message)
            stage = "web"
        elif diagnostics.validation_block_count:
            continue
        else:
            kind = "unexpected_workflow_error"
            stage = "workflow"
        errors.append(ErrorTrace(error_type=kind, stage=stage, safe_message=safe_message))
        existing_messages.add(safe_message)
    return tuple(errors)


def _provider_error_type(stage: str, value: str) -> str:
    return f"{stage}_timeout" if "timeout" in value.lower() else f"{stage}_provider_error"


def _planner_error_type(stage: str | None, value: str) -> str:
    if "timeout" in value.lower():
        return "planner_timeout"
    if stage == "structured_parse":
        return "planner_structured_parse_error"
    if stage == "planner_validation":
        return "planner_validation_error"
    return "planner_provider_error"


def _safe_message(message: str | None) -> str | None:
    if not message:
        return None
    return sanitize_error_message(message)[:500]


def _combined_usage(
    planner_value: int | None,
    synthesis_value: int | None,
    *,
    planner_provider_calls: int,
    synthesis_provider_calls: int,
) -> int | None:
    if planner_provider_calls > 1 or synthesis_provider_calls > 1:
        return None
    if planner_provider_calls and planner_value is None:
        return None
    if synthesis_provider_calls and synthesis_value is None:
        return None
    if not planner_provider_calls and not synthesis_provider_calls:
        return None
    return (planner_value or 0) + (synthesis_value or 0)


def _combined_cost(
    planner_cost: float | None,
    synthesis_cost: float | None,
    *,
    planner_provider_calls: int,
    synthesis_provider_calls: int,
) -> float | None:
    if planner_provider_calls > 1 or synthesis_provider_calls > 1:
        return None
    if planner_provider_calls and planner_cost is None:
        return None
    if synthesis_provider_calls and synthesis_cost is None:
        return None
    if not planner_provider_calls and not synthesis_provider_calls:
        return None
    return round((planner_cost or 0.0) + (synthesis_cost or 0.0), 12)


def _hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _unique(values: Any) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))
