"""Deterministic operational summaries for AI Scout JSONL traces."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

from pydantic import BaseModel, ConfigDict, Field

from app.ai.observability.models import AIScoutRunTrace


class AggregateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NumericSummary(AggregateModel):
    observed_runs: int = Field(ge=0)
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    max: float | None = None


class AIScoutTraceSummary(AggregateModel):
    trace_schema_versions: dict[str, int]
    total_runs: int = Field(ge=0)
    successful_runs: int = Field(ge=0)
    workflow_completed_runs: int = Field(ge=0)
    terminal_status_counts: dict[str, int]
    error_rate: float | None = None
    repair_rate: float | None = None
    block_rate: float | None = None

    total_latency: NumericSummary
    planner_latency: NumericSummary
    web_latency: NumericSummary
    synthesis_latency: NumericSummary

    total_tokens: int | None = None
    observed_total_input_tokens: int | None = None
    observed_total_output_tokens: int | None = None
    observed_total_tokens: int | None = None
    mean_tokens_per_run: float | None = None
    median_tokens_per_run: float | None = None
    planner_tokens: int | None = None
    synthesis_tokens: int | None = None
    total_llm_provider_calls: int = Field(default=0, ge=0)
    usage_observed_llm_calls: int = Field(default=0, ge=0)
    usage_coverage_rate: float | None = None
    usage_complete: bool = True
    total_web_provider_calls: int = Field(default=0, ge=0)
    total_provider_calls: int = Field(ge=0)
    mean_provider_calls_per_run: float | None = None

    total_estimated_cost_usd: float | None = None
    observed_estimated_cost_usd: float | None = None
    cost_observed_llm_calls: int = Field(default=0, ge=0)
    cost_coverage_rate: float | None = None
    cost_complete: bool = True
    mean_estimated_cost_per_run_usd: float | None = None
    median_estimated_cost_per_run_usd: float | None = None
    pricing_versions: dict[str, int]

    tool_execution_frequency: dict[str, int]
    mean_tools_per_run: float | None = None
    tool_failure_count: int = Field(ge=0)
    tool_failure_rate: float | None = None

    web_search_usage_rate: float | None = None
    mean_web_results_returned: float | None = None
    mean_web_results_selected: float | None = None
    web_failure_rate: float | None = None

    mean_evidence_generated: float | None = None
    mean_evidence_supplied: float | None = None
    mean_evidence_cited: float | None = None

    validation_outcome_counts: dict[str, int]
    validation_reached_runs: int = Field(ge=0)
    validation_pass_rate: float | None = None
    validation_rule_frequencies: dict[str, int]
    model_usage_counts: dict[str, int]
    prompt_version_counts: dict[str, int]


def load_jsonl_traces(path: str | Path) -> tuple[AIScoutRunTrace, ...]:
    traces: list[AIScoutRunTrace] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                traces.append(AIScoutRunTrace.model_validate_json(line))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid trace JSONL at line {line_number}: {exc}") from exc
    return tuple(traces)


def summarize_traces(traces: tuple[AIScoutRunTrace, ...] | list[AIScoutRunTrace]) -> AIScoutTraceSummary:
    items = tuple(traces)
    count = len(items)
    terminal_statuses = Counter(trace.terminal_status for trace in items)
    schema_versions = Counter(trace.trace_schema_version for trace in items)
    tool_frequency = Counter(
        tool.tool_name for trace in items for tool in trace.tool_executions
    )
    tool_failures = sum(
        tool.execution_status != "success"
        for trace in items
        for tool in trace.tool_executions
    )
    tool_executions = sum(len(trace.tool_executions) for trace in items)
    web_runs = tuple(trace for trace in items if trace.web_search_executed)
    validation_outcomes = Counter(_validation_bucket(trace) for trace in items)
    validation_reached_runs = count - validation_outcomes["not_reached"]
    validation_rules = Counter()
    for trace in items:
        validation_rules.update(trace.validation_rule_frequencies)
    model_usage = Counter()
    prompt_versions = Counter()
    for trace in items:
        model_usage[f"planner:{trace.planner_provider}:{trace.planner_model}"] += 1
        model_usage[f"synthesis:{trace.synthesis_provider}:{trace.synthesis_model}"] += 1
        prompt_versions[f"planner:{trace.planner_prompt_version}"] += 1
        prompt_versions[f"synthesis:{trace.synthesis_prompt_version}"] += 1

    planner_tokens = _complete_stage_sum(
        items,
        call_count_field="planner_provider_call_count",
        value_field="planner_total_tokens",
    )
    synthesis_tokens = _complete_stage_sum(
        items,
        call_count_field="synthesis_provider_call_count",
        value_field="synthesis_total_tokens",
    )
    total_llm_calls = sum(trace.llm_provider_call_count for trace in items)
    observed_usage = _observed_usage(items)
    observed_cost = _observed_cost(items)
    usage_complete = observed_usage.calls == total_llm_calls
    cost_complete = observed_cost.calls == total_llm_calls
    total_web_calls = sum(trace.web_provider_call_count for trace in items)
    total_tokens = (
        (planner_tokens or 0) + (synthesis_tokens or 0)
        if (total_llm_calls or _has_legacy_reported_usage(items))
        and planner_tokens is not None
        and synthesis_tokens is not None
        else None
    )
    per_run_tokens = [
        trace.total_tokens for trace in items if trace.total_tokens is not None
    ]
    costs = _complete_total_cost(items)
    pricing_versions = Counter(
        trace.pricing_version for trace in items if trace.pricing_version is not None
    )

    return AIScoutTraceSummary(
        trace_schema_versions=_sorted_counter(schema_versions),
        total_runs=count,
        successful_runs=sum(trace.success for trace in items),
        workflow_completed_runs=sum(trace.workflow_completed for trace in items),
        terminal_status_counts=_sorted_counter(terminal_statuses),
        error_rate=_rate(sum(bool(trace.errors) for trace in items), count),
        repair_rate=_rate(sum(trace.validation_repair_applied for trace in items), count),
        block_rate=_rate(sum(trace.validation_block_count > 0 for trace in items), count),
        total_latency=_numeric_summary([trace.total_latency_ms for trace in items]),
        planner_latency=_numeric_summary(
            [trace.planner_latency_ms for trace in items if trace.planner_latency_ms is not None]
        ),
        web_latency=_numeric_summary(
            [trace.web_latency_ms for trace in items if trace.web_latency_ms is not None]
        ),
        synthesis_latency=_numeric_summary(
            [trace.synthesis_latency_ms for trace in items if trace.synthesis_latency_ms is not None]
        ),
        total_tokens=total_tokens,
        observed_total_input_tokens=observed_usage.input_tokens,
        observed_total_output_tokens=observed_usage.output_tokens,
        observed_total_tokens=observed_usage.total_tokens,
        mean_tokens_per_run=(
            mean(per_run_tokens) if usage_complete and per_run_tokens else None
        ),
        median_tokens_per_run=(
            median(per_run_tokens) if usage_complete and per_run_tokens else None
        ),
        planner_tokens=planner_tokens,
        synthesis_tokens=synthesis_tokens,
        total_llm_provider_calls=total_llm_calls,
        usage_observed_llm_calls=observed_usage.calls,
        usage_coverage_rate=_rate(observed_usage.calls, total_llm_calls),
        usage_complete=usage_complete,
        total_web_provider_calls=total_web_calls,
        total_provider_calls=sum(trace.provider_call_count for trace in items),
        mean_provider_calls_per_run=(
            mean(trace.provider_call_count for trace in items) if items else None
        ),
        total_estimated_cost_usd=(round(sum(costs), 12) if costs else None),
        observed_estimated_cost_usd=round(observed_cost.cost, 12),
        cost_observed_llm_calls=observed_cost.calls,
        cost_coverage_rate=_rate(observed_cost.calls, total_llm_calls),
        cost_complete=cost_complete,
        mean_estimated_cost_per_run_usd=(mean(costs) if costs else None),
        median_estimated_cost_per_run_usd=(median(costs) if costs else None),
        pricing_versions=_sorted_counter(pricing_versions),
        tool_execution_frequency=_sorted_counter(tool_frequency),
        mean_tools_per_run=(tool_executions / count if count else None),
        tool_failure_count=tool_failures,
        tool_failure_rate=_rate(tool_failures, tool_executions),
        web_search_usage_rate=_rate(len(web_runs), count),
        mean_web_results_returned=(
            mean(trace.web_result_count for trace in web_runs) if web_runs else None
        ),
        mean_web_results_selected=(
            mean(trace.web_selected_result_count for trace in web_runs) if web_runs else None
        ),
        web_failure_rate=_rate(
            sum(trace.web_failure_stage is not None for trace in web_runs),
            len(web_runs),
        ),
        mean_evidence_generated=(
            mean(trace.evidence_generated_count for trace in items) if items else None
        ),
        mean_evidence_supplied=(
            mean(trace.evidence_supplied_count for trace in items) if items else None
        ),
        mean_evidence_cited=(
            mean(trace.evidence_cited_count for trace in items) if items else None
        ),
        validation_outcome_counts=_sorted_counter(validation_outcomes),
        validation_reached_runs=validation_reached_runs,
        validation_pass_rate=_rate(
            validation_outcomes["pass"],
            validation_reached_runs,
        ),
        validation_rule_frequencies=_sorted_counter(validation_rules),
        model_usage_counts=_sorted_counter(model_usage),
        prompt_version_counts=_sorted_counter(prompt_versions),
    )


def _validation_bucket(trace: AIScoutRunTrace) -> str:
    if trace.validation_outcome is None:
        return "not_reached"
    if trace.validation_block_count:
        return "block"
    if trace.validation_repair_count or trace.validation_repair_applied:
        return "repair"
    if trace.validation_warning_count:
        return "warn"
    return "pass"


def _complete_stage_sum(
    traces: tuple[AIScoutRunTrace, ...],
    *,
    call_count_field: str,
    value_field: str,
) -> int | None:
    """Sum one LLM stage only when every recorded call has reported usage."""
    values: list[int] = []
    any_calls = False
    for trace in traces:
        call_count = getattr(trace, call_count_field)
        value = getattr(trace, value_field)
        if not call_count:
            if value is not None:
                any_calls = True
                values.append(value)
            continue
        if call_count > 1:
            return None
        any_calls = True
        if value is None:
            return None
        values.append(value)
    return sum(values) if any_calls else 0


def _has_legacy_reported_usage(traces: tuple[AIScoutRunTrace, ...]) -> bool:
    """Recognize v1 traces written before stage-specific call counts existed."""
    return any(
        trace.planner_total_tokens is not None
        or trace.synthesis_total_tokens is not None
        for trace in traces
    )


def _complete_total_cost(
    traces: tuple[AIScoutRunTrace, ...],
) -> list[float] | None:
    """Retain no partial aggregate when any actual LLM call lacks a cost."""
    values: list[float] = []
    for trace in traces:
        if trace.estimated_total_cost_usd is not None:
            values.append(trace.estimated_total_cost_usd)
            continue
        if trace.llm_provider_call_count:
            return None
    return values


class _ObservedUsage:
    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0


class _ObservedCost:
    def __init__(self) -> None:
        self.calls = 0
        self.cost = 0.0


def _observed_usage(traces: tuple[AIScoutRunTrace, ...]) -> _ObservedUsage:
    """Sum complete recorded call usage without filling unobserved calls."""
    observed = _ObservedUsage()
    for trace in traces:
        for prefix in ("planner", "synthesis"):
            calls = getattr(trace, f"{prefix}_provider_call_count")
            values = (
                getattr(trace, f"{prefix}_input_tokens"),
                getattr(trace, f"{prefix}_output_tokens"),
                getattr(trace, f"{prefix}_total_tokens"),
            )
            if calls == 0 and any(value is not None for value in values):
                calls = 1  # Backward compatibility for pre-call-count traces.
            if not calls or any(value is None for value in values):
                continue
            observed.calls += 1
            observed.input_tokens += values[0]
            observed.output_tokens += values[1]
            observed.total_tokens += values[2]
    return observed


def _observed_cost(traces: tuple[AIScoutRunTrace, ...]) -> _ObservedCost:
    """Sum priced recorded calls while leaving missing calls visible in coverage."""
    observed = _ObservedCost()
    for trace in traces:
        for prefix in ("planner", "synthesis"):
            calls = getattr(trace, f"{prefix}_provider_call_count")
            value = getattr(trace, f"estimated_{prefix}_cost_usd")
            if calls == 0 and value is not None:
                calls = 1  # Backward compatibility for pre-call-count traces.
            if not calls or value is None:
                continue
            observed.calls += 1
            observed.cost += value
    return observed


def _numeric_summary(values: list[float]) -> NumericSummary:
    if not values:
        return NumericSummary(observed_runs=0)
    ordered = sorted(values)
    return NumericSummary(
        observed_runs=len(ordered),
        mean=mean(ordered),
        p50=median(ordered),
        p95=_nearest_rank(ordered, 0.95),
        max=ordered[-1],
    )


def _nearest_rank(ordered: list[float], percentile: float) -> float:
    """Nearest-rank percentile: sorted_values[ceil(p * n) - 1]."""
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))
