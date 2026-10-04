"""Stable, privacy-bounded models for AI Scout operational traces."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ai.diagnostics import ValidationIssue

TRACE_SCHEMA_VERSION = "ai-scout-trace-v1"


class TraceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolExecutionTrace(TraceModel):
    tool_name: str
    execution_order: int = Field(ge=1)
    execution_status: str
    latency_ms: float | None = Field(default=None, ge=0)
    safe_argument_summary: dict[str, Any]
    result_count: int | None = Field(default=None, ge=0)
    evidence_ids_produced: tuple[str, ...] = ()
    error_type: str | None = None


class PlannedToolTrace(TraceModel):
    tool_name: str
    selection_order: int = Field(ge=1)
    safe_argument_summary: dict[str, Any]


class ErrorTrace(TraceModel):
    error_type: str
    stage: str
    safe_message: str | None = None
    original_code: str | None = None
    validation_issues: tuple[ValidationIssue, ...] = ()


class EntityResolutionTrace(TraceModel):
    entity_type: Literal["player", "team"]
    query: str
    status: str
    stable_id: int | None = None
    display_name: str | None = None
    position_group: str | None = None
    analytics_supported: bool | None = None


class ValidationFindingTrace(TraceModel):
    """Privacy-bounded validation finding retained in internal run artifacts."""

    rule: str
    error_code: str
    action: str
    line: int | None = Field(default=None, ge=1)
    evidence_ids: tuple[str, ...] = ()
    evidence_categories: tuple[str, ...] = ()


class CitationRepairTrace(TraceModel):
    """One bounded deterministic correction of an emitted citation ID."""

    original_id: str
    canonical_id: str
    strategy: str


class AIScoutRunTrace(TraceModel):
    """One completed workflow run, normalized for local JSONL aggregation."""

    trace_schema_version: Literal["ai-scout-trace-v1"] = TRACE_SCHEMA_VERSION
    run_id: str
    case_id: str | None = None
    timestamp_utc: datetime
    environment: str
    validation_mode: str
    question_hash: str
    question_length: int = Field(ge=0)

    planner_provider: str
    planner_model: str
    planner_prompt_version: str
    synthesis_provider: str
    synthesis_model: str
    synthesis_prompt_version: str
    web_search_provider: str | None = None
    planner_fallback_attempted: bool = False
    planner_fallback_provider: str | None = None
    planner_fallback_model: str | None = None
    planner_fallback_reason: str | None = None
    planner_fallback_success: bool = False

    primary_intent: str | None = None
    normalized_intent: str | None = None
    terminal_status: str
    sufficiency_status: str | None = None
    terminal_branch_reason: str
    preparation_status: str | None = None
    preparation_reason_code: str | None = None
    entity_resolutions: tuple[EntityResolutionTrace, ...] = ()
    tools_planned: tuple[str, ...] = ()
    tools_executed: tuple[str, ...] = ()
    planned_tool_calls: tuple[PlannedToolTrace, ...] = ()
    normalized_tool_calls: tuple[PlannedToolTrace, ...] = ()
    tool_executions: tuple[ToolExecutionTrace, ...] = ()

    methodology_topics_requested: tuple[str, ...] = ()
    methodology_topics_retrieved: tuple[str, ...] = ()
    methodology_source_ids: tuple[str, ...] = ()
    methodology_retrieval_modes: tuple[str, ...] = ()

    web_search_required: bool = False
    web_search_executed: bool = False
    web_search_category: str | None = None
    web_query_hash: str | None = None
    web_query_length: int | None = Field(default=None, ge=0)
    publication_filter_start: datetime | None = None
    content_max_age_hours: int | None = Field(default=None, ge=1)
    evidence_max_age_hours: int | None = Field(default=None, ge=1)
    web_result_count: int = Field(default=0, ge=0)
    web_selected_result_count: int = Field(default=0, ge=0)
    web_selected_domains: tuple[str, ...] = ()
    web_selected_publication_dates: tuple[str | None, ...] = ()
    web_failure_stage: str | None = None
    web_failure_type: str | None = None
    degraded_due_to_web_failure: bool = False
    web_provider_attempts: int = Field(default=0, ge=0)
    web_retry_count: int = Field(default=0, ge=0)
    web_rate_limit_events: int = Field(default=0, ge=0)

    analytics_evidence_count: int = Field(ge=0)
    methodology_evidence_count: int = Field(ge=0)
    web_evidence_count: int = Field(ge=0)
    evidence_generated_count: int = Field(ge=0)
    evidence_supplied_count: int = Field(ge=0)
    evidence_cited_count: int = Field(ge=0)
    unused_supplied_evidence_count: int = Field(ge=0)
    uncited_generated_evidence_count: int = Field(ge=0)
    evidence_ids_generated: tuple[str, ...] = ()
    evidence_ids_supplied: tuple[str, ...] = ()
    evidence_ids_cited: tuple[str, ...] = ()

    validation_outcome: str | None = None
    validation_findings_count: int = Field(ge=0)
    validation_warning_count: int = Field(ge=0)
    validation_repair_count: int = Field(ge=0)
    validation_block_count: int = Field(ge=0)
    validation_repair_applied: bool = False
    validation_repair_strategy: str | None = None
    validation_citation_repairs: tuple[CitationRepairTrace, ...] = ()
    validation_rule_frequencies: dict[str, int] = Field(default_factory=dict)
    validation_findings: tuple[ValidationFindingTrace, ...] = ()

    planner_input_tokens: int | None = Field(default=None, ge=0)
    planner_cached_input_tokens: int | None = Field(default=None, ge=0)
    planner_output_tokens: int | None = Field(default=None, ge=0)
    planner_total_tokens: int | None = Field(default=None, ge=0)
    synthesis_input_tokens: int | None = Field(default=None, ge=0)
    synthesis_cached_input_tokens: int | None = Field(default=None, ge=0)
    synthesis_output_tokens: int | None = Field(default=None, ge=0)
    synthesis_total_tokens: int | None = Field(default=None, ge=0)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    planner_provider_call_count: int = Field(default=0, ge=0)
    synthesis_provider_call_count: int = Field(default=0, ge=0)
    llm_provider_call_count: int = Field(default=0, ge=0)
    web_provider_call_count: int = Field(default=0, ge=0)
    provider_call_count: int = Field(default=0, ge=0)

    planner_latency_ms: float | None = Field(default=None, ge=0)
    preparation_latency_ms: float | None = Field(default=None, ge=0)
    analytics_latency_ms: float | None = Field(default=None, ge=0)
    methodology_latency_ms: float | None = Field(default=None, ge=0)
    current_context_preparation_latency_ms: float | None = Field(default=None, ge=0)
    web_latency_ms: float | None = Field(default=None, ge=0)
    web_normalization_latency_ms: float | None = Field(default=None, ge=0)
    context_assembly_latency_ms: float | None = Field(default=None, ge=0)
    synthesis_latency_ms: float | None = Field(default=None, ge=0)
    validation_latency_ms: float | None = Field(default=None, ge=0)
    total_ai_latency_ms: float | None = Field(default=None, ge=0)
    total_latency_ms: float = Field(ge=0)

    estimated_planner_cost_usd: float | None = Field(default=None, ge=0)
    estimated_synthesis_cost_usd: float | None = Field(default=None, ge=0)
    estimated_total_cost_usd: float | None = Field(default=None, ge=0)
    pricing_version: str | None = None

    error_count: int = Field(ge=0)
    error_types: tuple[str, ...] = ()
    errors: tuple[ErrorTrace, ...] = ()
    workflow_completed: bool
    provider_failure: bool
    success: bool
