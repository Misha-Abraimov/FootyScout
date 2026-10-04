"""Bounded LangGraph orchestration for grounded AI Scout responses."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.ai.content_hygiene import (
    GroundingValidationError,
    has_substantive_claims,
    sanitize_user_facing_text,
)
from app.ai.context import (
    ContextStatus,
    SelectedContext,
    assemble_context,
    select_methodology_topics,
)
from app.ai.context.assembler import methodology_topics_already_present
from app.ai.diagnostics import ValidationIssue, sanitize_error_message
from app.ai.executor import PlanExecutionResult, PlanExecutionStatus, execute_plan
from app.ai.grounding import EvidenceLedger
from app.ai.methodology import (
    CuratedMethodologyService,
    MethodologyService,
    methodology_evidence,
)
from app.ai.observability.pricing import EMPTY_PRICING_REGISTRY, PricingRegistry
from app.ai.observability.sinks import NullTraceSink, TraceSink, record_trace_safely
from app.ai.observability.telemetry import (
    langsmith_graph_config,
    set_current_span_attributes,
    start_span,
)
from app.ai.observability.tracing import build_run_trace
from app.ai.plan_normalizer import prepare_plan
from app.ai.planner import PlannerResult, PlannerStatus, ScoutPlanner
from app.ai.prompts import PLANNER_PROMPT_VERSION
from app.ai.provider_schemas import LLMPlannerDecision
from app.ai.run import UsageMetadata
from app.ai.schemas import PlanPreparationResult, PlanPreparationStatus
from app.ai.synthesis import (
    SYNTHESIS_PROMPT_VERSION,
    AnswerSynthesizer,
    GroundedAnswerStatus,
    GroundedScoutAnswer,
    SynthesisResult,
    apply_no_qualifying_web_fallback,
    apply_role_fit_semantic_fallback,
    deterministic_governed_synthesis,
    validate_answer_with_policy,
)
from app.ai.validation import ValidationAction, ValidationMode, ValidationResolution
from app.ai.web import (
    CurrentContextCategory,
    CurrentContextRequest,
    WebSearchProvider,
    WebSearchProviderError,
    WebSearchRequest,
    WebSearchResponse,
)
from app.ai.web.currentness import build_search_query, detect_current_context
from app.ai.web.evidence import web_evidence
from app.ai.web.selection import select_web_results

_RAW_EVENT_ROW_REQUEST = re.compile(
    r"\b(?:every|all)\s+raw\s+(?:event|events|pass|passes|shot|shots|carry|carries)\b|"
    r"\braw\s+(?:event|events)\s+rows?\b",
    re.IGNORECASE,
)


class AIScoutGraphState(TypedDict, total=False):
    run_id: str
    question: str
    planner_result: PlannerResult
    preparation: PlanPreparationResult
    execution: PlanExecutionResult
    evidence: EvidenceLedger
    selected_context: SelectedContext
    context_status: ContextStatus
    synthesis_result: SynthesisResult
    final_answer: GroundedScoutAnswer
    stage_timings_ms: dict[str, float]
    errors: list[str]
    current_context_request: CurrentContextRequest
    web_search_response: WebSearchResponse
    web_search_latency_ms: float
    web_failure_stage: str
    web_error_type: str
    web_error_message: str
    selected_web_count: int
    web_provider_attempts: int
    web_retry_count: int
    web_rate_limit_events: int
    validation_error_code: str
    validation_failed_claim: str
    validation_failed_line: int
    validation_detected_evidence_ids: tuple[str, ...]
    validation_detected_evidence_categories: tuple[str, ...]
    validation_rule: str
    validation_outcome: str
    validation_findings: tuple[dict[str, object], ...]
    validation_warning_count: int
    validation_repair_count: int
    validation_block_count: int
    validation_repair_applied: bool
    validation_repair_strategy: str
    validation_citation_repairs: tuple[dict[str, str], ...]
    methodology_topics_requested: tuple[str, ...]
    synthesis_provider_attempts: int


class ValidationFindingDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    error_code: str
    action: ValidationAction
    line: int | None = None
    evidence_ids: tuple[str, ...] = ()
    evidence_categories: tuple[str, ...] = ()


class CitationRepairDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original_id: str
    canonical_id: str
    strategy: str


class WorkflowDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    planner_prompt_version: str
    synthesis_provider: str
    synthesis_model: str
    synthesis_prompt_version: str
    planner_usage: UsageMetadata | None = None
    synthesis_usage: UsageMetadata | None = None
    stage_timings_ms: dict[str, float]
    total_ms: float = Field(ge=0)
    tools_executed: tuple[str, ...] = ()
    evidence_ids_generated: tuple[str, ...] = ()
    evidence_ids_supplied: tuple[str, ...] = ()
    evidence_ids_cited: tuple[str, ...] = ()
    methodology_sources_derived: tuple[str, ...] = ()
    terminal_status: GroundedAnswerStatus
    planner_primary_intent: str | None = None
    planner_methodology_topic: str | None = None
    planner_player_reference: str | None = None
    planner_team_reference: str | None = None
    planner_requested_analyses: tuple[str, ...] = ()
    planner_status: str | None = None
    planner_failure_stage: str | None = None
    planner_error_type: str | None = None
    planner_response_received: bool = False
    planner_fallback_attempted: bool = False
    planner_fallback_provider: str | None = None
    planner_fallback_model: str | None = None
    planner_fallback_reason: str | None = None
    planner_fallback_success: bool = False
    planner_provider_attempts: int = 0
    synthesis_provider_attempts: int = 0
    planner_validation_issues: tuple[ValidationIssue, ...] = ()
    normalized_intent: str | None = None
    methodology_topics_requested: tuple[str, ...] = ()
    methodology_guard_applied: bool = False
    current_context_required: bool = False
    current_only_guard_applied: bool = False
    current_context_category: str | None = None
    analytics_requested_by_user: bool = False
    analytics_suppressed_for_current_only: bool = False
    pre_normalization_tool_names: tuple[str, ...] = ()
    post_normalization_tool_names: tuple[str, ...] = ()
    pre_normalization_tool_calls: tuple[dict[str, object], ...] = ()
    post_normalization_tool_calls: tuple[dict[str, object], ...] = ()
    terminal_branch_reason: str
    preparation_status: str | None = None
    preparation_reason_code: str | None = None
    resolved_entities: tuple[dict[str, object], ...] = ()
    web_search_enabled: bool = False
    web_search_required: bool = False
    web_search_reason: str | None = None
    web_search_category: str | None = None
    web_search_provider: str | None = None
    normalized_search_query: str | None = None
    publication_filter_start: datetime | None = None
    content_max_age_hours: int | None = None
    evidence_max_age_hours: int | None = None
    requested_max_results: int = 0
    returned_result_count: int = 0
    selected_result_count: int = 0
    web_provider_attempts: int = 0
    web_retry_count: int = 0
    web_rate_limit_events: int = 0
    web_search_latency_ms: float | None = None
    web_evidence_ids_generated: tuple[str, ...] = ()
    selected_domains: tuple[str, ...] = ()
    selected_publication_dates: tuple[str | None, ...] = ()
    web_failure_stage: str | None = None
    web_error_type: str | None = None
    web_error_message: str | None = None
    degraded_due_to_web_failure: bool = False
    validation_error_code: str | None = None
    validation_failed_claim: str | None = None
    validation_failed_line: int | None = None
    validation_detected_evidence_ids: tuple[str, ...] = ()
    validation_detected_evidence_categories: tuple[str, ...] = ()
    validation_rule: str | None = None
    validation_mode: ValidationMode = ValidationMode.STRICT
    validation_outcome: ValidationAction | None = None
    validation_findings_count: int = 0
    validation_warning_count: int = 0
    validation_repair_count: int = 0
    validation_block_count: int = 0
    validation_repair_applied: bool = False
    validation_repair_strategy: str | None = None
    validation_citation_repairs: tuple[CitationRepairDiagnostic, ...] = ()
    validation_findings: tuple[ValidationFindingDiagnostic, ...] = ()
    errors: tuple[str, ...] = ()


class AIScoutWorkflowResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    run_id: str
    question: str
    answer: GroundedScoutAnswer
    evidence: EvidenceLedger
    context: SelectedContext | None = None
    diagnostics: WorkflowDiagnostics


class AIScoutWorkflow:
    """One acyclic plan-execute-retrieve-synthesize graph."""

    def __init__(
        self,
        *,
        session: Session,
        planner: ScoutPlanner,
        synthesizer: AnswerSynthesizer,
        methodology_service: MethodologyService | None = None,
        web_search_provider: WebSearchProvider | None = None,
        web_search_enabled: bool = False,
        web_max_results: int = 5,
        web_content_max_age_hours: int | None = 24,
        web_current_status_max_age_hours: int | None = 168,
        validation_mode: ValidationMode = ValidationMode.STRICT,
        trace_sink: TraceSink | None = None,
        environment: str = "development",
        pricing_registry: PricingRegistry = EMPTY_PRICING_REGISTRY,
    ) -> None:
        self._session = session
        self._planner = planner
        self._synthesizer = synthesizer
        self._methodology = methodology_service or CuratedMethodologyService()
        self._web_search_provider = web_search_provider
        self._web_search_enabled = web_search_enabled
        self._web_max_results = min(max(web_max_results, 1), 5)
        self._web_content_max_age_hours = web_content_max_age_hours
        self._web_current_status_max_age_hours = web_current_status_max_age_hours
        self._validation_mode = validation_mode
        self._trace_sink = trace_sink if trace_sink is not None else NullTraceSink()
        self._environment = environment
        self._pricing_registry = pricing_registry
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AIScoutGraphState)
        graph.add_node("plan_request", self._observed_node("planner", self._plan_request))
        graph.add_node(
            "prepare_plan",
            self._observed_node("entity_resolution", self._prepare_plan),
        )
        graph.add_node(
            "execute_analytics",
            self._observed_node("analytics_preparation", self._execute_analytics),
        )
        graph.add_node(
            "retrieve_methodology",
            self._observed_node("methodology_retrieval", self._retrieve_methodology),
        )
        graph.add_node(
            "prepare_current_context",
            self._observed_node("current_context_preparation", self._prepare_current_context),
        )
        graph.add_node("search_web", self._observed_node("web_request", self._search_web))
        graph.add_node(
            "normalize_web_evidence",
            self._observed_node("web_normalization", self._normalize_web_evidence),
        )
        graph.add_node(
            "assemble_context",
            self._observed_node("context_assembly", self._assemble_context),
        )
        graph.add_node(
            "evaluate_sufficiency",
            self._observed_node("sufficiency", self._evaluate_sufficiency),
        )
        graph.add_node(
            "synthesize_answer",
            self._observed_node("synthesis", self._synthesize_answer),
        )
        graph.add_node(
            "validate_answer",
            self._observed_node("validation", self._validate_answer),
        )
        graph.add_node(
            "terminal_response",
            self._observed_node("terminal", self._terminal_response),
        )

        graph.add_edge(START, "plan_request")
        graph.add_conditional_edges(
            "plan_request",
            self._route_after_planning,
            {"prepare": "prepare_plan", "terminal": "terminal_response"},
        )
        graph.add_conditional_edges(
            "prepare_plan",
            self._route_after_preparation,
            {
                "execute": "execute_analytics",
                "current": "prepare_current_context",
                "terminal": "terminal_response",
            },
        )
        graph.add_edge("execute_analytics", "retrieve_methodology")
        graph.add_edge("retrieve_methodology", "prepare_current_context")
        graph.add_conditional_edges(
            "prepare_current_context",
            self._route_web_search,
            {"search": "search_web", "assemble": "assemble_context"},
        )
        graph.add_edge("search_web", "normalize_web_evidence")
        graph.add_edge("normalize_web_evidence", "assemble_context")
        graph.add_edge("assemble_context", "evaluate_sufficiency")
        graph.add_conditional_edges(
            "evaluate_sufficiency",
            self._route_after_sufficiency,
            {"synthesize": "synthesize_answer", "terminal": "terminal_response"},
        )
        graph.add_conditional_edges(
            "synthesize_answer",
            self._route_after_synthesis,
            {"validate": "validate_answer", "terminal": "terminal_response"},
        )
        graph.add_edge("validate_answer", END)
        graph.add_edge("terminal_response", END)
        return graph.compile()

    def _observed_node(
        self,
        stage: str,
        node: Callable[[AIScoutGraphState], dict[str, object]],
    ) -> Callable[[AIScoutGraphState], dict[str, object]]:
        """Wrap one graph node in a fail-open, content-free OTel span."""

        def observed(state: AIScoutGraphState) -> dict[str, object]:
            attributes: dict[str, str | bool | int | float | None] = {
                "ai.workflow": "ai_scout",
                "ai.stage": stage,
                "ai.correlation_id": state.get("run_id"),
                "ai.provider": self._planner.provider,
                "ai.model": self._planner.model,
                "ai.planner_prompt_version": PLANNER_PROMPT_VERSION,
                "ai.synthesis_prompt_version": SYNTHESIS_PROMPT_VERSION,
                "ai.validation_mode": self._validation_mode.value,
            }
            with start_span(f"ai_scout.{stage}", attributes):
                result = node(state)
                _set_stage_span_outcome(stage, result)
                return result

        return observed

    def run(
        self,
        question: str,
        *,
        case_id: str | None = None,
        run_id: str | None = None,
    ) -> AIScoutWorkflowResult:
        timestamp_utc = datetime.now(UTC)
        started = perf_counter()
        correlation_id = run_id or str(uuid4())
        span_attributes: dict[str, str | bool | int | float | None] = {
            "ai.workflow": "ai_scout",
            "ai.correlation_id": correlation_id,
            "ai.provider": self._planner.provider,
            "ai.model": self._planner.model,
            "ai.planner_prompt_version": PLANNER_PROMPT_VERSION,
            "ai.synthesis_prompt_version": SYNTHESIS_PROMPT_VERSION,
            "ai.validation_mode": self._validation_mode.value,
        }
        graph_config = langsmith_graph_config(
            correlation_id=correlation_id,
            provider=self._planner.provider,
            model=self._planner.model,
            planner_prompt_version=PLANNER_PROMPT_VERSION,
            synthesis_prompt_version=SYNTHESIS_PROMPT_VERSION,
            validation_mode=self._validation_mode.value,
        )
        with start_span("ai_scout.orchestration", span_attributes):
            final = self._graph.invoke(
                {
                    "run_id": correlation_id,
                    "question": question,
                    "evidence": EvidenceLedger(),
                    "stage_timings_ms": {},
                    "errors": [],
                },
                config=graph_config,
            )
            final_answer = final.get("final_answer")
            final_planner = final.get("planner_result")
            final_validation = final.get("validation_outcome")
            set_current_span_attributes(
                {
                    "ai.intent": _planner_intent_value(final_planner),
                    "ai.terminal_status": (
                        final_answer.status.value
                        if isinstance(final_answer, GroundedScoutAnswer)
                        else None
                    ),
                    "ai.web.used": final.get("selected_web_count", 0) > 0,
                    "ai.validation.outcome": (
                        final_validation.value
                        if isinstance(final_validation, ValidationAction)
                        else None
                    ),
                }
            )
        answer = final["final_answer"]
        ledger = final.get("evidence", EvidenceLedger())
        context = final.get("selected_context")
        planner_result = final.get("planner_result")
        synthesis_result = final.get("synthesis_result")
        preparation = final.get("preparation")
        current_request = final.get("current_context_request", CurrentContextRequest())
        provider_decision = (
            planner_result.provider_decision if planner_result is not None else None
        )
        plan = planner_result.plan if planner_result is not None else None
        execution = final.get("execution")
        tools = (
            tuple(record.tool_name.value for record in execution.evidence.records)
            if execution is not None
            else ()
        )
        pre_normalization_tools = (
            tuple(call.name.value for call in plan.calls) if plan is not None else ()
        )
        post_normalization_tools = (
            tuple(call.name.value for call in preparation.normalized_plan.calls)
            if preparation is not None and preparation.normalized_plan is not None
            else ()
        )
        pre_normalization_calls = (
            tuple(
                {
                    "tool_name": call.name.value,
                    "arguments": call.arguments.model_dump(mode="json"),
                }
                for call in plan.calls
            )
            if plan is not None
            else ()
        )
        post_normalization_calls = (
            tuple(
                {
                    "tool_name": call.name.value,
                    "arguments": dict(call.arguments),
                }
                for call in preparation.normalized_plan.calls
            )
            if preparation is not None and preparation.normalized_plan is not None
            else ()
        )
        current_only_guard_applied = current_request.required and current_request.current_only
        web_records = tuple(
            record
            for record in ledger.records
            if record.evidence_category.value == "web"
        )
        web_response = final.get("web_search_response")
        diagnostics = WorkflowDiagnostics(
            provider=self._planner.provider,
            model=self._planner.model,
            planner_prompt_version=PLANNER_PROMPT_VERSION,
            synthesis_provider=self._synthesizer.provider,
            synthesis_model=self._synthesizer.model,
            synthesis_prompt_version=SYNTHESIS_PROMPT_VERSION,
            planner_usage=(
                UsageMetadata.from_provider(planner_result.usage)
                if planner_result is not None
                else None
            ),
            synthesis_usage=(synthesis_result.usage if synthesis_result is not None else None),
            stage_timings_ms=final.get("stage_timings_ms", {}),
            total_ms=_elapsed_ms(started),
            tools_executed=tools,
            evidence_ids_generated=tuple(record.evidence_id for record in ledger.records),
            evidence_ids_supplied=context.evidence_ids if context is not None else (),
            evidence_ids_cited=answer.evidence_ids,
            methodology_sources_derived=answer.methodology_sources,
            terminal_status=answer.status,
            planner_primary_intent=(
                provider_decision.intent.value if provider_decision is not None else None
            ),
            planner_methodology_topic=(
                provider_decision.methodology_topic.value
                if provider_decision is not None
                else None
            ),
            planner_player_reference=_planner_player_reference(provider_decision),
            planner_team_reference=_planner_team_reference(provider_decision),
            planner_requested_analyses=(
                tuple(item.value for item in provider_decision.requested_analyses)
                if provider_decision is not None
                else ()
            ),
            planner_status=(planner_result.status.value if planner_result is not None else None),
            planner_failure_stage=(
                planner_result.diagnostic.failure_stage.value
                if planner_result is not None and planner_result.diagnostic is not None
                else None
            ),
            planner_error_type=(
                planner_result.diagnostic.error_type
                if planner_result is not None and planner_result.diagnostic is not None
                else None
            ),
            planner_response_received=(
                planner_result.response_received if planner_result is not None else False
            ),
            planner_fallback_attempted=(
                planner_result.fallback_attempted if planner_result is not None else False
            ),
            planner_fallback_provider=(
                planner_result.fallback_provider if planner_result is not None else None
            ),
            planner_fallback_model=(
                planner_result.fallback_model if planner_result is not None else None
            ),
            planner_fallback_reason=(
                planner_result.fallback_reason if planner_result is not None else None
            ),
            planner_fallback_success=(
                planner_result.fallback_success if planner_result is not None else False
            ),
            planner_provider_attempts=(
                planner_result.provider_attempts if planner_result is not None else 0
            ),
            synthesis_provider_attempts=final.get("synthesis_provider_attempts", 0),
            planner_validation_issues=(
                planner_result.diagnostic.validation_issues
                if planner_result is not None and planner_result.diagnostic is not None
                else ()
            ),
            normalized_intent=(
                preparation.normalized_plan.intent.kind.value
                if preparation is not None and preparation.normalized_plan is not None
                else plan.intent.kind.value if plan is not None else None
            ),
            methodology_topics_requested=final.get("methodology_topics_requested", ()),
            methodology_guard_applied=(
                planner_result.methodology_guard_applied
                if planner_result is not None
                else False
            ),
            current_context_required=current_request.required,
            current_only_guard_applied=current_only_guard_applied,
            current_context_category=(
                current_request.category.value if current_request.category else None
            ),
            analytics_requested_by_user=current_request.analytics_requested_by_user,
            analytics_suppressed_for_current_only=(
                current_only_guard_applied
                and bool(pre_normalization_tools or post_normalization_tools)
            ),
            pre_normalization_tool_names=pre_normalization_tools,
            post_normalization_tool_names=post_normalization_tools,
            pre_normalization_tool_calls=pre_normalization_calls,
            post_normalization_tool_calls=post_normalization_calls,
            terminal_branch_reason=_terminal_branch_reason(
                planner_result=planner_result,
                preparation=preparation,
                context=context,
                answer=answer,
                synthesis_result=synthesis_result,
            ),
            preparation_status=(preparation.status.value if preparation else None),
            preparation_reason_code=(
                preparation.reason_code.value
                if preparation is not None and preparation.reason_code is not None
                else None
            ),
            resolved_entities=_resolved_entity_diagnostics(preparation),
            web_search_enabled=self._web_search_enabled,
            web_search_required=current_request.required,
            web_search_reason=current_request.reason,
            web_search_category=(
                current_request.category.value if current_request.category else None
            ),
            web_search_provider=(
                self._web_search_provider.provider if self._web_search_provider else None
            ),
            normalized_search_query=current_request.query,
            publication_filter_start=current_request.publication_filter_start,
            content_max_age_hours=current_request.content_max_age_hours,
            evidence_max_age_hours=current_request.evidence_max_age_hours,
            requested_max_results=(self._web_max_results if current_request.required else 0),
            returned_result_count=(len(web_response.results) if web_response else 0),
            selected_result_count=len(web_records),
            web_provider_attempts=(
                web_response.provider_attempts
                if web_response
                else final.get("web_provider_attempts", 0)
            ),
            web_retry_count=(
                web_response.retry_count
                if web_response
                else final.get("web_retry_count", 0)
            ),
            web_rate_limit_events=(
                web_response.rate_limit_events
                if web_response
                else final.get("web_rate_limit_events", 0)
            ),
            web_search_latency_ms=final.get("web_search_latency_ms"),
            web_evidence_ids_generated=tuple(record.evidence_id for record in web_records),
            selected_domains=tuple(
                str(record.result["domain"])
                for record in web_records
                if record.result is not None
            ),
            selected_publication_dates=tuple(
                record.result.get("published_at")
                for record in web_records
                if record.result is not None
            ),
            web_failure_stage=final.get("web_failure_stage"),
            web_error_type=final.get("web_error_type"),
            web_error_message=final.get("web_error_message"),
            degraded_due_to_web_failure=(
                current_request.required
                and not web_records
                and bool(context and (context.analytics_evidence or context.methodology_evidence))
            ),
            validation_error_code=final.get("validation_error_code"),
            validation_failed_claim=final.get("validation_failed_claim"),
            validation_failed_line=final.get("validation_failed_line"),
            validation_detected_evidence_ids=final.get(
                "validation_detected_evidence_ids", ()
            ),
            validation_detected_evidence_categories=final.get(
                "validation_detected_evidence_categories", ()
            ),
            validation_rule=final.get("validation_rule"),
            validation_mode=self._validation_mode,
            validation_outcome=final.get("validation_outcome"),
            validation_findings_count=len(final.get("validation_findings", ())),
            validation_warning_count=final.get("validation_warning_count", 0),
            validation_repair_count=final.get("validation_repair_count", 0),
            validation_block_count=final.get("validation_block_count", 0),
            validation_repair_applied=final.get("validation_repair_applied", False),
            validation_repair_strategy=final.get("validation_repair_strategy"),
            validation_citation_repairs=final.get("validation_citation_repairs", ()),
            validation_findings=final.get("validation_findings", ()),
            errors=tuple(final.get("errors", [])),
        )
        result = AIScoutWorkflowResult(
            run_id=final["run_id"],
            question=question,
            answer=answer,
            evidence=ledger,
            context=context,
            diagnostics=diagnostics,
        )
        trace = build_run_trace(
            result,
            case_id=case_id,
            environment=self._environment,
            timestamp_utc=timestamp_utc,
            pricing_registry=self._pricing_registry,
        )
        record_trace_safely(self._trace_sink, trace)
        return result

    def _plan_request(self, state: AIScoutGraphState) -> dict[str, object]:
        started = perf_counter()
        result = self._planner.plan(state["question"])
        current_request = detect_current_context(
            state["question"],
            result.provider_decision,
            normalized_intent=(
                result.plan.intent.kind
                if result.plan is not None
                and result.plan.decision.value == "ready"
                else None
            ),
            content_max_age_hours=self._web_content_max_age_hours,
            evidence_max_age_hours=self._web_current_status_max_age_hours,
        )
        return {
            "planner_result": result,
            "current_context_request": current_request,
            "stage_timings_ms": _timing(state, "planner", started),
        }

    def _prepare_plan(self, state: AIScoutGraphState) -> dict[str, object]:
        started = perf_counter()
        plan = state["planner_result"].plan
        assert plan is not None
        preparation = prepare_plan(self._session, plan)
        return {
            "preparation": preparation,
            "stage_timings_ms": _timing(state, "preparation", started),
        }

    def _execute_analytics(self, state: AIScoutGraphState) -> dict[str, object]:
        started = perf_counter()
        preparation = state["preparation"]
        assert preparation.normalized_plan is not None
        execution = execute_plan(
            self._session,
            preparation.normalized_plan,
            run_id=state["run_id"],
        )
        return {
            "execution": execution,
            "evidence": execution.evidence,
            "stage_timings_ms": _timing(state, "analytics_execution", started),
        }

    def _retrieve_methodology(self, state: AIScoutGraphState) -> dict[str, object]:
        started = perf_counter()
        plan = state["planner_result"].plan
        assert plan is not None
        ledger = state["evidence"]
        present = methodology_topics_already_present(ledger.records)
        topics = tuple(
            topic for topic in select_methodology_topics(plan.intent) if topic.value not in present
        )
        errors = list(state.get("errors", []))
        try:
            retrieved = methodology_evidence(
                service=self._methodology,
                topics=topics,
                run_id=state["run_id"],
                start_order=len(ledger.records) + 1,
            )
            ledger = ledger.extend(retrieved)
        except Exception as exc:  # noqa: BLE001 - methodology sources are an I/O boundary
            errors.append(f"Methodology retrieval failed: {sanitize_error_message(str(exc))}")
        return {
            "evidence": ledger,
            "errors": errors,
            "methodology_topics_requested": tuple(topic.value for topic in topics),
            "stage_timings_ms": _timing(state, "methodology_retrieval", started),
        }

    def _prepare_current_context(self, state: AIScoutGraphState) -> dict[str, object]:
        started = perf_counter()
        request = state.get("current_context_request", CurrentContextRequest())
        canonical_subject = _subject_from_preparation(
            state.get("preparation"),
            request,
        )
        fallback_subject = canonical_subject or _subject_from_evidence(state["evidence"])
        request_for_query = request.model_copy(
            update={"subject": canonical_subject or request.subject}
        )
        query = build_search_query(request_for_query, fallback_subject=fallback_subject)
        request = request.model_copy(
            update={
                "subject": canonical_subject or request.subject,
                "query": query,
            }
        )
        failure_stage = None
        error_message = None
        if request.required and not self._web_search_enabled:
            failure_stage = "disabled"
            error_message = "Current external information is unavailable because web search is disabled."
        elif request.required and self._web_search_provider is None:
            failure_stage = "provider_unavailable"
            error_message = "Current external information could not be retrieved."
        elif request.required and query is None:
            failure_stage = "query_construction"
            error_message = "Current external information could not be searched without a subject."
        return {
            "current_context_request": request,
            "web_failure_stage": failure_stage,
            "web_error_message": error_message,
            "stage_timings_ms": _timing(state, "current_context_preparation", started),
        }

    def _search_web(self, state: AIScoutGraphState) -> dict[str, object]:
        started = perf_counter()
        request = state["current_context_request"]
        assert request.query is not None
        assert self._web_search_provider is not None
        try:
            response = self._web_search_provider.search(
                WebSearchRequest(
                    query=request.query,
                    max_results=self._web_max_results,
                    publication_filter_start=request.publication_filter_start,
                    content_max_age_hours=request.content_max_age_hours,
                )
            )
            return {
                "web_search_response": response,
                "web_search_latency_ms": _elapsed_ms(started),
                "stage_timings_ms": _timing(state, "web_search", started),
            }
        except Exception as exc:  # noqa: BLE001 - provider is an external boundary
            error_type = (
                exc.error_type if isinstance(exc, WebSearchProviderError) else type(exc).__name__
            )
            message = sanitize_error_message(str(exc))
            return {
                "web_failure_stage": "provider",
                "web_error_type": error_type,
                "web_error_message": message,
                "web_search_latency_ms": _elapsed_ms(started),
                "web_provider_attempts": (
                    exc.provider_attempts if isinstance(exc, WebSearchProviderError) else 1
                ),
                "web_retry_count": (
                    exc.retry_count if isinstance(exc, WebSearchProviderError) else 0
                ),
                "web_rate_limit_events": (
                    exc.rate_limit_events if isinstance(exc, WebSearchProviderError) else 0
                ),
                "errors": [*state.get("errors", []), f"Web search failed: {message}"],
                "stage_timings_ms": _timing(state, "web_search", started),
            }

    def _normalize_web_evidence(self, state: AIScoutGraphState) -> dict[str, object]:
        started = perf_counter()
        response = state.get("web_search_response")
        if response is None:
            return {
                "selected_web_count": 0,
                "stage_timings_ms": _timing(state, "web_normalization", started),
            }
        request = state["current_context_request"]
        selected = select_web_results(
            response.results,
            subject=request.subject,
            max_evidence_age_hours=request.evidence_max_age_hours,
            now=response.retrieved_at,
        )
        ledger = state["evidence"]
        records = web_evidence(
            results=selected,
            query=response.query,
            provider=response.provider,
            run_id=state["run_id"],
            start_order=len(ledger.records) + 1,
            publication_filter_start=request.publication_filter_start,
            content_max_age_hours=request.content_max_age_hours,
            evidence_max_age_hours=request.evidence_max_age_hours,
        )
        result: dict[str, object] = {
            "evidence": ledger.extend(records),
            "selected_web_count": len(records),
            "stage_timings_ms": _timing(state, "web_normalization", started),
        }
        if not records:
            result.update(
                {
                    "web_failure_stage": "no_relevant_results",
                    "web_error_message": (
                        "Current information could not be established from retrieved external "
                        "evidence."
                    ),
                }
            )
        return result

    def _assemble_context(self, state: AIScoutGraphState) -> dict[str, object]:
        started = perf_counter()
        plan = state["planner_result"].plan
        assert plan is not None
        # Plan limitations are provider-authored intent notes. Canonical public caveats
        # come from executed tools/methodology evidence, never from planner prose.
        limitations: list[str] = []
        if state.get("web_error_message"):
            limitations.append(state["web_error_message"])
        context = assemble_context(
            question=state["question"],
            intent=plan.intent,
            records=state["evidence"].records,
            limitations=tuple(limitations),
        )
        return {
            "selected_context": context,
            "stage_timings_ms": _timing(state, "context_assembly", started),
        }

    def _evaluate_sufficiency(self, state: AIScoutGraphState) -> dict[str, object]:
        context = state["selected_context"]
        execution = state.get("execution")
        status = context.status
        current_request = state.get("current_context_request", CurrentContextRequest())
        if current_request.required and current_request.current_only and not context.web_evidence:
            status = ContextStatus.INSUFFICIENT_EVIDENCE
        if execution is not None and execution.status is PlanExecutionStatus.STOPPED_ON_ERROR:
            status = ContextStatus.INSUFFICIENT_EVIDENCE
        if _RAW_EVENT_ROW_REQUEST.search(state["question"]):
            # Governed AI Scout tools expose aggregates and profiles, not complete
            # raw-event exports. Evidence can explain the loaded player, but cannot
            # satisfy the requested row-level payload.
            status = ContextStatus.INSUFFICIENT_EVIDENCE
        return {"context_status": status}

    def _synthesize_answer(self, state: AIScoutGraphState) -> dict[str, object]:
        started = perf_counter()
        errors = list(state.get("errors", []))
        provider_attempts = 0
        try:
            context = state["selected_context"]
            result = deterministic_governed_synthesis(context)
            if result is None:
                provider_attempts = 1
                result = self._synthesizer.synthesize(context)
            result = apply_role_fit_semantic_fallback(result, context)
            if _has_no_qualifying_web_for_mixed_context(state):
                current_request = state["current_context_request"]
                result = apply_no_qualifying_web_fallback(
                    result,
                    context,
                    category=(
                        current_request.category.value
                        if current_request.category is not None
                        else None
                    ),
                )
            return {
                "synthesis_result": result,
                "synthesis_provider_attempts": provider_attempts,
                "stage_timings_ms": _timing(state, "synthesis", started),
            }
        except Exception as exc:  # noqa: BLE001 - synthesis providers are untrusted
            errors.append(f"Synthesis failed: {sanitize_error_message(str(exc))}")
            return {
                "errors": errors,
                "synthesis_provider_attempts": provider_attempts,
                "stage_timings_ms": _timing(state, "synthesis", started),
            }

    def _validate_answer(self, state: AIScoutGraphState) -> dict[str, object]:
        started = perf_counter()
        result = state["synthesis_result"]
        errors = list(state.get("errors", []))
        try:
            validation = validate_answer_with_policy(
                result.answer,
                state["evidence"],
                limitations=state["selected_context"].limitations,
                mode=self._validation_mode,
                current_only=state.get(
                    "current_context_request",
                    CurrentContextRequest(),
                ).current_only,
                supplied_evidence_ids=state["selected_context"].evidence_ids,
            )
            answer = _finalize_validated_answer(validation.answer)
        except GroundingValidationError as exc:
            errors.append(sanitize_error_message(str(exc)))
            answer = _safe_answer(
                GroundedAnswerStatus.ERROR,
                _validation_failure_message(exc.error_code),
            )
            return {
                "final_answer": answer,
                "errors": errors,
                "validation_error_code": exc.error_code,
                "validation_failed_claim": (
                    sanitize_user_facing_text(exc.failed_claim).strip()[:1000]
                    if exc.failed_claim
                    else None
                ),
                "validation_failed_line": exc.failed_line,
                "validation_detected_evidence_ids": exc.detected_evidence_ids,
                "validation_detected_evidence_categories": (
                    exc.detected_evidence_categories
                ),
                "validation_rule": exc.rule,
                **_validation_resolution_diagnostics(exc.resolution),
                "stage_timings_ms": _timing(state, "answer_validation", started),
            }
        except ValueError as exc:
            errors.append(sanitize_error_message(str(exc)))
            answer = _safe_answer(
                GroundedAnswerStatus.ERROR,
                _validation_failure_message("answer_validation_error"),
            )
            return {
                "final_answer": answer,
                "errors": errors,
                "validation_error_code": "answer_validation_error",
                "validation_rule": "answer_must_satisfy_grounding_validation",
                "validation_outcome": ValidationAction.BLOCK.value,
                "validation_block_count": 1,
                "stage_timings_ms": _timing(state, "answer_validation", started),
            }
        return {
            "final_answer": answer,
            "errors": errors,
            **_validation_resolution_diagnostics(
                validation.resolution,
                repair_applied=validation.repair_applied,
                repair_strategy=validation.repair_strategy,
                citation_repairs=tuple(
                    {
                        "original_id": repair.original_id,
                        "canonical_id": repair.canonical_id,
                        "strategy": repair.strategy,
                    }
                    for repair in validation.citation_repairs
                ),
            ),
            "stage_timings_ms": _timing(state, "answer_validation", started),
        }

    def _terminal_response(self, state: AIScoutGraphState) -> dict[str, object]:
        planner = state.get("planner_result")
        preparation = state.get("preparation")
        current = state.get("current_context_request", CurrentContextRequest())
        if (
            current.required
            and current.current_only
            and state.get("context_status") is ContextStatus.INSUFFICIENT_EVIDENCE
        ):
            message = (
                state.get("web_error_message")
                or "Current information could not be established from retrieved external evidence."
            )
            status = GroundedAnswerStatus.INSUFFICIENT_EVIDENCE
        elif planner is None or planner.status is not PlannerStatus.SUCCESS:
            message = "I could not safely plan this request. Please try again."
            status = GroundedAnswerStatus.ERROR
        elif preparation is not None and preparation.status is PlanPreparationStatus.CLARIFICATION_REQUIRED:
            message = preparation.clarification_message or "Please clarify the requested entity."
            status = GroundedAnswerStatus.CLARIFICATION_REQUIRED
        elif preparation is not None and preparation.status is PlanPreparationStatus.UNSUPPORTED:
            message = preparation.unsupported_reason or "That request is not supported."
            status = GroundedAnswerStatus.UNSUPPORTED
        elif preparation is not None and preparation.status is PlanPreparationStatus.NOT_FOUND:
            message = preparation.error_message or "The requested entity was not found."
            status = GroundedAnswerStatus.INSUFFICIENT_EVIDENCE
        elif preparation is not None and preparation.status is PlanPreparationStatus.INVALID:
            message = "The validated plan could not be prepared safely."
            status = GroundedAnswerStatus.ERROR
        elif "synthesis_result" not in state and state.get("context_status") is ContextStatus.SUFFICIENT:
            message = "The evidence was retrieved, but a grounded answer could not be generated."
            status = GroundedAnswerStatus.ERROR
        else:
            message = "FootyScout does not have sufficient evidence to answer that request."
            status = GroundedAnswerStatus.INSUFFICIENT_EVIDENCE
        return {"final_answer": _safe_answer(status, message)}

    @staticmethod
    def _route_after_planning(state: AIScoutGraphState) -> str:
        result = state["planner_result"]
        return "prepare" if result.status is PlannerStatus.SUCCESS and result.plan else "terminal"

    @staticmethod
    def _route_after_preparation(state: AIScoutGraphState) -> str:
        current = state.get("current_context_request", CurrentContextRequest())
        if current.required and current.current_only:
            return "current"
        if state["preparation"].status is PlanPreparationStatus.READY:
            return "execute"
        return "terminal"

    def _route_web_search(self, state: AIScoutGraphState) -> str:
        request = state["current_context_request"]
        return (
            "search"
            if request.required
            and request.query is not None
            and self._web_search_enabled
            and self._web_search_provider is not None
            else "assemble"
        )

    @staticmethod
    def _route_after_sufficiency(state: AIScoutGraphState) -> str:
        return (
            "synthesize"
            if state["context_status"] is ContextStatus.SUFFICIENT
            else "terminal"
        )

    @staticmethod
    def _route_after_synthesis(state: AIScoutGraphState) -> str:
        return "validate" if "synthesis_result" in state else "terminal"


def _safe_answer(status: GroundedAnswerStatus, message: str) -> GroundedScoutAnswer:
    return GroundedScoutAnswer(answer_markdown=message, status=status)


_CITATION_VALIDATION_ERROR_CODES = {
    "undeclared_inline_citation",
    "unknown_evidence_id",
}


def _validation_failure_message(error_code: str) -> str:
    if error_code in _CITATION_VALIDATION_ERROR_CODES:
        return "I could not validate the evidence citations in the generated answer."
    return "I couldn't produce a response that passed the evidence checks."


def _finalize_validated_answer(answer: GroundedScoutAnswer) -> GroundedScoutAnswer:
    """Make successful grounded workflow state authoritative over provider status prose."""
    if (
        answer.status is not GroundedAnswerStatus.ANSWERED
        and answer.evidence_ids
        and has_substantive_claims(answer.answer_markdown)
    ):
        return answer.model_copy(update={"status": GroundedAnswerStatus.ANSWERED})
    return answer


def _validation_resolution_diagnostics(
    resolution: ValidationResolution | None,
    *,
    repair_applied: bool = False,
    repair_strategy: str | None = None,
    citation_repairs: tuple[dict[str, str], ...] = (),
) -> dict[str, object]:
    if resolution is None:
        return {}
    return {
        "validation_outcome": resolution.outcome.value,
        "validation_findings": tuple(
            {
                "rule": item.finding.rule,
                "error_code": item.finding.error_code,
                "action": item.action.value,
                "line": item.finding.line,
                "evidence_ids": item.finding.evidence_ids,
                "evidence_categories": item.finding.evidence_categories,
            }
            for item in resolution.findings
        ),
        "validation_warning_count": resolution.warning_count,
        "validation_repair_count": resolution.repair_count + len(citation_repairs),
        "validation_block_count": resolution.block_count,
        "validation_repair_applied": repair_applied,
        "validation_repair_strategy": repair_strategy,
        "validation_citation_repairs": citation_repairs,
    }


def _timing(
    state: AIScoutGraphState,
    stage: str,
    started: float,
) -> dict[str, float]:
    timings = dict(state.get("stage_timings_ms", {}))
    timings[stage] = _elapsed_ms(started)
    return timings


def _elapsed_ms(started: float) -> float:
    return max(0.0, (perf_counter() - started) * 1000.0)


def _set_stage_span_outcome(stage: str, result: dict[str, object]) -> None:
    """Attach safe stage outcomes without serializing user or evidence content."""
    attributes: dict[str, str | bool | int | float | None] = {"ai.stage": stage}
    if stage == "planner":
        attributes["ai.intent"] = _planner_intent_value(result.get("planner_result"))
    elif stage == "web_request":
        attributes["ai.web.used"] = result.get("web_search_response") is not None
    elif stage == "validation":
        outcome = result.get("validation_outcome")
        attributes["ai.validation.outcome"] = (
            outcome.value if isinstance(outcome, ValidationAction) else str(outcome or "")
        )
    elif stage == "terminal":
        answer = result.get("final_answer")
        attributes["ai.terminal_status"] = (
            answer.status.value if isinstance(answer, GroundedScoutAnswer) else None
        )
    set_current_span_attributes(attributes)


def _planner_intent_value(value: object) -> str | None:
    if not isinstance(value, PlannerResult):
        return None
    if value.plan is not None:
        return value.plan.intent.kind.value
    if value.provider_decision is not None:
        return value.provider_decision.intent.value
    return None


def _has_no_qualifying_web_for_mixed_context(state: AIScoutGraphState) -> bool:
    """Identify a successful mixed retrieval whose web candidates were all rejected."""
    request = state.get("current_context_request")
    response = state.get("web_search_response")
    context = state.get("selected_context")
    return bool(
        request is not None
        and request.required
        and not request.current_only
        and response is not None
        and response.results
        and state.get("web_failure_stage") == "no_relevant_results"
        and not state.get("web_error_type")
        and context is not None
        and context.analytics_evidence
        and context.methodology_evidence
        and not context.web_evidence
    )


def _subject_from_evidence(ledger: EvidenceLedger) -> str | None:
    for record in ledger.records:
        subject = _find_player_name(record.result)
        if subject:
            return subject
    return None


def _subject_from_preparation(
    preparation: PlanPreparationResult | None,
    request: CurrentContextRequest,
) -> str | None:
    """Reuse canonical resolution for web context instead of reparsing planner prose."""
    if preparation is None or preparation.normalized_plan is None:
        return None
    preferred_type = (
        "team"
        if request.category
        in {
            CurrentContextCategory.CURRENT_MANAGER,
            CurrentContextCategory.CURRENT_TEAM_CONTEXT,
        }
        else "player"
    )
    return next(
        (
            entity.display_name
            for entity in preparation.normalized_plan.resolved_entities
            if entity.entity_type == preferred_type
        ),
        None,
    )


def _find_player_name(value: object) -> str | None:
    if isinstance(value, dict):
        player_name = value.get("player_name")
        if isinstance(player_name, str) and player_name.strip():
            return player_name.strip()
        for child in value.values():
            found = _find_player_name(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_player_name(child)
            if found:
                return found
    return None


def _planner_player_reference(decision: LLMPlannerDecision | None) -> str | None:
    if decision is None:
        return None
    if decision.primary_player_id:
        return f"player_id:{decision.primary_player_id}"
    return decision.primary_player_name.strip() or None


def _planner_team_reference(decision: LLMPlannerDecision | None) -> str | None:
    if decision is None:
        return None
    if decision.team_id:
        return f"team_id:{decision.team_id}"
    return decision.team_name.strip() or None


def _terminal_branch_reason(
    *,
    planner_result: PlannerResult | None,
    preparation: PlanPreparationResult | None,
    context: SelectedContext | None,
    answer: GroundedScoutAnswer,
    synthesis_result: SynthesisResult | None,
) -> str:
    if planner_result is None or planner_result.status is not PlannerStatus.SUCCESS:
        return "planner_not_successful"
    if answer.status is GroundedAnswerStatus.ANSWERED:
        return answer.status.value
    if preparation is not None and preparation.status is not PlanPreparationStatus.READY:
        return f"preparation_{preparation.status.value}"
    if context is not None and context.status is not ContextStatus.SUFFICIENT:
        return f"context_{context.status.value}"
    if synthesis_result is None and answer.status is GroundedAnswerStatus.ERROR:
        return "synthesis_not_available"
    if answer.status is GroundedAnswerStatus.ERROR:
        return "answer_validation_error"
    return answer.status.value


def _resolved_entity_diagnostics(
    preparation: PlanPreparationResult | None,
) -> tuple[dict[str, object], ...]:
    """Expose bounded resolution outcomes needed to audit deterministic preflight."""
    if preparation is None:
        return ()
    entities: list[dict[str, object]] = []
    for resolution in preparation.player_resolutions:
        player = resolution.player
        entities.append(
            {
                "entity_type": "player",
                "query": str(resolution.query),
                "status": resolution.status.value,
                "stable_id": resolution.player_id,
                "display_name": player.player_name if player is not None else None,
                "position_group": player.position_group if player is not None else None,
                "analytics_supported": None,
            }
        )
    for resolution in preparation.team_resolutions:
        team = resolution.team
        entities.append(
            {
                "entity_type": "team",
                "query": str(resolution.query),
                "status": resolution.status.value,
                "stable_id": resolution.team_id,
                "display_name": team.team_name if team is not None else None,
                "position_group": None,
                "analytics_supported": resolution.analytics_supported,
            }
        )
    return tuple(entities)
