from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from app.ai.context import ContextStatus, SelectedContext
from app.ai.diagnostics import ValidationIssue
from app.ai.grounding import EvidenceCategory, EvidenceLedger, EvidenceRecord
from app.ai.observability import (
    InMemoryTraceSink,
    JsonlTraceSink,
    ModelPricing,
    PricingRegistry,
    build_run_trace,
    load_jsonl_traces,
    record_trace_safely,
    summarize_traces,
)
from app.ai.observability.pricing import (
    DEFAULT_PRICING_REGISTRY_PATH,
    estimate_cost_usd,
    load_default_pricing_registry,
)
from app.ai.run import UsageMetadata
from app.ai.schemas import (
    ProductionStatus,
    SourceCategory,
    ToolExecutionStatus,
    ToolName,
)
from app.ai.synthesis import GroundedAnswerStatus, GroundedScoutAnswer
from app.ai.validation import ValidationAction, ValidationMode
from app.ai.workflow import (
    AIScoutWorkflowResult,
    CitationRepairDiagnostic,
    ValidationFindingDiagnostic,
    WorkflowDiagnostics,
)
from app.services.ai_scout import public_ai_scout_response
from scripts.run_ai_scout_evals import _pricing_registry


def _record(
    *,
    order: int,
    category: EvidenceCategory,
    tool: ToolName,
    result: dict[str, object] | None,
    arguments: dict[str, object] | None = None,
    status: ToolExecutionStatus = ToolExecutionStatus.SUCCESS,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"run-1:evidence-{order}",
        evidence_category=category,
        tool_name=tool,
        normalized_arguments=arguments or {},
        execution_status=status,
        result=result,
        source_category=(
            SourceCategory.EXTERNAL_WEB
            if category is EvidenceCategory.WEB
            else SourceCategory.CURATED_DOCUMENTATION
            if category is EvidenceCategory.METHODOLOGY
            else SourceCategory.POSTGRESQL
        ),
        production_status=ProductionStatus.PRODUCTION,
        execution_order=order,
        methodology_topic=("role_fit" if category is EvidenceCategory.METHODOLOGY else None),
        methodology_sources=(
            ("role_fit:primary",) if category is EvidenceCategory.METHODOLOGY else ()
        ),
        provenance=(
            {"retrieval": "exact_topic"}
            if category is EvidenceCategory.METHODOLOGY
            else {}
        ),
    )


_PLANNER_USAGE = UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150)
_SYNTHESIS_USAGE = UsageMetadata(input_tokens=200, output_tokens=80, total_tokens=280)


def _result(
    *,
    status: GroundedAnswerStatus = GroundedAnswerStatus.ANSWERED,
    context_status: ContextStatus | None = ContextStatus.SUFFICIENT,
    planner_usage: UsageMetadata | None = _PLANNER_USAGE,
    synthesis_usage: UsageMetadata | None = _SYNTHESIS_USAGE,
    web: bool = True,
    validation_outcome: ValidationAction | None = ValidationAction.PASS,
    repair_count: int = 0,
    citation_repairs: tuple[CitationRepairDiagnostic, ...] = (),
    block_count: int = 0,
    planner_error_type: str | None = None,
    validation_mode: ValidationMode = ValidationMode.STRICT,
    planner_provider_attempts: int = 1,
    synthesis_provider_attempts: int | None = None,
) -> AIScoutWorkflowResult:
    records = [
        _record(
            order=1,
            category=EvidenceCategory.ANALYTICS,
            tool=ToolName.GET_ROLE_FIT,
            arguments={
                "player_id": 1,
                "target_team_id": 904,
                "authorization": "Bearer super-secret",
                "raw_response": "hidden response",
            },
            result={"items": [{"score": 0.3}], "private_body": "never copy me"},
        ),
        _record(
            order=2,
            category=EvidenceCategory.METHODOLOGY,
            tool=ToolName.GET_METHODOLOGY,
            arguments={"topic": "role_fit"},
            result={"summary": "methodology body must stay in the ledger"},
        ),
    ]
    if web:
        records.append(
            _record(
                order=3,
                category=EvidenceCategory.WEB,
                tool=ToolName.SEARCH_WEB,
                arguments={"query": "private current query"},
                result={
                    "title": "Public report",
                    "domain": "example.com",
                    "published_at": "2026-09-01T00:00:00Z",
                    "snippet": "full web snippet must not be duplicated",
                },
            )
        )
    ledger = EvidenceLedger(records=tuple(records))
    supplied = tuple(record.evidence_id for record in records)
    cited = supplied[:2]
    context = (
        SelectedContext(
            question="private user question",
            intent="role_fit",
            analytics_evidence=(records[0],),
            methodology_evidence=(records[1],),
            web_evidence=((records[2],) if web else ()),
            status=context_status,
        )
        if context_status is not None
        else None
    )
    finding = ValidationFindingDiagnostic(
        rule="citation_required",
        error_code="missing_citation",
        action=validation_outcome or ValidationAction.BLOCK,
        line=14,
        evidence_ids=("run-1:evidence-1",),
        evidence_categories=("analytics",),
    )
    timings = {
        "planner": 10.0,
        "preparation": 2.0,
        "analytics_execution": 15.0,
        "methodology_retrieval": 3.0,
        "context_assembly": 1.0,
    }
    if synthesis_usage is not None or status is GroundedAnswerStatus.ANSWERED:
        timings.update({"synthesis": 20.0, "answer_validation": 4.0})
    if web:
        timings["web_search"] = 8.0
    diagnostics = WorkflowDiagnostics(
        provider="openai",
        model="planner-test",
        planner_prompt_version="planner-v3",
        synthesis_provider="openai",
        synthesis_model="synthesis-test",
        synthesis_prompt_version="grounded-synthesis-v5",
        planner_usage=planner_usage,
        synthesis_usage=synthesis_usage,
        planner_provider_attempts=planner_provider_attempts,
        synthesis_provider_attempts=(
            int(synthesis_usage is not None)
            if synthesis_provider_attempts is None
            else synthesis_provider_attempts
        ),
        stage_timings_ms=timings,
        total_ms=65.0,
        tools_executed=("get_role_fit",),
        evidence_ids_generated=supplied,
        evidence_ids_supplied=supplied,
        evidence_ids_cited=cited,
        methodology_sources_derived=("role_fit:primary",),
        terminal_status=status,
        planner_primary_intent="role_fit",
        planner_status="success" if planner_error_type is None else "error",
        planner_failure_stage=("provider" if planner_error_type else None),
        planner_error_type=planner_error_type,
        normalized_intent="role_fit",
        methodology_topics_requested=("role_fit",),
        post_normalization_tool_names=("get_role_fit",),
        terminal_branch_reason=status.value,
        web_search_enabled=web,
        web_search_required=web,
        web_search_provider="exa" if web else None,
        web_search_category="current_club" if web else None,
        normalized_search_query="private current query" if web else None,
        returned_result_count=2 if web else 0,
        selected_result_count=1 if web else 0,
        web_search_latency_ms=8.0 if web else None,
        selected_domains=("example.com",) if web else (),
        selected_publication_dates=("2026-09-01T00:00:00Z",) if web else (),
        validation_outcome=validation_outcome,
        validation_mode=validation_mode,
        validation_findings_count=1 if validation_outcome else 0,
        validation_warning_count=(1 if validation_outcome is ValidationAction.WARN else 0),
        validation_repair_count=repair_count,
        validation_block_count=block_count,
        validation_repair_applied=repair_count > 0,
        validation_repair_strategy=("remove_claim" if repair_count else None),
        validation_citation_repairs=citation_repairs,
        validation_findings=((finding,) if validation_outcome else ()),
    )
    return AIScoutWorkflowResult(
        run_id="run-1",
        question="private user question with sk-test-secret",
        answer=GroundedScoutAnswer(
            answer_markdown="Grounded answer.",
            evidence_ids=cited,
            methodology_sources=("role_fit:primary",),
            status=status,
        ),
        evidence=ledger,
        context=context,
        diagnostics=diagnostics,
    )


def test_answered_trace_normalizes_existing_diagnostics_without_content_leakage() -> None:
    trace = build_run_trace(
        _result(),
        case_id="role_fit_current_001",
        environment="test",
        timestamp_utc=datetime(2026, 9, 30, tzinfo=UTC),
    )

    assert trace.trace_schema_version == "ai-scout-trace-v1"
    assert trace.case_id == "role_fit_current_001"
    assert trace.primary_intent == trace.normalized_intent == "role_fit"
    assert trace.tools_planned == trace.tools_executed == ("get_role_fit",)
    assert trace.methodology_topics_requested == ("role_fit",)
    assert trace.methodology_topics_retrieved == ("role_fit",)
    assert trace.methodology_retrieval_modes == ("exact_topic",)
    assert trace.analytics_evidence_count == 1
    assert trace.methodology_evidence_count == 1
    assert trace.web_evidence_count == 1
    assert trace.evidence_generated_count == 3
    assert trace.evidence_supplied_count == 3
    assert trace.evidence_cited_count == 2
    assert trace.unused_supplied_evidence_count == 1
    assert trace.total_input_tokens == 300
    assert trace.total_output_tokens == 130
    assert trace.total_tokens == 430
    assert trace.planner_provider_call_count == 1
    assert trace.synthesis_provider_call_count == 1
    assert trace.llm_provider_call_count == 2
    assert trace.planner_latency_ms == 10.0
    assert trace.web_latency_ms == 8.0
    assert trace.workflow_completed is True
    assert trace.success is True
    assert trace.validation_findings[0].rule == "citation_required"
    assert trace.validation_findings[0].error_code == "missing_citation"
    assert trace.validation_findings[0].action == "pass"
    assert trace.validation_findings[0].line == 14
    assert trace.validation_findings[0].evidence_ids == ("run-1:evidence-1",)
    assert trace.validation_findings[0].evidence_categories == ("analytics",)

    serialized = trace.model_dump_json()
    assert "private user question" not in serialized
    assert "private current query" not in serialized
    assert "full web snippet" not in serialized
    assert "methodology body" not in serialized
    assert "never copy me" not in serialized
    assert "super-secret" not in serialized
    assert "hidden response" not in serialized
    assert "Grounded answer" not in serialized
    assert trace.tool_executions[0].safe_argument_summary["authorization"] == "[redacted]"


@pytest.mark.parametrize(
    ("status", "context_status"),
    [
        (GroundedAnswerStatus.CLARIFICATION_REQUIRED, None),
        (GroundedAnswerStatus.INSUFFICIENT_EVIDENCE, ContextStatus.INSUFFICIENT_EVIDENCE),
    ],
)
def test_guarded_terminal_outcomes_are_completed_not_crashes(
    status: GroundedAnswerStatus,
    context_status: ContextStatus | None,
) -> None:
    trace = build_run_trace(
        _result(
            status=status,
            context_status=context_status,
            synthesis_usage=None,
            web=False,
        )
    )

    assert trace.workflow_completed is True
    assert trace.provider_failure is False
    assert trace.success is True
    assert trace.terminal_status == status.value


def test_provider_failure_has_taxonomy_and_missing_usage_and_cost_stay_null() -> None:
    result = _result(
        status=GroundedAnswerStatus.ERROR,
        context_status=None,
        planner_usage=None,
        synthesis_usage=None,
        web=False,
        planner_error_type="APITimeoutError",
        validation_outcome=None,
    )
    trace = build_run_trace(result)

    assert trace.error_types == ("planner_timeout",)
    assert trace.provider_failure is True
    assert trace.workflow_completed is False
    assert trace.success is False
    assert trace.planner_total_tokens is None
    assert trace.total_tokens is None
    assert trace.estimated_total_cost_usd is None


def test_trace_preserves_safe_structured_parse_locations_and_types() -> None:
    result = _result(
        status=GroundedAnswerStatus.ERROR,
        context_status=None,
        planner_usage=None,
        synthesis_usage=None,
        web=False,
        planner_error_type="ValidationError",
        validation_outcome=None,
    )
    result = result.model_copy(
        update={
            "diagnostics": result.diagnostics.model_copy(
                update={
                    "planner_failure_stage": "structured_parse",
                    "planner_validation_issues": (
                        ValidationIssue(
                            location=("decision", "intent"),
                            error_type="missing",
                        ),
                    ),
                }
            )
        }
    )

    trace = build_run_trace(result)

    assert trace.errors[0].validation_issues == (
        ValidationIssue(location=("decision", "intent"), error_type="missing"),
    )
    assert "input_value" not in trace.model_dump_json()


def test_trace_exposes_resolved_subject_and_governed_preflight_reason() -> None:
    result = _result(web=False)
    result = result.model_copy(
        update={
            "diagnostics": result.diagnostics.model_copy(
                update={
                    "preparation_status": "not_found",
                    "preparation_reason_code": "candidate_position_incompatible",
                    "resolved_entities": (
                        {
                            "entity_type": "player",
                            "query": "Xhaka",
                            "status": "resolved",
                            "stable_id": 3500,
                            "display_name": "Granit Xhaka",
                            "position_group": "MID",
                            "analytics_supported": None,
                        },
                    ),
                }
            )
        }
    )

    trace = build_run_trace(result)

    assert trace.preparation_reason_code == "candidate_position_incompatible"
    assert trace.entity_resolutions[0].stable_id == 3500
    assert trace.entity_resolutions[0].position_group == "MID"


def test_validation_block_trace_preserves_only_safe_structured_findings() -> None:
    trace = build_run_trace(
        _result(
            status=GroundedAnswerStatus.ERROR,
            validation_outcome=ValidationAction.BLOCK,
            block_count=1,
        )
    )

    assert trace.validation_outcome == "block"
    assert trace.validation_block_count == 1
    assert trace.validation_findings[0].model_dump() == {
        "rule": "citation_required",
        "error_code": "missing_citation",
        "action": "block",
        "line": 14,
        "evidence_ids": ("run-1:evidence-1",),
        "evidence_categories": ("analytics",),
    }
    serialized = trace.model_dump_json()
    assert "private user question" not in serialized
    assert "Grounded answer" not in serialized
    assert "sk-test-secret" not in serialized


def test_repair_and_strict_block_are_separate_from_provider_failure() -> None:
    repair = build_run_trace(
        _result(
            validation_outcome=ValidationAction.REPAIR,
            repair_count=1,
            citation_repairs=(
                CitationRepairDiagnostic(
                    original_id="run-missing:evidence-1",
                    canonical_id="run-1:evidence-1",
                    strategy="unique_same_ordinal_single_prefix_edit",
                ),
            ),
            validation_mode=ValidationMode.BALANCED,
        )
    )
    block = build_run_trace(
        _result(
            status=GroundedAnswerStatus.ERROR,
            validation_outcome=ValidationAction.BLOCK,
            block_count=1,
        )
    )

    assert repair.validation_repair_applied is True
    assert repair.validation_repair_count == 1
    assert repair.validation_citation_repairs[0].model_dump() == {
        "original_id": "run-missing:evidence-1",
        "canonical_id": "run-1:evidence-1",
        "strategy": "unique_same_ordinal_single_prefix_edit",
    }
    assert repair.validation_mode == "balanced"
    assert repair.success is True
    assert block.validation_block_count == 1
    assert block.error_types == ("validation_block",)
    assert block.provider_failure is False


def test_web_and_non_web_retrieval_fields_are_distinct() -> None:
    web = build_run_trace(_result(web=True))
    no_web = build_run_trace(_result(web=False))

    assert web.web_search_executed is True
    assert web.web_query_hash is not None
    assert web.web_query_length == len("private current query")
    assert web.web_selected_domains == ("example.com",)
    assert no_web.web_search_required is False
    assert no_web.web_search_executed is False
    assert no_web.web_query_hash is None
    assert no_web.web_latency_ms is None


def test_configured_pricing_is_deterministic_and_unknown_pricing_is_null() -> None:
    registry = PricingRegistry(
        version="test-pricing-v1",
        entries=(
            ModelPricing(
                provider="openai",
                model="planner-test",
                input_usd_per_million_tokens=2.0,
                output_usd_per_million_tokens=4.0,
            ),
            ModelPricing(
                provider="openai",
                model="synthesis-test",
                input_usd_per_million_tokens=1.0,
                output_usd_per_million_tokens=3.0,
            ),
        ),
    )
    priced = build_run_trace(_result(), pricing_registry=registry)
    unknown = build_run_trace(_result())

    assert priced.estimated_planner_cost_usd == pytest.approx(0.0004)
    assert priced.estimated_synthesis_cost_usd == pytest.approx(0.00044)
    assert priced.estimated_total_cost_usd == pytest.approx(0.00084)
    assert priced.pricing_version == "test-pricing-v1"
    assert unknown.estimated_planner_cost_usd is None
    assert unknown.estimated_total_cost_usd is None
    assert unknown.pricing_version is None


def test_default_pricing_version_propagates_to_trace_and_aggregation() -> None:
    registry = load_default_pricing_registry()
    base = _result(web=False)
    priced_result = base.model_copy(
        update={
            "diagnostics": base.diagnostics.model_copy(
                update={
                    "model": "gpt-5-mini",
                    "synthesis_model": "gpt-5-mini",
                }
            )
        }
    )

    trace = build_run_trace(priced_result, pricing_registry=registry)
    summary = summarize_traces([trace])

    assert trace.estimated_planner_cost_usd is not None
    assert trace.estimated_synthesis_cost_usd is not None
    assert trace.estimated_total_cost_usd == pytest.approx(
        trace.estimated_planner_cost_usd + trace.estimated_synthesis_cost_usd
    )
    assert trace.pricing_version == registry.version
    assert summary.pricing_versions == {"openai-2026-10-03": 1}
    assert summary.total_estimated_cost_usd == trace.estimated_total_cost_usd


def test_default_gpt_5_mini_pricing_and_cached_input_are_applied() -> None:
    registry = load_default_pricing_registry()
    pricing = registry.find(" OpenAI ", "GPT-5-MINI")

    assert registry.version == "openai-2026-10-03"
    assert DEFAULT_PRICING_REGISTRY_PATH.is_absolute()
    assert pricing is not None
    assert estimate_cost_usd(
        input_tokens=1000,
        cached_input_tokens=400,
        output_tokens=100,
        pricing=pricing,
    ) == pytest.approx(0.00036)


def test_default_pricing_load_is_independent_of_working_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    registry = load_default_pricing_registry()

    assert registry.version == "openai-2026-10-03"
    assert registry.find("openai", "gpt-5-mini") is not None


def test_evaluation_runner_uses_default_pricing_outside_backend_cwd(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    registry = _pricing_registry(None)

    assert registry.version == "openai-2026-10-03"
    assert registry.find("openai", "gpt-5-mini") is not None


def test_pricing_stays_null_for_unknown_model_or_missing_usage() -> None:
    registry = load_default_pricing_registry()
    known = registry.find("openai", "gpt-5-mini")

    assert registry.find("openai", "unknown-model") is None
    assert estimate_cost_usd(
        input_tokens=10,
        output_tokens=4,
        pricing=None,
    ) is None
    assert estimate_cost_usd(
        input_tokens=None,
        output_tokens=4,
        pricing=known,
    ) is None


def test_deterministic_synthesis_is_not_counted_as_provider_usage() -> None:
    trace = build_run_trace(
        _result(
            synthesis_usage=None,
            synthesis_provider_attempts=0,
            web=False,
        )
    )

    assert trace.synthesis_latency_ms == 20.0
    assert trace.planner_provider_call_count == 1
    assert trace.synthesis_provider_call_count == 0
    assert trace.llm_provider_call_count == 1
    assert trace.synthesis_total_tokens is None
    assert trace.total_tokens == trace.planner_total_tokens == 150


def test_clarification_without_synthesis_counts_only_actual_planner_call() -> None:
    trace = build_run_trace(
        _result(
            status=GroundedAnswerStatus.CLARIFICATION_REQUIRED,
            context_status=None,
            synthesis_usage=None,
            synthesis_provider_attempts=0,
            web=False,
        )
    )

    assert trace.synthesis_latency_ms is None
    assert trace.planner_provider_call_count == 1
    assert trace.synthesis_provider_call_count == 0
    assert trace.llm_provider_call_count == 1
    assert trace.total_tokens == 150


def test_aggregate_usage_uses_one_complete_provider_call_population() -> None:
    planner_only = build_run_trace(
        _result(
            synthesis_usage=None,
            synthesis_provider_attempts=0,
            web=False,
        )
    )
    planner_and_synthesis = build_run_trace(_result(web=False)).model_copy(
        update={"run_id": "run-2"}
    )

    summary = summarize_traces([planner_only, planner_and_synthesis])

    assert summary.planner_tokens == 300
    assert summary.synthesis_tokens == 280
    assert summary.total_tokens == 580
    assert summary.total_tokens == summary.planner_tokens + summary.synthesis_tokens
    assert summary.observed_total_tokens == 580
    assert summary.usage_observed_llm_calls == 3
    assert summary.usage_coverage_rate == 1.0
    assert summary.usage_complete is True
    assert summary.total_llm_provider_calls == 3
    assert summary.total_web_provider_calls == 0
    assert summary.total_provider_calls == 3


def test_aggregate_usage_does_not_partially_sum_missing_actual_call_usage() -> None:
    missing = build_run_trace(
        _result(
            synthesis_usage=None,
            synthesis_provider_attempts=1,
            web=False,
        )
    )

    summary = summarize_traces([missing])

    assert missing.total_tokens is None
    assert summary.planner_tokens == 150
    assert summary.synthesis_tokens is None
    assert summary.total_tokens is None
    assert summary.observed_total_input_tokens == 100
    assert summary.observed_total_output_tokens == 50
    assert summary.observed_total_tokens == 150
    assert summary.usage_observed_llm_calls == 1
    assert summary.total_llm_provider_calls == 2
    assert summary.usage_coverage_rate == 0.5
    assert summary.usage_complete is False
    assert summary.mean_tokens_per_run is None
    assert summary.median_tokens_per_run is None


def test_partial_aggregate_reports_observed_cost_and_call_coverage() -> None:
    partial = build_run_trace(
        _result(
            synthesis_usage=None,
            synthesis_provider_attempts=1,
            web=False,
        )
    ).model_copy(
        update={
            "estimated_planner_cost_usd": 0.0025,
            "estimated_synthesis_cost_usd": None,
            "estimated_total_cost_usd": None,
            "pricing_version": "test-pricing-v1",
        }
    )

    summary = summarize_traces([partial])

    assert summary.total_estimated_cost_usd is None
    assert summary.observed_estimated_cost_usd == pytest.approx(0.0025)
    assert summary.cost_observed_llm_calls == 1
    assert summary.total_llm_provider_calls == 2
    assert summary.cost_coverage_rate == 0.5
    assert summary.cost_complete is False


def test_jsonl_sink_appends_one_schema_valid_trace_per_line(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    sink = JsonlTraceSink(path)
    first = build_run_trace(_result(), case_id="case-1")
    second = first.model_copy(update={"run_id": "run-2", "case_id": "case-2"})

    sink.record(first)
    sink.record(second)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["trace_schema_version"] == "ai-scout-trace-v1"
    assert [trace.case_id for trace in load_jsonl_traces(path)] == ["case-1", "case-2"]


def test_trace_sink_failure_isolated_from_completed_result() -> None:
    class FailingSink:
        def record(self, trace) -> None:
            del trace
            raise OSError("disk unavailable")

    result = _result()
    trace = build_run_trace(result)
    assert record_trace_safely(FailingSink(), trace) is False
    assert result.answer.status is GroundedAnswerStatus.ANSWERED
    memory = InMemoryTraceSink()
    assert record_trace_safely(memory, trace) is True
    assert memory.traces == [trace]


def test_aggregate_summary_is_deterministic() -> None:
    first = build_run_trace(_result())
    second = first.model_copy(
        update={
            "run_id": "run-2",
            "success": False,
            "workflow_completed": False,
            "terminal_status": "error",
            "total_latency_ms": 135.0,
            "planner_latency_ms": 30.0,
            "total_tokens": 570,
            "validation_block_count": 1,
            "validation_outcome": "block",
            "validation_rule_frequencies": {"citation_required": 2},
        }
    )
    summary = summarize_traces([first, second])

    assert summary.total_runs == 2
    assert summary.successful_runs == 1
    assert summary.terminal_status_counts == {"answered": 1, "error": 1}
    assert summary.total_latency.mean == 100.0
    assert summary.total_latency.p50 == 100.0
    assert summary.total_latency.p95 == 135.0
    assert summary.total_latency.max == 135.0
    assert summary.total_tokens == 860
    assert summary.total_tokens == summary.planner_tokens + summary.synthesis_tokens
    assert summary.median_tokens_per_run == 500
    assert summary.tool_execution_frequency == {"get_role_fit": 2}
    assert summary.validation_outcome_counts == {"block": 1, "pass": 1}
    assert summary.validation_reached_runs == 2
    assert summary.validation_pass_rate == 0.5
    assert summary.validation_rule_frequencies == {"citation_required": 3}


def test_validation_not_reached_is_separate_and_excluded_from_pass_rate() -> None:
    passed = build_run_trace(_result(validation_outcome=ValidationAction.PASS))
    not_reached = build_run_trace(
        _result(
            status=GroundedAnswerStatus.ERROR,
            context_status=None,
            synthesis_usage=None,
            web=False,
            validation_outcome=None,
        )
    )

    summary = summarize_traces([passed, not_reached])

    assert summary.validation_outcome_counts == {"not_reached": 1, "pass": 1}
    assert summary.validation_reached_runs == 1
    assert summary.validation_pass_rate == 1.0


def test_all_validation_not_reached_has_no_pass_rate_denominator() -> None:
    trace = build_run_trace(
        _result(
            status=GroundedAnswerStatus.ERROR,
            context_status=None,
            synthesis_usage=None,
            web=False,
            validation_outcome=None,
        )
    )

    summary = summarize_traces([trace])

    assert summary.validation_outcome_counts == {"not_reached": 1}
    assert summary.validation_reached_runs == 0
    assert summary.validation_pass_rate is None


def test_case_id_is_trace_only_and_public_contract_is_unchanged() -> None:
    result = _result()
    trace = build_run_trace(result, case_id="golden-001")
    public = public_ai_scout_response(result).model_dump()

    assert trace.case_id == "golden-001"
    assert "case_id" not in public
    assert "total_tokens" not in public
    assert "validation_outcome" not in public
