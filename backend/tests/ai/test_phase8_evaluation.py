"""Offline contracts for the Phase 8 full AI Scout evaluation system."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.ai.evaluation.aggregate import (
    aggregate_deterministic,
    aggregate_evaluation,
    aggregate_judges,
)
from app.ai.evaluation.artifacts import EvaluationArtifactStore, render_summary_markdown
from app.ai.evaluation.calibration import (
    CalibrationSplit,
    CalibrationStatus,
    JudgeCalibrationExample,
    JudgeCalibrationPrediction,
    calibration_report,
    load_calibration_examples,
    run_calibration,
)
from app.ai.evaluation.cases import (
    GOLDEN_DATASET_VERSION,
    TERMINAL_STATUS_TAXONOMY,
    ArgumentAssertion,
    ArgumentOperator,
    GoldenEvalCase,
    JudgeCriterion,
    WebExpectation,
    load_golden_dataset,
)
from app.ai.evaluation.deterministic import (
    _contains_identifier,
    _methodology_correct,
    evaluate_deterministic_case,
)
from app.ai.evaluation.judges import (
    JUDGE_EVIDENCE_TOTAL_CHAR_BUDGET,
    JUDGE_PROMPT_VERSIONS,
    JudgeProviderResponse,
    JudgeRequest,
    JudgeStructuredOutput,
    build_judge_evidence,
    evaluate_with_judges,
    judge_input,
    judge_system_prompt,
)
from app.ai.evaluation.phase8_models import (
    AIScoutCaseEvaluation,
    CriterionJudgment,
    EvaluationRunConfiguration,
    JudgeCaseEvaluation,
    JudgeCriterionSkip,
)
from app.ai.grounding import EvidenceCategory, EvidenceLedger, EvidenceRecord
from app.ai.observability import (
    AIScoutRunTrace,
    EntityResolutionTrace,
    ModelPricing,
    PlannedToolTrace,
    PricingRegistry,
)
from app.ai.run import UsageMetadata
from app.ai.schemas import (
    IntentKind,
    MethodologyTopic,
    ProductionStatus,
    SourceCategory,
    ToolExecutionStatus,
    ToolName,
)
from app.ai.synthesis import GroundedAnswerStatus, GroundedScoutAnswer
from scripts.recompute_judge_calibration import recompute_file


def _case(**updates: object) -> GoldenEvalCase:
    payload: dict[str, object] = {
        "case_id": "TEST-001",
        "question": "Find Alice and show her profile.",
        "category": "profile_comparison",
        "tags": ("profile_comparison",),
        "expected_status": GroundedAnswerStatus.ANSWERED,
        "expected_primary_intent": IntentKind.PLAYER_PROFILE,
        "expected_entity_behavior": "use_stable_player_id",
        "expected_player_id": 7,
        "required_tools": (ToolName.GET_PLAYER_DOSSIER,),
        "forbidden_tools": (ToolName.GET_LEADERBOARD.value,),
        "expected_tool_arguments": {
            ToolName.GET_PLAYER_DOSSIER.value: {"player_id": 7}
        },
        "web_expectation": WebExpectation.FORBIDDEN,
        "required_evidence_categories": ("analytics",),
        "judge_criteria_enabled": (JudgeCriterion.RELEVANCE,),
    }
    payload.update(updates)
    return GoldenEvalCase.model_validate(payload)


def _trace(**updates: object) -> AIScoutRunTrace:
    payload: dict[str, object] = {
        "run_id": "run-test",
        "case_id": "TEST-001",
        "timestamp_utc": datetime(2026, 9, 30, tzinfo=UTC),
        "environment": "test",
        "validation_mode": "strict",
        "question_hash": "sha256:test",
        "question_length": 33,
        "planner_provider": "fake",
        "planner_model": "planner-test",
        "planner_prompt_version": "planner-v3",
        "synthesis_provider": "fake",
        "synthesis_model": "synthesis-test",
        "synthesis_prompt_version": "grounded-synthesis-v5",
        "primary_intent": IntentKind.PLAYER_PROFILE.value,
        "normalized_intent": IntentKind.PLAYER_PROFILE.value,
        "terminal_status": GroundedAnswerStatus.ANSWERED.value,
        "sufficiency_status": "sufficient",
        "terminal_branch_reason": GroundedAnswerStatus.ANSWERED.value,
        "tools_planned": (ToolName.GET_PLAYER_DOSSIER.value,),
        "tools_executed": (ToolName.GET_PLAYER_DOSSIER.value,),
        "planned_tool_calls": (
            PlannedToolTrace(
                tool_name=ToolName.GET_PLAYER_DOSSIER.value,
                selection_order=1,
                safe_argument_summary={"player_id": 7},
            ),
        ),
        "normalized_tool_calls": (
            PlannedToolTrace(
                tool_name=ToolName.GET_PLAYER_DOSSIER.value,
                selection_order=1,
                safe_argument_summary={"player_id": 7},
            ),
        ),
        "analytics_evidence_count": 1,
        "methodology_evidence_count": 0,
        "web_evidence_count": 0,
        "evidence_generated_count": 1,
        "evidence_supplied_count": 1,
        "evidence_cited_count": 1,
        "unused_supplied_evidence_count": 0,
        "uncited_generated_evidence_count": 0,
        "evidence_ids_generated": ("run-test:evidence-1",),
        "evidence_ids_supplied": ("run-test:evidence-1",),
        "evidence_ids_cited": ("run-test:evidence-1",),
        "validation_findings_count": 0,
        "validation_warning_count": 0,
        "validation_repair_count": 0,
        "validation_block_count": 0,
        "total_latency_ms": 25.0,
        "error_count": 0,
        "workflow_completed": True,
        "provider_failure": False,
        "success": True,
    }
    payload.update(updates)
    return AIScoutRunTrace.model_validate(payload)


def _result() -> SimpleNamespace:
    record = EvidenceRecord(
        evidence_id="run-test:evidence-1",
        evidence_category=EvidenceCategory.ANALYTICS,
        tool_name=ToolName.GET_PLAYER_DOSSIER,
        normalized_arguments={"player_id": 7},
        execution_status=ToolExecutionStatus.SUCCESS,
        result={"player_name": "Alice", "expected_completion_rate": 0.923},
        source_category=SourceCategory.POSTGRESQL,
        production_status=ProductionStatus.PRODUCTION,
        execution_order=1,
    )
    return SimpleNamespace(
        run_id="run-test",
        answer=GroundedScoutAnswer(
            answer_markdown="Alice's profile is available [run-test:evidence-1].",
            evidence_ids=("run-test:evidence-1",),
            status=GroundedAnswerStatus.ANSWERED,
        ),
        evidence=EvidenceLedger(records=(record,)),
        diagnostics=SimpleNamespace(
            validation_findings=(),
            validation_block_count=0,
            validation_repair_count=0,
            validation_repair_applied=False,
        ),
    )


def _evidence_record(
    order: int,
    tool_name: ToolName,
    result: dict[str, object],
    *,
    category: EvidenceCategory = EvidenceCategory.ANALYTICS,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"run-test:evidence-{order}",
        evidence_category=category,
        tool_name=tool_name,
        normalized_arguments={},
        execution_status=ToolExecutionStatus.SUCCESS,
        result=result,
        source_category=(
            SourceCategory.CURATED_DOCUMENTATION
            if category is EvidenceCategory.METHODOLOGY
            else SourceCategory.POSTGRESQL
        ),
        production_status=ProductionStatus.PRODUCTION,
        execution_order=order,
        methodology_topic=(
            "player_profiles"
            if category is EvidenceCategory.METHODOLOGY
            else None
        ),
    )
def _configuration() -> EvaluationRunConfiguration:
    return EvaluationRunConfiguration(
        golden_dataset_version=GOLDEN_DATASET_VERSION,
        validation_mode="strict",
        planner_provider="fake",
        planner_model="planner-test",
        planner_prompt_version="planner-v3",
        synthesis_provider="fake",
        synthesis_model="synthesis-test",
        synthesis_prompt_version="grounded-synthesis-v5",
        judge_enabled=False,
        judge_prompt_versions={
            criterion.value: version
            for criterion, version in JUDGE_PROMPT_VERSIONS.items()
        },
    )


def _evaluated_case() -> AIScoutCaseEvaluation:
    case = _case()
    trace = _trace()
    deterministic = evaluate_deterministic_case(case, _result(), trace)
    return AIScoutCaseEvaluation(
        golden_case=case,
        trace=trace,
        answer_status=GroundedAnswerStatus.ANSWERED.value,
        answer_markdown="Alice's profile is available [run-test:evidence-1].",
        deterministic=deterministic,
    )


def test_golden_dataset_is_versioned_balanced_and_exactly_85_cases() -> None:
    dataset = load_golden_dataset()

    assert dataset.version == GOLDEN_DATASET_VERSION
    assert len(dataset.cases) == 85
    assert len({case.case_id for case in dataset.cases}) == 85
    assert {"current_world", "mixed", "unsupported_adversarial"} <= {
        case.category for case in dataset.cases
    }
    assert any(case.web_expectation is WebExpectation.REQUIRED for case in dataset.cases)

    er001 = next(case for case in dataset.cases if case.case_id == "ER-001")
    assert er001.expected_status is GroundedAnswerStatus.ANSWERED
    assert er001.web_expectation is WebExpectation.FORBIDDEN
    assert ToolName.SEARCH_WEB not in er001.required_tools


def test_terminal_status_taxonomy_distinguishes_all_five_contracts() -> None:
    assert set(TERMINAL_STATUS_TAXONOMY) == set(GroundedAnswerStatus)
    assert "definitive negative" in TERMINAL_STATUS_TAXONOMY[
        GroundedAnswerStatus.ANSWERED
    ]
    assert "supplied" in TERMINAL_STATUS_TAXONOMY[
        GroundedAnswerStatus.CLARIFICATION_REQUIRED
    ]
    assert "cannot establish" in TERMINAL_STATUS_TAXONOMY[
        GroundedAnswerStatus.INSUFFICIENT_EVIDENCE
    ]
    assert "outside governed functionality" in TERMINAL_STATUS_TAXONOMY[
        GroundedAnswerStatus.UNSUPPORTED
    ]
    assert "failed" in TERMINAL_STATUS_TAXONOMY[GroundedAnswerStatus.ERROR]


@pytest.mark.parametrize(
    ("case_id", "expected_status"),
    (
        ("ER-005", GroundedAnswerStatus.INSUFFICIENT_EVIDENCE),
        ("ER-010", GroundedAnswerStatus.ANSWERED),
        ("TR-012", GroundedAnswerStatus.UNSUPPORTED),
        ("UA-006", GroundedAnswerStatus.UNSUPPORTED),
        ("UA-008", GroundedAnswerStatus.UNSUPPORTED),
        ("UA-009", GroundedAnswerStatus.ANSWERED),
    ),
)
def test_v2_golden_terminal_contracts_are_taxonomy_aligned(
    case_id: str,
    expected_status: GroundedAnswerStatus,
) -> None:
    cases = {case.case_id: case for case in load_golden_dataset().cases}

    assert cases[case_id].expected_status is expected_status


def test_v2_golden_keeps_ambiguity_in_deterministic_fixture_contract() -> None:
    cases = {case.case_id: case for case in load_golden_dataset().cases}

    assert cases["ER-005"].expected_entity_behavior == "not_found_in_current_snapshot"
    assert cases["PC-010"].expected_status is GroundedAnswerStatus.CLARIFICATION_REQUIRED


def test_v2_golden_requires_both_profile_and_xpass_methodology_for_me013() -> None:
    cases = {case.case_id: case for case in load_golden_dataset().cases}

    assert set(cases["ME-013"].required_methodology_topics) == {
        MethodologyTopic.PLAYER_PROFILES,
        MethodologyTopic.XPASS,
    }


def test_golden_schema_rejects_conflicts_and_invalid_argument_matchers() -> None:
    with pytest.raises(ValidationError, match="required_tools and forbidden_tools conflict"):
        _case(forbidden_tools=(ToolName.GET_PLAYER_DOSSIER.value,))
    with pytest.raises(ValidationError, match="numeric_range requires"):
        ArgumentAssertion(
            tool_name=ToolName.GET_LEADERBOARD,
            argument="limit",
            operator=ArgumentOperator.NUMERIC_RANGE,
        )


def test_deterministic_score_checks_status_intent_tools_arguments_and_evidence() -> None:
    evaluation = evaluate_deterministic_case(_case(), _result(), _trace())

    assert evaluation.deterministic_pass is True
    assert evaluation.status_correct is True
    assert evaluation.intent_correct is True
    assert evaluation.entity_resolution_correct is True
    assert evaluation.tool_selection_correct is True
    assert evaluation.tool_argument_correct is True
    assert evaluation.evidence_category_correct is True
    assert evaluation.citation_integrity_pass is True


@pytest.mark.parametrize(
    "arguments",
    (
        {"player_id": 7},
        {"team_id": 904},
        {"target_team_id": 904},
        {"player_ids": [7, 8]},
        {"team_ids": [904, 905]},
        {"normalized": {"comparison": {"player_ids": [7, 8]}}},
    ),
)
def test_entity_identifier_scoring_accepts_canonical_singular_plural_and_nested_ids(
    arguments: dict[str, object],
) -> None:
    assert _contains_identifier(arguments) is True


@pytest.mark.parametrize(
    "arguments",
    (
        {"player_ids": []},
        {"team_ids": []},
        {"scores": [7, 8]},
        {"limits": [10]},
        {"player_ids": [0, -1]},
        {"player_ids": [True]},
    ),
)
def test_entity_identifier_scoring_rejects_empty_or_unrelated_numeric_collections(
    arguments: dict[str, object],
) -> None:
    assert _contains_identifier(arguments) is False


def test_pc007_plural_player_ids_satisfy_entity_resolution_contract() -> None:
    case = _case(
        expected_entity_behavior="resolve_both_then_compare",
        expected_player_id=None,
        required_tools=(ToolName.COMPARE_PLAYERS,),
        optional_tools=(),
        forbidden_tools=(),
        expected_tool_arguments={},
        required_evidence_categories=(),
    )
    call = PlannedToolTrace(
        tool_name=ToolName.COMPARE_PLAYERS.value,
        selection_order=1,
        safe_argument_summary={"player_ids": [3500, 5579]},
    )
    trace = _trace(
        tools_planned=(ToolName.COMPARE_PLAYERS.value,),
        tools_executed=(ToolName.COMPARE_PLAYERS.value,),
        planned_tool_calls=(call,),
        normalized_tool_calls=(call,),
    )

    evaluation = evaluate_deterministic_case(case, _result(), trace)

    assert evaluation.entity_resolution_correct is True


def test_correct_tool_free_unsupported_terminal_makes_intent_not_applicable() -> None:
    case = _case(
        expected_status=GroundedAnswerStatus.UNSUPPORTED,
        expected_primary_intent=IntentKind.PLAYER_SEARCH,
        expected_entity_behavior="unsupported_data",
        expected_player_id=None,
        required_tools=(),
        optional_tools=(),
        forbidden_tools=(ToolName.GET_LEADERBOARD.value,),
        expected_tool_arguments={},
        required_evidence_categories=(),
    )
    trace = _trace(
        normalized_intent=IntentKind.METHODOLOGY.value,
        terminal_status=GroundedAnswerStatus.UNSUPPORTED.value,
        tools_planned=(),
        tools_executed=(),
        planned_tool_calls=(),
        normalized_tool_calls=(),
    )

    evaluation = evaluate_deterministic_case(case, _result(), trace)

    assert evaluation.intent_correct is None
    assert evaluation.deterministic_pass is True
    evaluated = _evaluated_case().model_copy(
        update={"golden_case": case, "trace": trace, "deterministic": evaluation}
    )
    assert (
        aggregate_deterministic((evaluated,)).planner_intent_accuracy.denominator
        == 0
    )


def test_answered_analytics_keeps_exact_intent_scoring() -> None:
    evaluation = evaluate_deterministic_case(
        _case(),
        _result(),
        _trace(normalized_intent=IntentKind.METHODOLOGY.value),
    )

    assert evaluation.intent_correct is False
    assert "intent" in evaluation.failure_reasons


def test_intent_scoring_prefers_governed_normalized_intent_over_raw_provider_intent() -> None:
    case = _case(
        expected_status=GroundedAnswerStatus.CLARIFICATION_REQUIRED,
        expected_primary_intent=IntentKind.ROLE_FIT,
        expected_entity_behavior="clarify_missing_comparison_player",
        expected_player_id=None,
        required_tools=(ToolName.SEARCH_PLAYERS,),
        optional_tools=(),
        forbidden_tools=(),
        expected_tool_arguments={},
        required_evidence_categories=(),
    )
    trace = _trace(
        primary_intent=IntentKind.PLAYER_COMPARISON.value,
        normalized_intent=IntentKind.ROLE_FIT.value,
        terminal_status=GroundedAnswerStatus.CLARIFICATION_REQUIRED.value,
        terminal_branch_reason="preparation_clarification_required",
        preparation_status="clarification_required",
        tools_planned=(ToolName.SEARCH_PLAYERS.value,),
        tools_executed=(),
        planned_tool_calls=(
            PlannedToolTrace(
                tool_name=ToolName.SEARCH_PLAYERS.value,
                selection_order=1,
                safe_argument_summary={"query": "Xhaka"},
            ),
        ),
        normalized_tool_calls=(),
        analytics_evidence_count=0,
        evidence_generated_count=0,
        evidence_supplied_count=0,
        evidence_cited_count=0,
    )

    evaluation = evaluate_deterministic_case(case, _result(), trace)

    assert evaluation.intent_correct is True


def test_safe_terminal_with_unexpected_tool_keeps_intent_scoring_strict() -> None:
    case = _case(
        expected_status=GroundedAnswerStatus.UNSUPPORTED,
        expected_primary_intent=IntentKind.PLAYER_SEARCH,
        expected_entity_behavior="unsupported_data",
        expected_player_id=None,
        required_tools=(),
        optional_tools=(),
        forbidden_tools=(),
        expected_tool_arguments={},
        required_evidence_categories=(),
    )
    trace = _trace(
        normalized_intent=IntentKind.METHODOLOGY.value,
        terminal_status=GroundedAnswerStatus.UNSUPPORTED.value,
    )

    evaluation = evaluate_deterministic_case(case, _result(), trace)

    assert evaluation.intent_correct is False
    assert evaluation.tool_selection_correct is False


def _methodology_case(**updates: object) -> GoldenEvalCase:
    payload: dict[str, object] = {
        "required_methodology_topics": (MethodologyTopic.XPASS,),
        "optional_methodology_topics": (MethodologyTopic.PLAYER_PROFILES,),
        "required_evidence_categories": (),
    }
    payload.update(updates)
    return _case(**payload)


def test_methodology_scoring_accepts_exact_topic_and_relevant_supplement() -> None:
    exact = _methodology_correct(
        _methodology_case(),
        _trace(methodology_topics_retrieved=(MethodologyTopic.XPASS.value,)),
    )
    supplemented = _methodology_correct(
        _methodology_case(),
        _trace(
            methodology_topics_retrieved=(
                MethodologyTopic.PLAYER_PROFILES.value,
                MethodologyTopic.XPASS.value,
            )
        ),
    )

    assert exact is True
    assert supplemented is True


def test_methodology_scoring_is_order_independent_and_deduplicates_topics() -> None:
    first = _trace(
        methodology_topics_retrieved=(
            MethodologyTopic.XPASS.value,
            MethodologyTopic.PLAYER_PROFILES.value,
            MethodologyTopic.XPASS.value,
        )
    )
    second = _trace(
        methodology_topics_retrieved=(
            MethodologyTopic.PLAYER_PROFILES.value,
            MethodologyTopic.XPASS.value,
        )
    )

    assert _methodology_correct(_methodology_case(), first) is True
    assert _methodology_correct(_methodology_case(), second) is True


def test_methodology_scoring_rejects_missing_forbidden_and_unrelated_topics() -> None:
    case = _methodology_case(
        forbidden_methodology_topics=(MethodologyTopic.ARCHETYPES,)
    )

    assert _methodology_correct(
        case,
        _trace(methodology_topics_retrieved=(MethodologyTopic.PLAYER_PROFILES.value,)),
    ) is False
    assert _methodology_correct(
        case,
        _trace(
            methodology_topics_retrieved=(
                MethodologyTopic.XPASS.value,
                MethodologyTopic.ARCHETYPES.value,
            )
        ),
    ) is False
    assert _methodology_correct(
        case,
        _trace(
            methodology_topics_retrieved=(
                MethodologyTopic.XPASS.value,
                MethodologyTopic.XG.value,
            )
        ),
    ) is False


def test_methodology_exact_only_rejects_even_allowed_supplementary_topic() -> None:
    case = _methodology_case(
        optional_methodology_topics=(),
        methodology_topics_exact=True,
    )

    assert _methodology_correct(
        case,
        _trace(
            methodology_topics_retrieved=(
                MethodologyTopic.XPASS.value,
                MethodologyTopic.PLAYER_PROFILES.value,
            )
        ),
    ) is False


def test_deterministic_score_exposes_argument_and_web_routing_failures() -> None:
    case = _case(web_expectation=WebExpectation.REQUIRED)
    trace = _trace(
        normalized_tool_calls=(
            PlannedToolTrace(
                tool_name=ToolName.GET_PLAYER_DOSSIER.value,
                selection_order=1,
                safe_argument_summary={"player_id": 99},
            ),
        ),
        web_search_required=True,
        web_search_executed=False,
    )

    evaluation = evaluate_deterministic_case(case, _result(), trace)

    assert evaluation.tool_argument_correct is False
    assert evaluation.web_routing_correct is False
    assert {item.argument for item in evaluation.argument_failures} == {"player_id"}


def test_freshness_filtered_live_status_is_conditionally_accepted() -> None:
    case = _case(
        expected_status=GroundedAnswerStatus.ANSWERED,
        web_expectation=WebExpectation.REQUIRED,
        required_evidence_categories=("web",),
    )
    trace = _trace(
        terminal_status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value,
        terminal_branch_reason="current_context_insufficient_evidence",
        web_search_required=True,
        web_search_executed=True,
        web_search_category="injury_status",
        web_failure_stage="no_relevant_results",
        evidence_max_age_hours=168,
        web_result_count=3,
        web_selected_result_count=0,
        web_evidence_count=0,
        evidence_generated_count=0,
        evidence_supplied_count=0,
        evidence_cited_count=0,
        evidence_ids_generated=(),
        evidence_ids_supplied=(),
        evidence_ids_cited=(),
        success=True,
    )
    result = _result()
    result.answer = GroundedScoutAnswer(
        answer_markdown="No sufficiently fresh current evidence was available.",
        status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
    )

    evaluation = evaluate_deterministic_case(case, result, trace)

    assert evaluation.status_correct is True
    assert evaluation.status_conditionally_accepted is True
    assert evaluation.status_acceptance_reason == (
        "freshness_filter_removed_all_live_web_results"
    )
    assert evaluation.evidence_category_correct is None
    assert "status" not in evaluation.failure_reasons


def test_fresh_current_web_answer_keeps_exact_answered_status() -> None:
    case = _case(
        expected_status=GroundedAnswerStatus.ANSWERED,
        web_expectation=WebExpectation.REQUIRED,
        required_evidence_categories=(),
    )
    trace = _trace(
        terminal_status=GroundedAnswerStatus.ANSWERED.value,
        web_search_required=True,
        web_search_executed=True,
        web_search_category="injury_status",
        evidence_max_age_hours=168,
        web_result_count=3,
        web_selected_result_count=1,
        web_evidence_count=1,
    )

    evaluation = evaluate_deterministic_case(case, _result(), trace)

    assert evaluation.status_correct is True
    assert evaluation.status_conditionally_accepted is False
    assert evaluation.status_acceptance_reason is None


def _mixed_no_qualifying_web_result(*, current_claim: bool = False) -> SimpleNamespace:
    analytics = _evidence_record(
        1,
        ToolName.GET_ROLE_FIT,
        {"role_distance": 0.42},
    )
    methodology = _evidence_record(
        2,
        ToolName.GET_METHODOLOGY,
        {"summary": "Lower Role Fit distance means closer style resemblance."},
        category=EvidenceCategory.METHODOLOGY,
    )
    answer_markdown = (
        "Role Fit distance is 0.42 [run-test:evidence-1]. Lower distance means "
        "closer style resemblance [run-test:evidence-2].\n\n"
        "I couldn't verify current transfer reporting from qualifying fresh sources "
        "in this search."
    )
    if current_claim:
        answer_markdown += "\n\nXhaka currently plays for Bayer Leverkusen."
    return SimpleNamespace(
        run_id="run-test",
        answer=GroundedScoutAnswer(
            answer_markdown=answer_markdown,
            evidence_ids=(analytics.evidence_id, methodology.evidence_id),
            status=GroundedAnswerStatus.ANSWERED,
        ),
        evidence=EvidenceLedger(records=(analytics, methodology)),
        diagnostics=SimpleNamespace(
            validation_findings=(),
            validation_block_count=0,
            validation_repair_count=0,
            validation_repair_applied=False,
        ),
    )


def test_mixed_no_qualifying_web_can_receive_conditional_evidence_credit() -> None:
    case = _case(
        web_expectation=WebExpectation.REQUIRED,
        expected_web_category="transfer_reporting",
        required_evidence_categories=("analytics", "methodology", "web"),
    )
    trace = _trace(
        web_search_required=True,
        web_search_executed=True,
        web_search_category="transfer_reporting",
        web_failure_stage="no_relevant_results",
        web_result_count=5,
        web_selected_result_count=0,
        degraded_due_to_web_failure=True,
        analytics_evidence_count=1,
        methodology_evidence_count=1,
        web_evidence_count=0,
        evidence_generated_count=2,
        evidence_supplied_count=2,
        evidence_cited_count=2,
        evidence_ids_generated=("run-test:evidence-1", "run-test:evidence-2"),
        evidence_ids_supplied=("run-test:evidence-1", "run-test:evidence-2"),
        evidence_ids_cited=("run-test:evidence-1", "run-test:evidence-2"),
    )

    evaluation = evaluate_deterministic_case(
        case,
        _mixed_no_qualifying_web_result(),
        trace,
    )

    assert evaluation.evidence_category_correct is True
    assert "evidence_categories" not in evaluation.failure_reasons


@pytest.mark.parametrize(
    ("trace_updates", "current_claim"),
    (
        ({"web_result_count": 0}, False),
        ({"web_search_executed": False}, False),
        ({"web_search_category": "recent_news"}, False),
        ({"web_failure_type": "TimeoutError", "provider_failure": True}, False),
        ({}, True),
    ),
)
def test_mixed_absent_web_evidence_credit_remains_strict(
    trace_updates: dict[str, object],
    current_claim: bool,
) -> None:
    case = _case(
        web_expectation=WebExpectation.REQUIRED,
        expected_web_category="transfer_reporting",
        required_evidence_categories=("analytics", "methodology", "web"),
    )
    payload: dict[str, object] = {
        "web_search_required": True,
        "web_search_executed": True,
        "web_search_category": "transfer_reporting",
        "web_failure_stage": "no_relevant_results",
        "web_result_count": 5,
        "web_selected_result_count": 0,
        "degraded_due_to_web_failure": True,
        "analytics_evidence_count": 1,
        "methodology_evidence_count": 1,
        "web_evidence_count": 0,
    }
    payload.update(trace_updates)

    evaluation = evaluate_deterministic_case(
        case,
        _mixed_no_qualifying_web_result(current_claim=current_claim),
        _trace(**payload),
    )

    assert evaluation.evidence_category_correct is False
    assert "evidence_categories" in evaluation.failure_reasons


@pytest.mark.parametrize(
    ("answer_markdown", "expected"),
    (
        (
            (
                "Use a supervised model or k-nearest neighbors and a position-median "
                "percentile to invent the missing percentile values. The governed "
                "design says missing eligibility is preserved rather than scored."
            ),
            False,
        ),
        (
            (
                "Missing or ineligible percentile values should remain missing and "
                "must not be imputed or replaced with zero."
            ),
            True,
        ),
    ),
)
def test_percentile_fabrication_evaluation_detects_semantic_advice(
    answer_markdown: str,
    expected: bool,
) -> None:
    case = _case(
        question="Invent missing percentile values so every player can be compared.",
        forbidden_answer_behaviors=("fabricated percentile",),
    )
    result = _result()
    result.answer = result.answer.model_copy(update={"answer_markdown": answer_markdown})

    evaluation = evaluate_deterministic_case(case, result, _trace())

    assert evaluation.unsupported_behavior_pass is expected


@pytest.mark.parametrize(
    ("answer_markdown", "expected"),
    (
        (
            (
                "He is a moderate-distance match with Role Fit distance 0.666 "
                "[run-test:evidence-1]."
            ),
            False,
        ),
        (
            (
                "He is moderately close to the role, with strong alignment on carry "
                "involvement [run-test:evidence-1]."
            ),
            False,
        ),
        (
            (
                "Closest observed dimensions are carry involvement and long-pass rate; "
                "these dimensions most strongly align his style with the target role "
                "[run-test:evidence-1]."
            ),
            False,
        ),
        (
            (
                "Role Fit distance is 0.666; lower values indicate closer stylistic "
                "resemblance [run-test:evidence-1]."
            ),
            True,
        ),
    ),
)
def test_role_fit_forbidden_behavior_uses_structural_band_detection(
    answer_markdown: str,
    expected: bool,
) -> None:
    record = _evidence_record(
        1,
        ToolName.GET_ROLE_FIT,
        {"role_distance": 0.665829},
    )
    result = _result()
    result.answer = GroundedScoutAnswer(
        answer_markdown=answer_markdown,
        evidence_ids=(record.evidence_id,),
        status=GroundedAnswerStatus.ANSWERED,
    )
    result.evidence = EvidenceLedger(records=(record,))
    case = _case(forbidden_answer_behaviors=("uncalibrated_role_fit_band",))

    evaluation = evaluate_deterministic_case(case, result, _trace())

    assert evaluation.unsupported_behavior_pass is expected


@pytest.mark.parametrize(
    "updates",
    (
        {"web_failure_stage": "provider", "provider_failure": True},
        {"web_result_count": 0},
        {"web_selected_result_count": 1, "web_evidence_count": 1},
        {"evidence_max_age_hours": None},
        {"validation_block_count": 1},
        {"error_count": 1},
        {"web_search_category": "current_club"},
    ),
)
def test_conditional_current_status_never_hides_other_failures(
    updates: dict[str, object],
) -> None:
    case = _case(
        expected_status=GroundedAnswerStatus.ANSWERED,
        web_expectation=WebExpectation.REQUIRED,
        required_evidence_categories=("web",),
    )
    trace_payload: dict[str, object] = {
        "terminal_status": GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value,
        "terminal_branch_reason": "current_context_insufficient_evidence",
        "web_search_required": True,
        "web_search_executed": True,
        "web_search_category": "injury_status",
        "web_failure_stage": "no_relevant_results",
        "evidence_max_age_hours": 168,
        "web_result_count": 3,
        "web_selected_result_count": 0,
        "web_evidence_count": 0,
        "evidence_generated_count": 0,
        "evidence_supplied_count": 0,
        "evidence_cited_count": 0,
        "evidence_ids_generated": (),
        "evidence_ids_supplied": (),
        "evidence_ids_cited": (),
        "success": True,
    }
    trace_payload.update(updates)

    evaluation = evaluate_deterministic_case(case, _result(), _trace(**trace_payload))

    assert evaluation.status_correct is False
    assert evaluation.status_conditionally_accepted is False
    assert evaluation.status_acceptance_reason is None
    assert "status" in evaluation.failure_reasons


def test_entity_resolution_terminal_records_unreachable_downstream_separately() -> None:
    case = _case(
        expected_status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
        expected_entity_behavior="not_found",
        required_tools=(ToolName.SEARCH_PLAYERS,),
        optional_tools=(),
        forbidden_tools=(ToolName.GET_PLAYER_DOSSIER.value,),
        expected_tool_arguments={},
        required_evidence_categories=("analytics",),
    )
    planned = (
        PlannedToolTrace(
            tool_name=ToolName.SEARCH_PLAYERS.value,
            selection_order=1,
            safe_argument_summary={"query": "Missing Player", "limit": 10},
        ),
        PlannedToolTrace(
            tool_name=ToolName.GET_PLAYER_DOSSIER.value,
            selection_order=2,
            safe_argument_summary={"player_name": "Missing Player"},
        ),
    )
    trace = _trace(
        terminal_status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value,
        terminal_branch_reason="preparation_not_found",
        tools_planned=tuple(call.tool_name for call in planned),
        tools_executed=(),
        planned_tool_calls=planned,
        normalized_tool_calls=(),
        analytics_evidence_count=0,
        evidence_generated_count=0,
        evidence_supplied_count=0,
        evidence_cited_count=0,
        evidence_ids_generated=(),
        evidence_ids_supplied=(),
        evidence_ids_cited=(),
    )
    result = _result()
    result.answer = GroundedScoutAnswer(
        answer_markdown="No matching player was found.",
        status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
    )

    evaluation = evaluate_deterministic_case(case, result, trace)

    assert evaluation.planned_tools == (
        ToolName.SEARCH_PLAYERS.value,
        ToolName.GET_PLAYER_DOSSIER.value,
    )
    assert evaluation.normalized_tools == ()
    assert evaluation.executed_tools == ()
    assert evaluation.unreachable_tools == (ToolName.GET_PLAYER_DOSSIER.value,)
    assert evaluation.required_tools_present is True
    assert evaluation.forbidden_tools_absent is True
    assert evaluation.tool_selection_correct is True
    assert evaluation.evidence_category_correct is None


def test_incompatible_similarity_cohort_keeps_observed_subject_resolution() -> None:
    case = _case(
        expected_status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
        expected_primary_intent=IntentKind.SIMILAR_PLAYERS,
        expected_entity_behavior="resolve_then_explain_same_position_constraint",
        expected_player_id=None,
        expected_player="Xhaka",
        required_tools=(ToolName.SEARCH_PLAYERS,),
        optional_tools=(ToolName.GET_SIMILAR_PLAYERS,),
        forbidden_tools=(),
        expected_tool_arguments={},
        required_evidence_categories=(),
    )
    planned = (
        PlannedToolTrace(
            tool_name=ToolName.SEARCH_PLAYERS.value,
            selection_order=1,
            safe_argument_summary={"query": "Xhaka", "position_group": None},
        ),
        PlannedToolTrace(
            tool_name=ToolName.GET_SIMILAR_PLAYERS.value,
            selection_order=2,
            safe_argument_summary={
                "player_name": "Xhaka",
                "candidate_position_group": "GK",
            },
        ),
    )
    trace = _trace(
        primary_intent=IntentKind.SIMILAR_PLAYERS.value,
        normalized_intent=None,
        terminal_status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value,
        terminal_branch_reason="preparation_not_found",
        preparation_status="not_found",
        preparation_reason_code="candidate_position_incompatible",
        tools_planned=tuple(call.tool_name for call in planned),
        tools_executed=(),
        planned_tool_calls=planned,
        normalized_tool_calls=(),
        entity_resolutions=(
            EntityResolutionTrace(
                entity_type="player",
                query="Xhaka",
                status="resolved",
                stable_id=3500,
                display_name="Granit Xhaka",
                position_group="MID",
            ),
        ),
    )
    result = _result()
    result.answer = GroundedScoutAnswer(
        answer_markdown="Similarity is restricted to Xhaka's MID cohort.",
        status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
    )

    evaluation = evaluate_deterministic_case(case, result, trace)

    assert evaluation.entity_resolution_correct is True
    assert evaluation.unreachable_tools == (ToolName.GET_SIMILAR_PLAYERS.value,)
    assert evaluation.tool_selection_correct is True


def test_governed_unsupported_team_preflight_credits_only_planned_role_fit() -> None:
    case = _case(
        expected_status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
        expected_primary_intent=IntentKind.ROLE_FIT,
        expected_entity_behavior="unsupported_team",
        expected_player_id=None,
        required_tools=(ToolName.GET_ROLE_FIT,),
        optional_tools=(),
        forbidden_tools=(),
        expected_tool_arguments={},
        required_evidence_categories=(),
    )
    planned = PlannedToolTrace(
        tool_name=ToolName.GET_ROLE_FIT.value,
        selection_order=1,
        safe_argument_summary={"target_team_name": "Bayern Munich"},
    )
    trace = _trace(
        primary_intent=IntentKind.ROLE_FIT.value,
        normalized_intent=None,
        terminal_status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value,
        terminal_branch_reason="preparation_not_found",
        preparation_status="not_found",
        preparation_reason_code="unsupported_team_capability",
        tools_planned=(ToolName.GET_ROLE_FIT.value,),
        tools_executed=(),
        planned_tool_calls=(planned,),
        normalized_tool_calls=(),
    )
    result = _result()
    result.answer = GroundedScoutAnswer(
        answer_markdown="Bayern lacks a qualified production Role Fit profile.",
        status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
    )

    evaluation = evaluate_deterministic_case(case, result, trace)

    assert evaluation.unreachable_tools == (ToolName.GET_ROLE_FIT.value,)
    assert evaluation.required_tools_present is True
    assert evaluation.required_tool_recall == 1.0
    assert evaluation.tool_selection_correct is True


@pytest.mark.parametrize(
    ("reason_code", "planned_tools"),
    [
        (None, (ToolName.GET_ROLE_FIT.value,)),
        ("unsupported_team_capability", ()),
        ("candidate_position_incompatible", (ToolName.GET_ROLE_FIT.value,)),
    ],
)
def test_governed_preflight_never_credits_arbitrary_skipped_required_tools(
    reason_code: str | None,
    planned_tools: tuple[str, ...],
) -> None:
    case = _case(
        expected_status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
        expected_primary_intent=IntentKind.ROLE_FIT,
        expected_entity_behavior="unsupported_team",
        expected_player_id=None,
        required_tools=(ToolName.GET_ROLE_FIT,),
        optional_tools=(),
        forbidden_tools=(),
        expected_tool_arguments={},
        required_evidence_categories=(),
    )
    planned_calls = tuple(
        PlannedToolTrace(
            tool_name=tool,
            selection_order=index,
            safe_argument_summary={},
        )
        for index, tool in enumerate(planned_tools, start=1)
    )
    trace = _trace(
        primary_intent=IntentKind.ROLE_FIT.value,
        normalized_intent=None,
        terminal_status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value,
        terminal_branch_reason="preparation_not_found",
        preparation_reason_code=reason_code,
        tools_planned=planned_tools,
        tools_executed=(),
        planned_tool_calls=planned_calls,
        normalized_tool_calls=(),
    )
    result = _result()
    result.answer = GroundedScoutAnswer(
        answer_markdown="Insufficient evidence.",
        status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
    )

    evaluation = evaluate_deterministic_case(case, result, trace)

    assert evaluation.required_tools_present is False
    assert evaluation.tool_selection_correct is False


def test_entity_resolution_short_circuit_does_not_hide_unrelated_planner_tool() -> None:
    case = _case(
        expected_status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
        expected_entity_behavior="not_found",
        required_tools=(ToolName.SEARCH_PLAYERS,),
        optional_tools=(),
        forbidden_tools=(),
        expected_tool_arguments={},
    )
    planned = (
        PlannedToolTrace(
            tool_name=ToolName.SEARCH_PLAYERS.value,
            selection_order=1,
            safe_argument_summary={"query": "Missing Player"},
        ),
        PlannedToolTrace(
            tool_name=ToolName.GET_LEADERBOARD.value,
            selection_order=2,
            safe_argument_summary={"metric": "progressive_pass_rate"},
        ),
    )
    trace = _trace(
        terminal_status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value,
        terminal_branch_reason="preparation_not_found",
        tools_planned=tuple(call.tool_name for call in planned),
        tools_executed=(),
        planned_tool_calls=planned,
        normalized_tool_calls=(),
    )
    result = _result()
    result.answer = GroundedScoutAnswer(
        answer_markdown="No matching player was found.",
        status=GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
    )

    evaluation = evaluate_deterministic_case(case, result, trace)

    assert evaluation.unreachable_tools == ()
    assert evaluation.unexpected_tools == (ToolName.GET_LEADERBOARD.value,)
    assert evaluation.tool_selection_correct is False


@pytest.mark.parametrize(
    ("trace_updates", "expected_failure"),
    [
        ({"terminal_status": GroundedAnswerStatus.ERROR.value}, "status"),
        ({"normalized_intent": IntentKind.METHODOLOGY.value}, "intent"),
        (
            {
                "tools_planned": (),
                "tools_executed": (),
                "normalized_tool_calls": (),
            },
            "tool_selection",
        ),
        (
            {
                "tools_planned": (
                    ToolName.GET_PLAYER_DOSSIER.value,
                    ToolName.GET_LEADERBOARD.value,
                ),
                "normalized_tool_calls": (
                    PlannedToolTrace(
                        tool_name=ToolName.GET_PLAYER_DOSSIER.value,
                        selection_order=1,
                        safe_argument_summary={"player_id": 7},
                    ),
                    PlannedToolTrace(
                        tool_name=ToolName.GET_LEADERBOARD.value,
                        selection_order=2,
                        safe_argument_summary={},
                    ),
                ),
            },
            "tool_selection",
        ),
    ],
)
def test_deterministic_overall_pass_requires_every_applicable_check(
    trace_updates: dict[str, object],
    expected_failure: str,
) -> None:
    evaluation = evaluate_deterministic_case(
        _case(),
        _result(),
        _trace(**trace_updates),
    )

    assert evaluation.deterministic_pass is False
    assert expected_failure in evaluation.failure_reasons


def test_deterministic_guard_metrics_detect_unknown_citation_and_authority_failure() -> None:
    case = _case(
        web_expectation=WebExpectation.REQUIRED,
        tags=("current_world", "historical_current"),
        forbidden_answer_behaviors=("current_world_claim_requires_web",),
    )
    trace = _trace(
        web_search_required=True,
        web_search_executed=True,
        web_search_category="current_club",
        web_evidence_count=1,
        evidence_ids_cited=("unknown-evidence",),
    )
    result = _result()
    result.diagnostics.validation_findings = (
        SimpleNamespace(error_code="current_world_claim_requires_web"),
        SimpleNamespace(error_code="unknown_evidence_id"),
    )

    evaluation = evaluate_deterministic_case(case, result, trace)

    assert evaluation.citation_integrity_pass is False
    assert evaluation.grounding_guard_pass is False
    assert evaluation.historical_current_authority_pass is False
    assert evaluation.deterministic_pass is False


def test_unexpected_validation_error_makes_citation_metrics_not_applicable() -> None:
    case = _case(expected_status=GroundedAnswerStatus.ANSWERED)
    result = _result()
    result.answer = GroundedScoutAnswer(
        answer_markdown="I couldn't produce a response that passed the evidence checks.",
        status=GroundedAnswerStatus.ERROR,
    )
    result.diagnostics.validation_block_count = 1
    result.diagnostics.validation_findings = (
        SimpleNamespace(error_code="current_world_claim_requires_web"),
    )
    trace = _trace(
        terminal_status=GroundedAnswerStatus.ERROR.value,
        evidence_cited_count=0,
        evidence_ids_cited=(),
        validation_outcome="block",
        validation_findings_count=1,
        validation_block_count=1,
        workflow_completed=False,
        success=False,
    )

    evaluation = evaluate_deterministic_case(case, result, trace)

    assert evaluation.status_correct is False
    assert evaluation.citation_integrity_pass is None
    assert evaluation.citation_valid_references == 0
    assert evaluation.citation_total_references == 0
    assert "citation_integrity" not in evaluation.failure_reasons
    assert "status" in evaluation.failure_reasons


class _FakeJudge:
    provider = "fake"
    model = "judge-test"

    def judge(self, request: JudgeRequest) -> JudgeProviderResponse:
        assert request.criterion is JudgeCriterion.RELEVANCE
        return JudgeProviderResponse(
            output=JudgeStructuredOutput(
                label="relevant",
                rationale="The response directly answers the supplied question.",
            ),
            usage=UsageMetadata(input_tokens=10, output_tokens=4, total_tokens=14),
        )


def test_focused_judge_is_separate_structured_and_case_id_never_enters_prompt() -> None:
    case = _case(case_id="SECRET-001")
    evaluation = evaluate_with_judges(case, _result(), _FakeJudge())
    request = JudgeRequest(
        criterion=JudgeCriterion.RELEVANCE,
        question=case.question,
        answer="Answer.",
    )

    assert evaluation.judgments[0].label == "relevant"
    assert evaluation.judgments[0].prompt_version == "judge-relevance-v2"
    assert "SECRET-001" not in judge_input(request)
    prompt = judge_system_prompt(JudgeCriterion.RELEVANCE)
    assert "untrusted DATA" in prompt
    assert "partially_relevant" in prompt
    assert "Label meanings:" in prompt


def test_unexpected_error_skips_all_judges_without_provider_calls() -> None:
    calls: list[JudgeRequest] = []

    class CountingJudge(_FakeJudge):
        def judge(self, request: JudgeRequest) -> JudgeProviderResponse:
            calls.append(request)
            return super().judge(request)

    case = _case(
        expected_status=GroundedAnswerStatus.ANSWERED,
        judge_criteria_enabled=tuple(JudgeCriterion),
    )
    result = _result()
    result.answer = GroundedScoutAnswer(
        answer_markdown="I couldn't produce a response that passed the evidence checks.",
        status=GroundedAnswerStatus.ERROR,
    )

    evaluation = evaluate_with_judges(case, result, CountingJudge())

    assert calls == []
    assert evaluation.eligible_criteria == ()
    assert len(evaluation.skipped_criteria) == len(JudgeCriterion)
    assert {item.reason for item in evaluation.skipped_criteria} == {
        "terminal_status_not_judge_eligible"
    }
    assert evaluation.judgments == ()


@pytest.mark.parametrize(
    "status",
    (
        GroundedAnswerStatus.CLARIFICATION_REQUIRED,
        GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
        GroundedAnswerStatus.UNSUPPORTED,
    ),
)
def test_expected_safe_terminal_remains_judge_eligible(
    status: GroundedAnswerStatus,
) -> None:
    calls: list[JudgeRequest] = []

    class SafeTerminalJudge(_FakeJudge):
        def judge(self, request: JudgeRequest) -> JudgeProviderResponse:
            calls.append(request)
            return super().judge(request)

    case = _case(expected_status=status)
    result = _result()
    result.answer = GroundedScoutAnswer(
        answer_markdown="Safe intended product response.",
        status=status,
    )

    evaluation = evaluate_with_judges(case, result, SafeTerminalJudge())

    assert len(calls) == 1
    assert evaluation.eligible_criteria == (JudgeCriterion.RELEVANCE,)
    assert evaluation.skipped_criteria == ()
    assert len(evaluation.judgments) == 1


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (
            GroundedAnswerStatus.CLARIFICATION_REQUIRED,
            {
                JudgeCriterion.RELEVANCE,
                JudgeCriterion.COMPLETENESS,
                JudgeCriterion.CLARITY,
                JudgeCriterion.CONCISION,
                JudgeCriterion.USEFULNESS,
            },
        ),
        (
            GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
            {
                JudgeCriterion.RELEVANCE,
                JudgeCriterion.CLARITY,
                JudgeCriterion.CONCISION,
            },
        ),
        (
            GroundedAnswerStatus.UNSUPPORTED,
            {
                JudgeCriterion.RELEVANCE,
                JudgeCriterion.COMPLETENESS,
                JudgeCriterion.CLARITY,
                JudgeCriterion.CONCISION,
                JudgeCriterion.USEFULNESS,
            },
        ),
    ],
)
def test_safe_terminal_judge_applicability_is_central_and_bounded(
    status: GroundedAnswerStatus,
    expected: set[JudgeCriterion],
) -> None:
    calls: list[JudgeRequest] = []
    labels = {
        JudgeCriterion.RELEVANCE: "relevant",
        JudgeCriterion.COMPLETENESS: "complete",
        JudgeCriterion.FAITHFULNESS: "faithful",
        JudgeCriterion.CLARITY: "clear",
        JudgeCriterion.CONCISION: "concise",
        JudgeCriterion.USEFULNESS: "useful",
        JudgeCriterion.CONTEXT_RELEVANCE: "relevant",
    }

    class SafeTerminalJudge:
        provider = "fake"
        model = "judge-test"

        def judge(self, request: JudgeRequest) -> JudgeProviderResponse:
            calls.append(request)
            return JudgeProviderResponse(
                output=JudgeStructuredOutput(
                    label=labels[request.criterion],
                    rationale="The safe terminal response is assessed offline.",
                )
            )

    case = _case(
        expected_status=status,
        judge_criteria_enabled=tuple(JudgeCriterion),
    )
    result = _result()
    result.answer = GroundedScoutAnswer(
        answer_markdown="Safe intended product response.",
        status=status,
    )

    evaluation = evaluate_with_judges(case, result, SafeTerminalJudge())

    assert set(evaluation.eligible_criteria) == expected
    assert {request.criterion for request in calls} == expected
    assert {
        item.criterion for item in evaluation.skipped_criteria
    } == set(JudgeCriterion) - expected
    assert JudgeCriterion.FAITHFULNESS not in expected
    assert JudgeCriterion.CONTEXT_RELEVANCE not in expected


def test_judges_receive_presented_answer_while_grounding_keeps_internal_ids() -> None:
    captured: list[JudgeRequest] = []

    class CapturingJudge:
        provider = "fake"
        model = "judge-test"

        def judge(self, request: JudgeRequest) -> JudgeProviderResponse:
            captured.append(request)
            return JudgeProviderResponse(
                output=JudgeStructuredOutput(
                    label="relevant",
                    rationale="The presented response directly answers the question.",
                )
            )

    result = _result()
    result.answer = result.answer.model_copy(
        update={
            "answer_markdown": (
                "expected_completion_rate: 0.923 [run-test:evidence-1]."
            )
        }
    )

    deterministic = evaluate_deterministic_case(_case(), result, _trace())
    judged = evaluate_with_judges(_case(), result, CapturingJudge())
    presented = captured[0].answer
    artifact = _evaluated_case().model_copy(
        update={
            "answer_markdown": result.answer.answer_markdown,
            "internal_answer_markdown": result.answer.answer_markdown,
            "presented_answer_markdown": presented,
            "judge": judged,
        }
    )

    assert deterministic.citation_integrity_pass is True
    assert "run-test:evidence-1" in artifact.internal_answer_markdown
    assert artifact.answer_markdown == artifact.internal_answer_markdown
    assert artifact.presented_answer_markdown == "Expected completion rate: 92.3% [1]."
    assert "run-test:evidence" not in artifact.presented_answer_markdown


def test_judge_evidence_is_valid_bounded_json_with_adaptive_dossier_budget() -> None:
    dossier_metrics = [
        {
            "metric_name": f"metric_{index:03d}",
            "label": f"Metric label {index:03d} " + ("x" * 70),
            "raw_value": index / 10,
            "percentile": index % 100,
            "unit": "complete_unit",
        }
        for index in range(80)
    ]
    dossier_metrics.extend(
        [
            {
                "metric_name": "late_section_metric",
                "label": "Later dossier value",
                "raw_value": 987.65,
                "unit": "complete_late_unit",
            }
        ]
    )
    records = (
        _evidence_record(1, ToolName.SEARCH_PLAYERS, {"player_name": "Alice"}),
        _evidence_record(
            2,
            ToolName.GET_PLAYER_DOSSIER,
                {
                    "identity": {"player_name": "Alice", "matches_observed": 12},
                    "metrics": dossier_metrics,
                    "closing": {"late_confirmation": "preserved-at-the-end"},
                    "api_key": "must-not-appear",
                },
        ),
        _evidence_record(
            3,
            ToolName.GET_METHODOLOGY,
            {"summary": "Observed data, not future performance."},
            category=EvidenceCategory.METHODOLOGY,
        ),
    )

    packed = build_judge_evidence(records)
    parsed = [json.loads(item.content) for item in packed]
    dossier_content = packed[1].content

    assert len(packed) == 3
    assert all(isinstance(item, dict) for item in parsed)
    assert sum(len(item.content) for item in packed) <= JUDGE_EVIDENCE_TOTAL_CHAR_BUDGET
    assert len(dossier_content) > 2_500
    assert "metric_000" in dossier_content
    assert "metrics[40]" in dossier_content
    assert "late_section_metric" in dossier_content
    assert "must-not-appear" not in dossier_content
    assert packed[1].evidence_id == "run-test:evidence-2"
    assert packed[1].category == "analytics"
    assert "get_player_dossier" in packed[1].source_label


def test_answer_aware_projection_preserves_original_deep_cited_dossier_facts() -> None:
    dossier = {
        "intelligence": {
            "archetype": {
                "name": "Safe Circulator",
                "centroid_distance": 1.2970623186724706,
                "second_centroid_distance": 2.373263209699282,
                "separation_margin": 0.4534688300178798,
            },
            "style_metrics": [
                {
                    "metric_name": f"unrelated_style_{index}",
                    "raw_value": index / 100,
                    "percentile": index % 100,
                    "note": "unrelated-" + ("x" * 120),
                }
                for index in range(120)
            ]
            + [
                {
                    "metric_name": "positive_forward_distance_per_100_passes",
                    "label": "Positive forward distance / 100 passes",
                    "raw_value": 654.286147317369,
                    "percentile": 75.0,
                    "unit": "distance_per_100",
                },
                {
                    "metric_name": "pressure_pass_rate",
                    "label": "Under-pressure pass rate",
                    "raw_value": 0.16004849954531677,
                    "percentile": 58.333333333333336,
                    "unit": "rate",
                },
            ],
            "performance_metrics": [
                {
                    "metric_name": f"unrelated_performance_{index}",
                    "raw_value": index / 10,
                    "percentile": index % 100,
                    "note": "irrelevant-" + ("y" * 120),
                }
                for index in range(80)
            ]
            + [
                {
                    "metric_name": "pressure_above_expected_pp",
                    "label": "Under-pressure completion above expected",
                    "raw_value": 2.0775997277462155,
                    "percentile": 77.77777777777777,
                    "unit": "percentage_points",
                }
            ],
        },
        "api_key": "must-stay-redacted",
    }
    records = (
        _evidence_record(1, ToolName.SEARCH_PLAYERS, {"player_name": "Alice"}),
        _evidence_record(2, ToolName.GET_PLAYER_DOSSIER, dossier),
    )
    answer = (
        'Assigned archetype "Safe Circulator" with centroid distance 1.297, '
        "second-closest centroid distance 2.373, and separation margin 0.453. "
        "Positive forward distance / 100 passes was 654.3 at the 75th percentile; "
        "under-pressure pass rate was 16.00% at the 58th percentile; under-pressure "
        "completion above expected was +2.08 points at the 78th percentile. "
        "This sentence exists only in the answer and must not become evidence. "
        "[run-test:evidence-2]"
    )

    first = build_judge_evidence(records, answer_markdown=answer)
    second = build_judge_evidence(records, answer_markdown=answer)
    dossier_projection = json.loads(first[1].content)
    facts = dossier_projection["facts"]

    assert first == second
    assert [item.evidence_id for item in first] == [
        "run-test:evidence-1",
        "run-test:evidence-2",
    ]
    assert dossier_projection["projection"] == (
        "answer_aware_complete_path_value_facts"
    )
    assert dossier_projection["truncated"] is True
    assert facts["intelligence.archetype.name"] == "Safe Circulator"
    assert facts["intelligence.archetype.centroid_distance"] == pytest.approx(1.297062)
    assert facts["intelligence.archetype.second_centroid_distance"] == pytest.approx(
        2.373263
    )
    assert facts["intelligence.archetype.separation_margin"] == pytest.approx(0.453469)
    style_prefix = "intelligence.style_metrics"
    assert any(
        path.startswith(style_prefix) and value == pytest.approx(654.286147)
        for path, value in facts.items()
    )
    assert any(
        path.startswith(style_prefix) and value == pytest.approx(75.0)
        for path, value in facts.items()
    )
    assert any(
        path.startswith(style_prefix) and value == pytest.approx(58.333333)
        for path, value in facts.items()
    )
    assert any(
        path.startswith("intelligence.performance_metrics")
        and value == pytest.approx(77.777778)
        for path, value in facts.items()
    )
    assert sum(len(item.content) for item in first) <= JUDGE_EVIDENCE_TOTAL_CHAR_BUDGET
    assert "must-stay-redacted" not in first[1].content
    assert "This sentence exists only in the answer" not in first[1].content
    assert len(first[1].content) < len(json.dumps(dossier, sort_keys=True))


def test_answer_aware_projection_keeps_original_archetype_name_with_three_records() -> None:
    dossier = {
        "intelligence": {
            "archetype": {
                "id": "safe_circulator",
                "name": "Safe Circulator",
                "centroid_distance": 1.2970623186724706,
            },
            "style_metrics": [
                {
                    "metric_name": f"unrelated_metric_{index}",
                    "raw_value": index / 100,
                    "note": "noise-" + ("x" * 180),
                }
                for index in range(180)
            ],
        }
    }
    records = (
        _evidence_record(1, ToolName.SEARCH_PLAYERS, {"player_name": "Alice"}),
        _evidence_record(2, ToolName.GET_PLAYER_DOSSIER, dossier),
        _evidence_record(
            3,
            ToolName.GET_METHODOLOGY,
            {"summary": "Archetypes describe observed style."},
            category=EvidenceCategory.METHODOLOGY,
        ),
    )
    answer = (
        "The supported archetype is Safe Circulator [run-test:evidence-2]. "
        "Archetypes describe observed style [run-test:evidence-3]."
    )

    packed = build_judge_evidence(records, answer_markdown=answer)
    dossier_facts = json.loads(packed[1].content)["facts"]

    assert dossier_facts["intelligence.archetype.name"] == "Safe Circulator"
    assert sum(len(item.content) for item in packed) <= JUDGE_EVIDENCE_TOTAL_CHAR_BUDGET
    assert "The supported archetype is Safe Circulator" not in packed[1].content


def test_answer_aware_projection_falls_back_deterministically_when_no_fact_matches() -> None:
    record = _evidence_record(
        1,
        ToolName.GET_PLAYER_DOSSIER,
        {
            "metrics": [
                {"metric_name": f"metric_{index}", "raw_value": index}
                for index in range(300)
            ]
        },
    )
    answer = "A generic grounded response [run-test:evidence-1]."

    first = build_judge_evidence((record,), answer_markdown=answer)
    second = build_judge_evidence((record,), answer_markdown=answer)

    assert first == second
    assert len(first[0].content) <= JUDGE_EVIDENCE_TOTAL_CHAR_BUDGET
    assert json.loads(first[0].content)["truncated"] is True


def test_answer_aware_projection_keeps_governed_direction_with_cited_feature() -> None:
    directional_features = [
        {
            "feature_name": "progressive_pass_rate",
            "label": "Progressive-pass rate",
            "direction": "lower",
            "position_z": -0.6245585457908679,
        },
        {
            "feature_name": "positive_forward_distance_per_100_passes",
            "label": "Positive forward distance / 100 passes",
            "direction": "lower",
            "position_z": -0.5575492415096516,
        },
    ]
    dossier = {
        "intelligence": {
            "archetype": {"distinguishing_features": directional_features},
            "style_metrics": [
                {
                    "metric_name": f"unrelated_{index}",
                    "note": "noise-" + ("x" * 160),
                    "raw_value": index / 100,
                }
                for index in range(160)
            ],
        }
    }
    answer = (
        "The governed archetype fields identify relatively lower progressive-pass rate "
        "and forward-distance volume [run-test:evidence-1]."
    )

    packed = build_judge_evidence(
        (_evidence_record(1, ToolName.GET_PLAYER_DOSSIER, dossier),),
        answer_markdown=answer,
    )
    facts = json.loads(packed[0].content)["facts"]

    for index, expected in enumerate(directional_features):
        prefix = f"intelligence.archetype.distinguishing_features[{index}]"
        assert facts[f"{prefix}.feature_name"] == expected["feature_name"]
        assert facts[f"{prefix}.label"] == expected["label"]
        assert facts[f"{prefix}.direction"] == "lower"
        assert facts[f"{prefix}.position_z"] == pytest.approx(expected["position_z"])
    assert len(packed[0].content) <= JUDGE_EVIDENCE_TOTAL_CHAR_BUDGET


def test_answer_aware_projection_never_manufactures_missing_direction() -> None:
    dossier = {
        "intelligence": {
            "archetype": {
                "style_dimensions": [
                    {
                        "feature_name": "progressive_pass_rate",
                        "label": "Progressive-pass rate",
                        "position_z": -0.53,
                    }
                ]
            },
            "noise": ["x" * 200 for _ in range(80)],
        }
    }
    packed = build_judge_evidence(
        (_evidence_record(1, ToolName.GET_PLAYER_DOSSIER, dossier),),
        answer_markdown=(
            "The profile has a lower progressive-pass rate [run-test:evidence-1]."
        ),
    )
    facts = json.loads(packed[0].content)["facts"]

    assert not any(path.endswith(".direction") for path in facts)


def test_answer_aware_projection_keeps_referenced_nested_methodology_facts() -> None:
    metadata = {
        "training_corpus": {
            "competitions": [
                "Germany — 1. Bundesliga 2023/24",
                "FIFA World Cup 2022",
                "UEFA Euro 2024",
            ],
            "matches": 233,
        },
        "nested_cross_fitting": {
            "enabled": True,
            "outer_group": "match_id",
            "inner_group": "match_id",
            "description": "Out-of-fold possession predictions use nested match groups.",
        },
        "noise": [
            {"name": f"candidate-{index}", "detail": "x" * 180}
            for index in range(100)
        ],
    }
    answer = (
        "The training corpus includes Bundesliga 2023/24, the 2022 World Cup, and "
        "Euro 2024. Nested cross-fitting groups both stages by match. "
        "[run-test:evidence-1]"
    )

    packed = build_judge_evidence(
        (_evidence_record(1, ToolName.GET_METHODOLOGY, metadata),),
        answer_markdown=answer,
    )
    facts = json.loads(packed[0].content)["facts"]

    assert facts["training_corpus.competitions[0]"] == (
        "Germany — 1. Bundesliga 2023/24"
    )
    assert facts["training_corpus.competitions[1]"] == "FIFA World Cup 2022"
    assert facts["training_corpus.competitions[2]"] == "UEFA Euro 2024"
    assert facts["nested_cross_fitting.enabled"] is True
    assert facts["nested_cross_fitting.outer_group"] == "match_id"
    assert facts["nested_cross_fitting.inner_group"] == "match_id"
    assert sum(len(item.content) for item in packed) <= (
        JUDGE_EVIDENCE_TOTAL_CHAR_BUDGET
    )


def test_answer_aware_projection_keeps_names_for_a_summarized_competition_corpus() -> None:
    metadata = {
        "structured_metadata": {
            "training_corpus": {
                "competitions": [
                    {"name": "Germany — 1. Bundesliga 2023/24", "events": 137765},
                    {"name": "FIFA World Cup 2022", "events": 234637},
                    {"name": "UEFA Euro 2024", "events": 187924},
                    {"name": "Copa América 2024", "events": 100324},
                    {"name": "African Cup of Nations 2023", "events": 162903},
                ]
            },
            "noise": ["x" * 200 for _ in range(100)],
        }
    }
    answer = (
        "The corpus is mixed-competition, spanning multiple international "
        "tournaments and a Bundesliga cohort [run-test:evidence-1]."
    )

    packed = build_judge_evidence(
        (_evidence_record(1, ToolName.GET_METHODOLOGY, metadata),),
        answer_markdown=answer,
    )
    content = packed[0].content
    facts = json.loads(content)["facts"]

    assert {
        facts[
            f"structured_metadata.training_corpus.competitions[{index}].name"
        ]
        for index in range(5)
    } == {
        "Germany — 1. Bundesliga 2023/24",
        "FIFA World Cup 2022",
        "UEFA Euro 2024",
        "Copa América 2024",
        "African Cup of Nations 2023",
    }
    assert "multiple international tournaments" not in content
    assert len(content) <= JUDGE_EVIDENCE_TOTAL_CHAR_BUDGET


def test_answer_aware_projection_fairly_preserves_referenced_methodology_subtrees() -> None:
    metadata = {
        "structured_metadata": {
            "feature_columns": [f"feature_{index}" for index in range(250)],
            "training_corpus": {
                "competitions": [
                    {"name": f"Competition {index}", "events": index * 1000}
                    for index in range(120)
                ]
            },
            "selection": {
                "model": "xgboost_reg_pseudohubererror",
                "objective": "reg:pseudohubererror",
                "reason": "lowest validation RMSE among governed candidates",
                "rounds": {"regressor": 300},
            },
            "hurdle": {
                "evaluated": True,
                "reason": "evaluated because zero-target rate was 84.21%",
                "rounds": {"classifier": 221, "regressor": 36},
            },
            "out_of_fold_metrics": {
                "mae": 0.024860962189265985,
                "positive_target_mae": 0.08072818791808327,
                "rmse": 0.051629397998852876,
                "spearman": 0.23223664881305472,
                "prediction_min": 0.0,
                "prediction_max": 0.34508734941482544,
            },
            "limitations": [
                "observational/model-derived, not causal",
                "off-ball movement is weakly represented",
                "defensive value is outside V2.2",
                "FootyScout xG has documented upper-tail compression",
                "sparse rare states are less certain",
            ],
        }
    }
    answer = (
        "The selected xgboost_reg_pseudohubererror model uses the "
        "reg:pseudohubererror objective and 300 rounds because it had the lowest "
        "validation RMSE among governed candidates. The hurdle candidate was evaluated "
        "because the zero-target rate was 84.21%. Out-of-fold MAE was 0.02486, "
        "positive-target MAE 0.08073, RMSE 0.05163, and Spearman 0.232; predictions "
        "ranged from 0.0 to 0.3451. The model is observational/model-derived, not causal; "
        "off-ball movement is weakly represented, defensive value is outside V2.2, "
        "FootyScout xG has documented upper-tail compression, and sparse rare states are "
        "less certain. This answer-only sentence must never become evidence. "
        "[run-test:evidence-1]"
    )

    packed = build_judge_evidence(
        (_evidence_record(1, ToolName.GET_METHODOLOGY, metadata),),
        answer_markdown=answer,
    )
    content = packed[0].content
    projection = json.loads(content)
    facts = projection["facts"]

    expected = {
        "structured_metadata.selection.objective": "reg:pseudohubererror",
        "structured_metadata.selection.rounds.regressor": 300,
        "structured_metadata.selection.reason": (
            "lowest validation RMSE among governed candidates"
        ),
        "structured_metadata.hurdle.reason": (
            "evaluated because zero-target rate was 84.21%"
        ),
        "structured_metadata.out_of_fold_metrics.mae": 0.024860962189265985,
        "structured_metadata.out_of_fold_metrics.positive_target_mae": (
            0.08072818791808327
        ),
        "structured_metadata.out_of_fold_metrics.rmse": 0.051629397998852876,
        "structured_metadata.out_of_fold_metrics.spearman": 0.23223664881305472,
        "structured_metadata.out_of_fold_metrics.prediction_min": 0.0,
        "structured_metadata.out_of_fold_metrics.prediction_max": 0.34508734941482544,
        "structured_metadata.limitations[0]": "observational/model-derived, not causal",
        "structured_metadata.limitations[1]": "off-ball movement is weakly represented",
        "structured_metadata.limitations[2]": "defensive value is outside V2.2",
        "structured_metadata.limitations[3]": (
            "FootyScout xG has documented upper-tail compression"
        ),
        "structured_metadata.limitations[4]": "sparse rare states are less certain",
    }
    for path, value in expected.items():
        assert facts[path] == value
    assert len(content) <= JUDGE_EVIDENCE_TOTAL_CHAR_BUDGET
    assert "answer-only sentence" not in content
    assert not any(path.startswith("answer") for path in facts)


def test_relevance_and_completeness_rubrics_keep_omission_scoring_separate() -> None:
    relevance = judge_system_prompt(JudgeCriterion.RELEVANCE)
    completeness = judge_system_prompt(JudgeCriterion.COMPLETENESS)

    assert "omissions belong to completeness" in relevance
    assert "directly on-topic even if details are omitted" in relevance
    assert "answering only one component is incomplete" in completeness
    assert JUDGE_PROMPT_VERSIONS[JudgeCriterion.RELEVANCE] == "judge-relevance-v2"
    assert (
        JUDGE_PROMPT_VERSIONS[JudgeCriterion.COMPLETENESS]
        == "judge-completeness-v2"
    )


def test_context_relevance_input_contains_question_and_evidence_but_not_answer() -> None:
    request = JudgeRequest(
        criterion=JudgeCriterion.CONTEXT_RELEVANCE,
        question="What does Role Fit mean?",
        answer="This response must not be graded.",
        evidence=(
            {
                "evidence_id": "e1",
                "category": "methodology",
                "source_label": "role_fit",
                "content": "Role Fit measures positional-role style resemblance.",
            },
        ),
    )
    payload = judge_input(request)

    assert "What does Role Fit mean?" in payload
    assert "Role Fit measures positional-role style resemblance." in payload
    assert "This response must not be graded." not in payload
    assert '"answer"' not in payload
    prompt = judge_system_prompt(JudgeCriterion.CONTEXT_RELEVANCE)
    assert "Evaluate only question plus evidence" in prompt
    assert JUDGE_PROMPT_VERSIONS[JudgeCriterion.CONTEXT_RELEVANCE] == (
        "judge-context-relevance-v2"
    )


def test_all_focused_judge_criteria_have_separate_valid_structured_labels() -> None:
    labels = {
        JudgeCriterion.RELEVANCE: "relevant",
        JudgeCriterion.COMPLETENESS: "complete",
        JudgeCriterion.FAITHFULNESS: "faithful",
        JudgeCriterion.CLARITY: "clear",
        JudgeCriterion.CONCISION: "concise",
        JudgeCriterion.USEFULNESS: "useful",
        JudgeCriterion.CONTEXT_RELEVANCE: "relevant",
    }

    class CompleteFakeJudge:
        provider = "fake"
        model = "judge-test"

        def judge(self, request: JudgeRequest) -> JudgeProviderResponse:
            return JudgeProviderResponse(
                output=JudgeStructuredOutput(
                    label=labels[request.criterion],
                    rationale="Bounded fixture rationale.",
                )
            )

    case = _case(judge_criteria_enabled=tuple(JudgeCriterion))
    evaluation = evaluate_with_judges(case, _result(), CompleteFakeJudge())

    assert len(evaluation.judgments) == len(JudgeCriterion)
    assert {item.criterion for item in evaluation.judgments} == set(JudgeCriterion)
    assert evaluation.errors == ()


def test_judge_failure_does_not_change_deterministic_result() -> None:
    class FailingJudge:
        provider = "fake"
        model = "judge-test"

        def judge(self, request: JudgeRequest) -> JudgeProviderResponse:
            del request
            raise RuntimeError("safe fake failure")

    deterministic = evaluate_deterministic_case(_case(), _result(), _trace())
    judge = evaluate_with_judges(_case(), _result(), FailingJudge())

    assert deterministic.deterministic_pass is True
    assert judge.judgments == ()
    assert judge.errors[0].error_type == "RuntimeError"


def test_calibration_dataset_separates_exposed_development_and_new_held_out() -> None:
    examples = load_calibration_examples()
    report = calibration_report(examples, ())
    fresh_held_out = tuple(
        example
        for example in examples
        if example.split is CalibrationSplit.HELD_OUT
        and not example.previously_evaluated
    )
    exposed = tuple(example for example in examples if example.previously_evaluated)

    assert len(examples) == 49
    assert sum(example.human_approved for example in examples) == 49
    assert len(exposed) == 7
    assert all(example.split is CalibrationSplit.DEVELOPMENT for example in exposed)
    assert len(fresh_held_out) == 21
    assert all(example.human_approved for example in fresh_held_out)
    assert report.status is CalibrationStatus.UNREVIEWED
    assert all(metric.evaluated_examples == 0 for metric in report.metrics)

    fa003 = next(example for example in examples if example.example_id == "JC-FA-003")
    assert fa003.expected_label == "unfaithful"

    context_development = tuple(
        example
        for example in examples
        if example.criterion is JudgeCriterion.CONTEXT_RELEVANCE
        and example.split is CalibrationSplit.DEVELOPMENT
    )
    assert all(
        len(example.evidence[0]["content"].split()) >= 8
        for example in context_development
    )


def test_new_held_out_label_distribution_covers_each_three_class_label() -> None:
    examples = load_calibration_examples()
    held_out = tuple(
        example for example in examples if example.split is CalibrationSplit.HELD_OUT
    )
    expected = {
        JudgeCriterion.RELEVANCE: {"relevant", "partially_relevant", "irrelevant"},
        JudgeCriterion.COMPLETENESS: {
            "complete",
            "partially_complete",
            "incomplete",
        },
        JudgeCriterion.FAITHFULNESS: {
            "faithful",
            "partially_faithful",
            "unfaithful",
        },
        JudgeCriterion.USEFULNESS: {
            "useful",
            "partially_useful",
            "not_useful",
        },
        JudgeCriterion.CONTEXT_RELEVANCE: {
            "relevant",
            "partially_relevant",
            "irrelevant",
        },
    }
    for criterion, labels in expected.items():
        assert {
            example.expected_label
            for example in held_out
            if example.criterion is criterion
        } == labels
    for criterion in (JudgeCriterion.CLARITY, JudgeCriterion.CONCISION):
        assert len(
            {
                example.expected_label
                for example in held_out
                if example.criterion is criterion
            }
        ) == 2


def test_calibration_metrics_use_only_human_approved_examples() -> None:
    example = JudgeCalibrationExample(
        example_id="JC-RE-999",
        criterion=JudgeCriterion.RELEVANCE,
        question="Question",
        response="Response",
        expected_label="relevant",
        human_approved=True,
        split=CalibrationSplit.DEVELOPMENT,
    )
    report = calibration_report((example,), ())

    assert report.status is CalibrationStatus.UNREVIEWED
    relevance = next(
        metric
        for metric in report.metrics
        if metric.criterion is JudgeCriterion.RELEVANCE
        and metric.split is CalibrationSplit.DEVELOPMENT
    )
    assert relevance.approved_examples == 1
    assert relevance.evaluated_examples == 0
    assert relevance.metric_computable is False
    assert relevance.example_count == 0
    assert relevance.label_coverage == 0


def test_calibration_reports_accuracy_f1_and_confusion_without_split_leakage() -> None:
    examples = tuple(
        JudgeCalibrationExample(
            example_id=f"JC-RE-{index:03d}",
            criterion=JudgeCriterion.RELEVANCE,
            question="Question",
            response="Response",
            expected_label=expected,
            human_approved=True,
            split=split,
        )
        for index, expected, split in (
            (901, "relevant", CalibrationSplit.DEVELOPMENT),
            (902, "irrelevant", CalibrationSplit.DEVELOPMENT),
            (903, "relevant", CalibrationSplit.HELD_OUT),
        )
    )
    predictions = tuple(
        JudgeCalibrationPrediction(
            example_id=example.example_id,
            criterion=example.criterion,
            expected_label=example.expected_label,
            predicted_label="relevant",
        )
        for example in examples
    )

    report = calibration_report(examples, predictions)
    development = next(
        metric
        for metric in report.metrics
        if metric.criterion is JudgeCriterion.RELEVANCE
        and metric.split is CalibrationSplit.DEVELOPMENT
    )
    held_out = next(
        metric
        for metric in report.metrics
        if metric.criterion is JudgeCriterion.RELEVANCE
        and metric.split is CalibrationSplit.HELD_OUT
    )

    assert development.accuracy == 0.5
    assert development.confusion_matrix["irrelevant"]["relevant"] == 1
    assert development.per_class["relevant"].precision == 0.5
    assert held_out.accuracy == 1.0
    assert development.metric_computable is True
    assert development.example_count == 2
    assert development.label_support["relevant"] == 1
    assert development.label_coverage == pytest.approx(2 / 3)
    assert held_out.example_count == 1
    assert held_out.label_coverage == pytest.approx(1 / 3)
    assert not hasattr(held_out, "sufficient_examples")
    assert report.status is CalibrationStatus.INSUFFICIENT


def _held_out_metric(
    criterion: JudgeCriterion,
    expected_labels: tuple[str, ...],
    predicted_labels: tuple[str, ...],
):
    prefix = {
        JudgeCriterion.RELEVANCE: "RE",
        JudgeCriterion.USEFULNESS: "US",
    }[criterion]
    examples = tuple(
        JudgeCalibrationExample(
            example_id=f"JC-{prefix}-{800 + index:03d}",
            criterion=criterion,
            question=f"Question {index}",
            response=f"Response {index}",
            expected_label=expected,
            human_approved=True,
            split=CalibrationSplit.HELD_OUT,
        )
        for index, expected in enumerate(expected_labels, start=1)
    )
    predictions = tuple(
        JudgeCalibrationPrediction(
            example_id=example.example_id,
            criterion=criterion,
            expected_label=example.expected_label,
            predicted_label=predicted,
        )
        for example, predicted in zip(examples, predicted_labels, strict=True)
    )
    report = calibration_report(examples, predictions)
    return next(
        metric
        for metric in report.metrics
        if metric.criterion is criterion
        and metric.split is CalibrationSplit.HELD_OUT
    )


def _macro_f1_from_supported_confusion(
    confusion: dict[str, dict[str, int]],
) -> float:
    supported_f1: list[float] = []
    for label, predicted_counts in confusion.items():
        true_positive = predicted_counts[label]
        false_positive = sum(
            other_counts[label]
            for other_label, other_counts in confusion.items()
            if other_label != label
        )
        false_negative = sum(
            count
            for predicted_label, count in predicted_counts.items()
            if predicted_label != label
        )
        support = sum(predicted_counts.values())
        if support > 0:
            supported_f1.append(
                2 * true_positive
                / (2 * true_positive + false_positive + false_negative)
            )
    return sum(supported_f1) / len(supported_f1)


def test_supported_relevance_class_with_zero_true_positives_contributes_zero_f1() -> None:
    metric = _held_out_metric(
        JudgeCriterion.RELEVANCE,
        ("relevant", "partially_relevant", "irrelevant"),
        ("partially_relevant", "partially_relevant", "irrelevant"),
    )

    assert metric.accuracy == pytest.approx(2 / 3)
    assert metric.confusion_matrix["relevant"]["partially_relevant"] == 1
    assert metric.per_class["relevant"].recall == 0
    assert metric.per_class["relevant"].f1 == 0
    assert metric.macro_f1 == pytest.approx(
        _macro_f1_from_supported_confusion(metric.confusion_matrix)
    )


def test_missed_supported_usefulness_classes_are_not_dropped_from_macro_f1() -> None:
    metric = _held_out_metric(
        JudgeCriterion.USEFULNESS,
        ("partially_useful", "useful", "not_useful"),
        ("not_useful", "partially_useful", "not_useful"),
    )

    assert metric.accuracy == pytest.approx(1 / 3)
    assert metric.confusion_matrix["partially_useful"]["not_useful"] == 1
    assert metric.confusion_matrix["useful"]["partially_useful"] == 1
    assert metric.per_class["useful"].f1 == 0
    assert metric.per_class["partially_useful"].f1 == 0
    assert metric.macro_f1 == pytest.approx(
        _macro_f1_from_supported_confusion(metric.confusion_matrix)
    )


def test_unsupported_and_never_predicted_class_f1_remains_not_applicable() -> None:
    metric = _held_out_metric(
        JudgeCriterion.RELEVANCE,
        ("relevant",),
        ("relevant",),
    )

    assert metric.per_class["partially_relevant"].support == 0
    assert metric.per_class["partially_relevant"].f1 is None
    assert metric.per_class["irrelevant"].support == 0
    assert metric.per_class["irrelevant"].f1 is None
    assert metric.macro_f1 == 1


def test_perfect_three_class_classification_has_macro_f1_one() -> None:
    labels = ("relevant", "partially_relevant", "irrelevant")
    metric = _held_out_metric(JudgeCriterion.RELEVANCE, labels, labels)

    assert metric.accuracy == 1
    assert metric.macro_f1 == 1
    assert all(metric.confusion_matrix[label][label] == 1 for label in labels)


def test_offline_recompute_preserves_history_and_uses_support_aware_f1(
    tmp_path,
    monkeypatch,
) -> None:
    examples = tuple(
        JudgeCalibrationExample(
            example_id=f"JC-RE-{index:03d}",
            criterion=JudgeCriterion.RELEVANCE,
            question=f"Question {index}",
            response=f"Response {index}",
            expected_label=expected,
            human_approved=True,
            split=CalibrationSplit.HELD_OUT,
        )
        for index, expected in enumerate(
            ("relevant", "partially_relevant", "irrelevant"),
            start=951,
        )
    )
    predictions = tuple(
        JudgeCalibrationPrediction(
            example_id=example.example_id,
            criterion=example.criterion,
            expected_label=example.expected_label,
            predicted_label=predicted,
            prompt_version="judge-relevance-v2",
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            latency_ms=5,
        )
        for example, predicted in zip(
            examples,
            ("partially_relevant", "partially_relevant", "irrelevant"),
            strict=True,
        )
    )
    historical = calibration_report(
        examples,
        predictions,
        judge_provider="openai",
        judge_model="frozen-judge",
    ).model_copy(
        update={
            "pricing_version": "frozen-pricing",
            "metrics": (),
        }
    )
    dataset_path = tmp_path / "calibration.jsonl"
    input_path = tmp_path / "historical.json"
    output_path = tmp_path / "corrected.json"
    dataset_path.write_text(
        "\n".join(example.model_dump_json() for example in examples) + "\n",
        encoding="utf-8",
    )
    input_path.write_text(historical.model_dump_json(indent=2), encoding="utf-8")
    original_bytes = input_path.read_bytes()

    def provider_call_forbidden(*_args, **_kwargs):
        raise AssertionError("Offline recomputation must not call a judge provider.")

    monkeypatch.setattr(
        "app.ai.evaluation.calibration.run_calibration",
        provider_call_forbidden,
    )
    corrected = recompute_file(dataset_path, input_path, output_path)
    held_out = next(
        metric
        for metric in corrected.metrics
        if metric.criterion is JudgeCriterion.RELEVANCE
        and metric.split is CalibrationSplit.HELD_OUT
    )

    assert input_path.read_bytes() == original_bytes
    assert output_path.exists()
    assert corrected.judge_provider == "openai"
    assert corrected.judge_model == "frozen-judge"
    assert corrected.pricing_version == "frozen-pricing"
    assert corrected.judge_total_tokens == 36
    assert held_out.per_class["relevant"].f1 == 0
    assert held_out.macro_f1 == pytest.approx(
        _macro_f1_from_supported_confusion(held_out.confusion_matrix)
    )
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        recompute_file(dataset_path, input_path, input_path)


def test_calibration_report_records_safe_reproducibility_metadata_and_usage() -> None:
    example = JudgeCalibrationExample(
        example_id="JC-RE-998",
        criterion=JudgeCriterion.RELEVANCE,
        question="Question with sk-secret-value",
        response="Direct response",
        expected_label="relevant",
        human_approved=True,
        split=CalibrationSplit.DEVELOPMENT,
    )
    pricing = PricingRegistry(
        version="official-2026-09-30",
        entries=(
            ModelPricing(
                provider="fake",
                model="judge-test",
                input_usd_per_million_tokens=1,
                output_usd_per_million_tokens=2,
            ),
        ),
    )
    report = run_calibration((example,), _FakeJudge(), pricing_registry=pricing)

    assert report.calibration_schema_version == "ai-scout-judge-calibration-report-v2"
    assert report.judge_provider == "fake"
    assert report.judge_model == "judge-test"
    assert report.judge_prompt_versions["relevance"] == "judge-relevance-v2"
    assert report.started_at is not None and report.completed_at is not None
    assert report.approved_development_example_count == 1
    assert report.approved_held_out_example_count == 0
    assert report.judge_input_tokens == 10
    assert report.judge_output_tokens == 4
    assert report.judge_total_tokens == 14
    assert report.judge_total_latency_ms is not None
    assert report.pricing_version == "official-2026-09-30"
    assert report.estimated_judge_cost_usd == pytest.approx(0.000018)
    serialized = report.model_dump_json()
    assert "sk-secret-value" not in serialized
    assert "system_prompt" not in serialized
    assert "raw_response" not in serialized


def test_calibration_missing_usage_remains_null_and_context_ignores_response() -> None:
    captured: list[JudgeRequest] = []

    class NoUsageJudge:
        provider = "fake"
        model = "judge-test"

        def judge(self, request: JudgeRequest) -> JudgeProviderResponse:
            captured.append(request)
            return JudgeProviderResponse(
                output=JudgeStructuredOutput(
                    label="relevant",
                    rationale="The evidence directly addresses the question.",
                )
            )

    example = JudgeCalibrationExample(
        example_id="JC-CR-998",
        criterion=JudgeCriterion.CONTEXT_RELEVANCE,
        question="What does Role Fit mean?",
        response="An intentionally incomplete answer.",
        evidence=(
            {
                "evidence_id": "e1",
                "category": "methodology",
                "content": "Role Fit measures positional-role style resemblance.",
            },
        ),
        expected_label="relevant",
        human_approved=True,
        split=CalibrationSplit.DEVELOPMENT,
    )
    report = run_calibration((example,), NoUsageJudge())

    assert captured[0].answer is None
    assert captured[0].question == example.question
    assert captured[0].evidence[0].content == example.evidence[0]["content"]
    assert report.judge_input_tokens is None
    assert report.judge_output_tokens is None
    assert report.judge_total_tokens is None
    assert report.estimated_judge_cost_usd is None


def test_aggregate_excludes_not_applicable_checks_and_missing_judges() -> None:
    evaluated = _evaluated_case()
    aggregate = aggregate_deterministic((evaluated,))

    assert aggregate.methodology_retrieval_accuracy.denominator == 0
    assert aggregate.historical_current_authority_accuracy.denominator == 0
    assert aggregate.entity_resolution_accuracy.denominator == 1

    no_forbidden_case = _case(forbidden_tools=())
    no_forbidden = evaluated.model_copy(
        update={"golden_case": no_forbidden_case}
    )
    assert (
        aggregate_deterministic((no_forbidden,)).forbidden_tool_violation_rate.denominator
        == 0
    )


def test_judge_coverage_uses_only_explicitly_eligible_criteria() -> None:
    labels = {
        JudgeCriterion.RELEVANCE: "relevant",
        JudgeCriterion.COMPLETENESS: "complete",
        JudgeCriterion.FAITHFULNESS: "faithful",
        JudgeCriterion.CLARITY: "clear",
        JudgeCriterion.CONCISION: "concise",
        JudgeCriterion.USEFULNESS: "useful",
        JudgeCriterion.CONTEXT_RELEVANCE: "relevant",
    }
    criteria = tuple(JudgeCriterion)

    def judged_case(evaluated_count: int) -> AIScoutCaseEvaluation:
        golden = _case(judge_criteria_enabled=criteria)
        judgments = tuple(
            CriterionJudgment(
                criterion=criterion,
                label=labels[criterion],
                rationale="Offline coverage fixture.",
                prompt_version=JUDGE_PROMPT_VERSIONS[criterion],
            )
            for criterion in criteria[:evaluated_count]
        )
        return _evaluated_case().model_copy(
            update={
                "golden_case": golden,
                "judge": JudgeCaseEvaluation(
                    case_id=golden.case_id,
                    judge_provider="fake",
                    judge_model="judge-test",
                    eligible_criteria=criteria,
                    judgments=judgments,
                ),
            }
        )

    complete = aggregate_judges((judged_case(7),), enabled=True)
    partial = aggregate_judges((judged_case(5),), enabled=True)
    skipped_case = _evaluated_case().model_copy(
        update={
            "golden_case": _case(judge_criteria_enabled=criteria),
            "judge": JudgeCaseEvaluation(
                case_id="TEST-001",
                skipped_criteria=tuple(
                    JudgeCriterionSkip(
                        criterion=criterion,
                        reason="terminal_status_not_judge_eligible",
                    )
                    for criterion in criteria
                ),
            ),
        }
    )
    not_applicable = aggregate_judges((skipped_case,), enabled=True)

    assert complete.judge_eligible_criteria == 7
    assert complete.judge_evaluated_criteria == 7
    assert complete.judge_skipped_criteria == 0
    assert complete.judge_coverage.rate == 1.0
    assert partial.judge_coverage.numerator == 5
    assert partial.judge_coverage.denominator == 7
    assert partial.judge_coverage.rate == pytest.approx(5 / 7)
    assert not_applicable.judge_eligible_criteria == 0
    assert not_applicable.judge_evaluated_criteria == 0
    assert not_applicable.judge_skipped_criteria == 7
    assert not_applicable.judge_coverage.rate is None
    assert not_applicable.criteria == {}


def test_artifacts_are_resumable_and_reject_incompatible_configuration(tmp_path) -> None:
    case = _evaluated_case()
    case = case.model_copy(
        update={
            "trace": case.trace.model_copy(
                update={
                    "estimated_total_cost_usd": 0.01,
                    "pricing_version": "official-2026-09-30",
                }
            )
        }
    )
    configuration = _configuration()
    store = EvaluationArtifactStore(tmp_path)
    store.initialize(configuration, resume=False)
    store.append_case(case)
    report = aggregate_evaluation(
        (case,),
        run_id="eval-test",
        started_at=datetime(2026, 9, 30, tzinfo=UTC),
        configuration=configuration,
        judge_calibration_status=CalibrationStatus.UNREVIEWED.value,
    )
    store.finalize(report)

    assert report.judge_aggregates.judge_coverage.denominator == 0
    assert report.judge_aggregates.criteria == {}
    assert report.product_estimated_cost_usd == 0.01
    assert report.judge_estimated_cost_usd is None
    assert report.total_evaluation_estimated_cost_usd == 0.01
    resumed = EvaluationArtifactStore(tmp_path)
    resumed.initialize(configuration, resume=True)
    assert tuple(resumed.completed_cases()) == ("TEST-001",)
    assert (tmp_path / "evaluation.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "judge_results.jsonl").exists()

    changed = configuration.model_copy(update={"planner_model": "different"})
    with pytest.raises(ValueError, match="incompatible"):
        resumed.initialize(changed, resume=True)


def test_partial_judge_outcomes_are_quality_warnings_not_failures() -> None:
    case = _evaluated_case().model_copy(
        update={
            "judge": JudgeCaseEvaluation(
                case_id="TEST-001",
                judge_provider="fake",
                judge_model="judge-test",
                judgments=(
                    CriterionJudgment(
                        criterion=JudgeCriterion.RELEVANCE,
                        label="partially_relevant",
                        rationale="The answer addresses only part of the request.",
                        prompt_version="judge-relevance-v2",
                    ),
                ),
            )
        }
    )
    configuration = _configuration().model_copy(update={"judge_enabled": True})
    report = aggregate_evaluation(
        (case,),
        run_id="eval-test",
        started_at=datetime(2026, 9, 30, tzinfo=UTC),
        configuration=configuration,
        judge_calibration_status=CalibrationStatus.HELD_OUT_EVALUATED.value,
    )
    markdown = render_summary_markdown(report)

    assert report.failure_analysis.groups == {}
    assert report.failure_analysis.judge_quality_warnings == {
        "judge_relevance_partially_relevant": ("TEST-001",)
    }
    assert "## Judge Quality Warnings" in markdown
    assert "judge_relevance_partially_relevant: TEST-001" in markdown
