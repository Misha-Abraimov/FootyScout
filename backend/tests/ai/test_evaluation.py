"""Golden case loading and deterministic metric calculations."""

import pytest
from sqlalchemy.orm import Session

from app.ai.diagnostics import FailureStage
from app.ai.evaluation import evaluate_runs, load_golden_cases
from app.ai.evaluation.cases import GoldenPlannerCase
from app.ai.planner import ScoutPlanner
from app.ai.provider import PlannerUsage, ProviderPlanResponse
from app.ai.provider_schemas import LLMPlannerDecision
from app.ai.run import run_ai_scout
from scripts.evaluate_ai_scout_planner import print_verbose_diagnostics


class FakeProvider:
    provider = "anthropic"
    model = "fake-eval-model"

    def __init__(self, response: LLMPlannerDecision | Exception) -> None:
        self.response = response

    def create_plan(self, *, question: str, system_prompt: str) -> ProviderPlanResponse:
        del question, system_prompt
        if isinstance(self.response, Exception):
            raise self.response
        return ProviderPlanResponse(
            decision=self.response,
            usage=PlannerUsage(input_tokens=3, output_tokens=2, total_tokens=5),
        )


def test_checked_in_golden_set_loads_all_85_cases() -> None:
    cases = load_golden_cases()
    assert len(cases) == 85
    assert len({case.id for case in cases}) == 85


def test_all_85_cases_run_through_provider_neutral_evaluator_with_fake_anthropic(
    ai_session: Session,
) -> None:
    cases = load_golden_cases()
    decision = LLMPlannerDecision(
        decision="ready",
        intent="player_search",
        search_query="Alice",
        limit=5,
    )
    planner = ScoutPlanner(FakeProvider(decision))

    runs = [run_ai_scout(ai_session, question=case.question, planner=planner) for case in cases]
    report = evaluate_runs(cases, runs)

    assert report.metrics.case_count == 85
    assert len(report.cases) == 85
    assert report.provider == "anthropic"
    assert report.model == "fake-eval-model"
    provider_runs = sum(run.usage is not None for run in runs)
    guarded_cases = {
        case.id for case, run in zip(cases, runs, strict=True) if run.usage is None
    }
    assert guarded_cases == {"SA-012", "UA-010"}
    assert provider_runs == 83
    assert report.total_tokens == provider_runs * 5


def test_evaluator_scores_programmatic_planner_behavior(ai_session: Session) -> None:
    case = GoldenPlannerCase(
        id="TEST-001",
        category="test",
        question="Find Alice",
        expected_intent="player_search",
        expected_entity_behavior="bounded_filtered_search",
        allowed_tools=["search_players"],
        required_tools=["search_players"],
        forbidden_tools=["get_leaderboard"],
        expected_normalized_arguments={"search_players": {"query": "Alice", "limit": 5}},
        clarification_expected=False,
        insufficient_evidence_expected=False,
        forbidden_claims=[],
        notes="Synthetic provider-free evaluator contract.",
    )
    decision = LLMPlannerDecision(
        decision="ready",
        intent="player_search",
        search_query="Alice",
        limit=5,
    )
    run = run_ai_scout(
        ai_session,
        question=case.question,
        planner=ScoutPlanner(FakeProvider(decision)),
    )
    report = evaluate_runs([case], [run])
    metrics = report.metrics
    assert metrics.intent_accuracy == 1.0
    assert metrics.tool_selection_exact_match == 1.0
    assert metrics.required_tool_recall == 1.0
    assert metrics.forbidden_tool_violation_rate == 0.0
    assert metrics.normalized_argument_accuracy == 1.0
    assert metrics.entity_resolution_accuracy == 1.0
    assert metrics.clarification_accuracy == 1.0
    assert metrics.unsupported_request_accuracy == 1.0
    assert metrics.structured_output_validity == 1.0
    assert metrics.plan_length_violation_rate == 0.0
    assert report.total_tokens == 5
    assert report.provider == "anthropic"
    assert report.model == "fake-eval-model"
    assert report.mean_planning_latency_ms >= 0
    assert report.mean_total_latency_ms >= 0
    assert report.cases[0].failure is None
    diagnostics = report.cases[0].diagnostics
    assert diagnostics.question == "Find Alice"
    assert diagnostics.expected_primary_intent == "player_search"
    assert diagnostics.actual_primary_intent == "player_search"
    assert diagnostics.expected_decision_type == "ready"
    assert diagnostics.actual_decision_type == "ready"
    assert diagnostics.expected_tool_names == ("search_players",)
    assert diagnostics.actual_tool_names == ("search_players",)
    assert diagnostics.raw_scout_plan_tool_names == ("search_players",)
    assert diagnostics.normalized_tool_names == ("search_players",)
    assert diagnostics.raw_scout_plan_arguments["search_players"]["query"] == "Alice"
    assert diagnostics.normalized_tool_arguments["search_players"]["query"] == "Alice"
    assert diagnostics.provider == "anthropic"
    assert diagnostics.model == "fake-eval-model"
    assert diagnostics.token_usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert diagnostics.mismatches == {}
    assert run.failure is None


def test_semantic_diagnostics_do_not_change_existing_scores_and_print_without_failure(
    ai_session: Session,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case = GoldenPlannerCase(
        id="MISMATCH-001",
        category="test",
        question="Who is Alice?",
        expected_intent="player_search",
        expected_entity_behavior="bounded_filtered_search",
        allowed_tools=["search_players"],
        required_tools=["search_players"],
        forbidden_tools=[],
        expected_normalized_arguments={"search_players": {"query": "Alice", "limit": 5}},
        clarification_expected=False,
        insufficient_evidence_expected=False,
        forbidden_claims=[],
        notes="Synthetic semantic mismatch diagnostic.",
    )
    run = run_ai_scout(
        ai_session,
        question=case.question,
        planner=ScoutPlanner(
            FakeProvider(
                LLMPlannerDecision(
                    decision="ready",
                    intent="player_search",
                    search_query="Bob",
                    limit=5,
                )
            )
        ),
    )

    report = evaluate_runs([case], [run])
    result = report.cases[0]

    assert report.metrics.intent_accuracy == float(result.intent_correct) == 1.0
    assert report.metrics.tool_selection_exact_match == float(result.tool_selection_exact) == 1.0
    assert report.metrics.normalized_argument_accuracy == 0.0
    assert result.failure is None
    assert result.diagnostics.expected_extracted_entities == {"search_query": "Alice"}
    assert result.diagnostics.actual_extracted_entities == {"search_query": "Bob"}
    assert result.diagnostics.mismatches["extracted_arguments"] == {
        "expected": {"search_players": {"query": "Alice", "limit": 5}},
        "actual": {
            "search_players": {
                "query": "Bob",
                "position_group": None,
                "min_pass_attempts": 0,
                "sort_by": "player_name",
                "sort_order": "asc",
                "limit": 5,
            }
        },
    }

    print_verbose_diagnostics(report)
    output = capsys.readouterr().out
    assert "MISMATCH-001 semantic mismatches:" in output
    assert "extracted_arguments:" in output
    assert '"query":"Alice"' in output
    assert '"query":"Bob"' in output
    assert "sanitized failure:" not in output


def test_diagnostics_separate_raw_plan_from_failed_normalization(
    ai_session: Session,
) -> None:
    case = GoldenPlannerCase(
        id="RAW-001",
        category="test",
        question="Show me Williams's profile",
        expected_intent="player_profile",
        expected_entity_behavior="clarify_if_multiple_matches",
        allowed_tools=["search_players"],
        required_tools=["search_players"],
        forbidden_tools=["get_player_dossier"],
        expected_normalized_arguments={"search_players": {"query": "Williams", "limit": 10}},
        clarification_expected=True,
        insufficient_evidence_expected=False,
        forbidden_claims=[],
        notes="Raw plan remains inspectable after ambiguous entity resolution.",
    )
    run = run_ai_scout(
        ai_session,
        question=case.question,
        planner=ScoutPlanner(
            FakeProvider(
                LLMPlannerDecision(
                    decision="ready",
                    intent="player_profile",
                    primary_player_name="Williams",
                )
            )
        ),
    )

    diagnostics = evaluate_runs([case], [run]).cases[0].diagnostics

    assert diagnostics.raw_scout_plan_tool_names == (
        "search_players",
        "get_player_dossier",
    )
    assert diagnostics.raw_scout_plan_arguments["search_players"]["query"] == "Williams"
    assert diagnostics.normalized_tool_names == ()
    assert diagnostics.normalized_tool_arguments == {}
    assert diagnostics.entity_resolution_outcome["players"][0]["status"] == "ambiguous"


def test_failed_evaluation_case_records_sanitized_diagnostic(
    ai_session: Session,
) -> None:
    case = GoldenPlannerCase(
        id="FAIL-001",
        category="test",
        question="Find Alice",
        expected_intent="player_search",
        expected_entity_behavior="bounded_filtered_search",
        allowed_tools=["search_players"],
        required_tools=["search_players"],
        forbidden_tools=[],
        expected_normalized_arguments={},
        clarification_expected=False,
        insufficient_evidence_expected=False,
        forbidden_claims=[],
        notes="Synthetic failure diagnostic contract.",
    )
    run = run_ai_scout(
        ai_session,
        question=case.question,
        planner=ScoutPlanner(
            FakeProvider(RuntimeError("OPENAI_API_KEY=never-save-this sk-secret123456"))
        ),
    )
    report = evaluate_runs([case], [run])
    failure = report.cases[0].failure
    assert failure is not None
    assert failure.case_id == "FAIL-001"
    assert failure.failure_stage is FailureStage.PROVIDER
    assert failure.error_type == "RuntimeError"
    assert failure.planner_status == "error"
    assert failure.run_status == "planner_error"
    assert failure.provider_response_received is False
    serialized = report.model_dump_json()
    assert "never-save-this" not in serialized
    assert "sk-secret123456" not in serialized
    assert "[REDACTED]" in serialized
