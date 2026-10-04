"""Offline scoring of structured AI Scout runs against golden expectations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.ai.diagnostics import FailureDiagnostic, FailureStage
from app.ai.evaluation.cases import GoldenPlannerCase
from app.ai.evaluation.metrics import EvaluationMetrics, is_subset, mean
from app.ai.policy import MAX_TOOL_CALLS
from app.ai.run import AIScoutRunResult, AIScoutRunStatus
from app.ai.schemas import ToolName


class RequestUnderstandingMetrics(BaseModel):
    """Provider-facing decision quality, separate from deterministic planning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    structured_decision_validity: float
    intent_accuracy: float
    extracted_entity_accuracy: float
    extracted_argument_accuracy: float
    clarification_decision_accuracy: float
    unsupported_decision_accuracy: float
    requested_analyses_accuracy: float
    plan_builder_validity: float


class CaseDiagnostics(BaseModel):
    """Safe expected-vs-actual evidence for one already-scored golden case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    question: str
    expected_primary_intent: str
    actual_primary_intent: str | None
    expected_decision_type: str
    actual_decision_type: str | None
    expected_requested_analyses: tuple[str, ...]
    actual_requested_analyses: tuple[str, ...]
    expected_extracted_entities: dict[str, Any]
    actual_extracted_entities: dict[str, Any]
    expected_extracted_arguments: dict[str, dict[str, Any]]
    actual_extracted_arguments: dict[str, dict[str, Any]]
    expected_tool_names: tuple[str, ...]
    actual_tool_names: tuple[str, ...]
    raw_scout_plan_tool_names: tuple[str, ...]
    raw_scout_plan_arguments: dict[str, dict[str, Any]]
    normalized_tool_names: tuple[str, ...]
    normalized_tool_arguments: dict[str, dict[str, Any]]
    expected_normalized_tool_arguments: dict[str, dict[str, Any]]
    actual_normalized_tool_arguments: dict[str, dict[str, Any]]
    expected_clarification: bool
    actual_clarification: bool
    expected_unsupported: bool
    actual_unsupported: bool
    entity_resolution_outcome: dict[str, Any]
    run_status: str
    planner_status: str
    provider: str
    model: str
    prompt_version: str
    latency: dict[str, float]
    token_usage: dict[str, int | None] | None
    sanitized_failure: FailureDiagnostic | None
    mismatches: dict[str, Any]


class CaseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    intent_correct: bool
    tool_selection_exact: bool
    required_tool_recall: float
    forbidden_tool_violation: bool
    normalized_arguments_correct: bool
    entity_resolution_correct: bool
    clarification_correct: bool
    unsupported_correct: bool
    structured_output_valid: bool
    plan_length_violation: bool
    request_understanding_valid: bool
    request_intent_correct: bool
    extracted_entities_correct: bool
    extracted_arguments_correct: bool
    clarification_decision_correct: bool
    unsupported_decision_correct: bool
    requested_analyses_correct: bool
    plan_builder_valid: bool
    diagnostics: CaseDiagnostics
    failure: FailureDiagnostic | None = None


class PlannerEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    prompt_version: str
    metrics: EvaluationMetrics
    request_understanding_metrics: RequestUnderstandingMetrics
    cases: tuple[CaseEvaluation, ...]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    mean_planning_latency_ms: float
    mean_total_latency_ms: float


def evaluate_runs(
    cases: Sequence[GoldenPlannerCase],
    runs: Sequence[AIScoutRunResult],
) -> PlannerEvaluationReport:
    if len(cases) != len(runs):
        raise ValueError("Golden cases and run results must have the same length.")
    if not runs:
        raise ValueError("At least one run is required for evaluation.")

    results = tuple(_evaluate_case(case, run) for case, run in zip(cases, runs, strict=True))
    metrics = EvaluationMetrics(
        case_count=len(results),
        intent_accuracy=mean([result.intent_correct for result in results]),
        tool_selection_exact_match=mean([result.tool_selection_exact for result in results]),
        required_tool_recall=mean([result.required_tool_recall for result in results]),
        forbidden_tool_violation_rate=mean([result.forbidden_tool_violation for result in results]),
        normalized_argument_accuracy=mean(
            [result.normalized_arguments_correct for result in results]
        ),
        entity_resolution_accuracy=mean([result.entity_resolution_correct for result in results]),
        clarification_accuracy=mean([result.clarification_correct for result in results]),
        unsupported_request_accuracy=mean([result.unsupported_correct for result in results]),
        structured_output_validity=mean([result.structured_output_valid for result in results]),
        plan_length_violation_rate=mean([result.plan_length_violation for result in results]),
    )
    request_metrics = RequestUnderstandingMetrics(
        structured_decision_validity=mean(
            [result.request_understanding_valid for result in results]
        ),
        intent_accuracy=mean([result.request_intent_correct for result in results]),
        extracted_entity_accuracy=mean(
            [result.extracted_entities_correct for result in results]
        ),
        extracted_argument_accuracy=mean(
            [result.extracted_arguments_correct for result in results]
        ),
        clarification_decision_accuracy=mean(
            [result.clarification_decision_correct for result in results]
        ),
        unsupported_decision_accuracy=mean(
            [result.unsupported_decision_correct for result in results]
        ),
        requested_analyses_accuracy=mean(
            [result.requested_analyses_correct for result in results]
        ),
        plan_builder_validity=mean([result.plan_builder_valid for result in results]),
    )
    token_rows = [run.usage for run in runs if run.usage is not None]
    return PlannerEvaluationReport(
        provider=runs[0].provider,
        model=runs[0].model,
        prompt_version=runs[0].prompt_version,
        metrics=metrics,
        request_understanding_metrics=request_metrics,
        cases=results,
        input_tokens=_sum_optional([row.input_tokens for row in token_rows]),
        output_tokens=_sum_optional([row.output_tokens for row in token_rows]),
        total_tokens=_sum_optional([row.total_tokens for row in token_rows]),
        mean_planning_latency_ms=(sum(run.latency.planning_ms for run in runs) / len(runs)),
        mean_total_latency_ms=sum(run.latency.total_ms for run in runs) / len(runs),
    )


def _evaluate_case(case: GoldenPlannerCase, run: AIScoutRunResult) -> CaseEvaluation:
    planned_calls = run.planned_plan.calls if run.planned_plan is not None else []
    planned_names = [call.name for call in planned_calls]
    required = case.required_tools
    required_recall = (
        1.0 if not required else len(set(planned_names).intersection(required)) / len(set(required))
    )
    forbidden = {name for name in case.forbidden_tools if name in ToolName._value2member_map_}
    forbidden_violation = any(name.value in forbidden for name in planned_names)
    normalized_arguments_correct = _arguments_correct(case, run)
    structured_valid = run.planned_plan is not None
    provider_decision = run.provider_decision
    request_valid = provider_decision is not None
    entity_correct = _entity_resolution_correct(case, run)
    unsupported_actual = _actual_unsupported(run)
    unsupported_correct = unsupported_actual == case.insufficient_evidence_expected
    failure = run.failure
    if failure is None and not structured_valid:
        failure = FailureDiagnostic(
            case_id=case.id,
            failure_stage=FailureStage.PLANNER_VALIDATION,
            error_type="MissingStructuredPlanError",
            sanitized_message=(
                run.error_message or "The run did not contain a structured ScoutPlan."
            ),
            planner_status=run.planner_status.value,
            run_status=run.status.value,
            provider=run.provider,
            provider_response_received=False,
            input_tokens=run.usage.input_tokens if run.usage else None,
            output_tokens=run.usage.output_tokens if run.usage else None,
            total_tokens=run.usage.total_tokens if run.usage else None,
        )
    elif failure is not None:
        failure = failure.model_copy(update={"case_id": case.id})
    result = CaseEvaluation(
        case_id=case.id,
        intent_correct=(run.intent is not None and run.intent.kind is case.expected_intent),
        tool_selection_exact=planned_names == list(required),
        required_tool_recall=required_recall,
        forbidden_tool_violation=forbidden_violation,
        normalized_arguments_correct=normalized_arguments_correct,
        entity_resolution_correct=_entity_resolution_correct(case, run),
        clarification_correct=run.clarification_required == case.clarification_expected,
        unsupported_correct=unsupported_correct,
        structured_output_valid=structured_valid,
        plan_length_violation=len(planned_calls) > MAX_TOOL_CALLS,
        request_understanding_valid=request_valid,
        request_intent_correct=(
            provider_decision is not None and provider_decision.intent is case.expected_intent
        ),
        extracted_entities_correct=request_valid and entity_correct,
        extracted_arguments_correct=request_valid and normalized_arguments_correct,
        clarification_decision_correct=(
            provider_decision is not None
            and (
                provider_decision.decision.value == "clarification_required"
            )
            == case.clarification_expected
        ),
        unsupported_decision_correct=(
            provider_decision is not None
            and (provider_decision.decision.value == "unsupported")
            == case.insufficient_evidence_expected
        ),
        requested_analyses_correct=_requested_analyses_correct(case, run),
        plan_builder_valid=request_valid and structured_valid,
        failure=failure,
        diagnostics=_case_diagnostics(
            case,
            run,
            failure=failure,
            actual_tool_names=planned_names,
            actual_unsupported=unsupported_actual,
        ),
    )
    return result


_TOOL_ANALYSIS = {
    ToolName.GET_PLAYER_DOSSIER: "player_profile",
    ToolName.COMPARE_PLAYERS: "player_comparison",
    ToolName.GET_SIMILAR_PLAYERS: "similar_players",
    ToolName.GET_LEADERBOARD: "leaderboard",
    ToolName.GET_TEAM_INTELLIGENCE: "team_analysis",
    ToolName.GET_ROLE_FIT: "role_fit",
    ToolName.GET_ROLE_RECOMMENDATIONS: "role_recommendations",
    ToolName.GET_METHODOLOGY: "methodology",
}


def _requested_analyses_correct(
    case: GoldenPlannerCase,
    run: AIScoutRunResult,
) -> bool:
    decision = run.provider_decision
    if decision is None:
        return False
    expected = {
        _TOOL_ANALYSIS[tool]
        for tool in case.required_tools
        if tool in _TOOL_ANALYSIS
    } or {case.expected_intent.value}
    actual = {decision.intent.value, *(item.value for item in decision.requested_analyses)}
    return actual == expected


def _case_diagnostics(
    case: GoldenPlannerCase,
    run: AIScoutRunResult,
    *,
    failure: FailureDiagnostic | None,
    actual_tool_names: list[ToolName],
    actual_unsupported: bool,
) -> CaseDiagnostics:
    decision = run.provider_decision
    expected_tools = tuple(tool.value for tool in case.required_tools)
    actual_tools = tuple(tool.value for tool in actual_tool_names)
    expected_analyses = _expected_requested_analyses(case)
    actual_analyses = (
        tuple(item.value for item in decision.requested_analyses) if decision else ()
    )
    expected_normalized_arguments = {
        name: dict(arguments) for name, arguments in case.expected_normalized_arguments.items()
    }
    expected_extracted_arguments = {
        name: _without_entity_arguments(arguments)
        for name, arguments in expected_normalized_arguments.items()
    }
    actual_extracted_arguments = {
        call.name.value: _without_entity_arguments(call.arguments.model_dump(mode="json"))
        for call in (run.planned_plan.calls if run.planned_plan else [])
    }
    actual_normalized = {
        call.name.value: dict(call.arguments)
        for call in (run.normalized_plan.calls if run.normalized_plan else [])
    }
    raw_arguments = {
        call.name.value: call.arguments.model_dump(mode="json")
        for call in (run.planned_plan.calls if run.planned_plan else [])
    }
    normalized_names = tuple(
        call.name.value for call in (run.normalized_plan.calls if run.normalized_plan else [])
    )
    expected_decision = _expected_decision_type(case)
    actual_decision = decision.decision.value if decision else None
    expected_entities = _expected_entities(case)
    actual_entities = _actual_entities(run)
    mismatch_inputs: dict[str, tuple[Any, Any]] = {
        "intent": (case.expected_intent.value, decision.intent.value if decision else None),
        "decision": (expected_decision, actual_decision),
        "requested_analyses": (expected_analyses, actual_analyses),
        "extracted_entities": (expected_entities, actual_entities),
        "extracted_arguments": (expected_extracted_arguments, actual_extracted_arguments),
        "tool_names": (expected_tools, actual_tools),
        "normalized_tool_arguments": (expected_normalized_arguments, actual_normalized),
        "clarification": (case.clarification_expected, run.clarification_required),
        "unsupported": (case.insufficient_evidence_expected, actual_unsupported),
    }
    mismatches = {
        name: {"expected": expected, "actual": actual}
        for name, (expected, actual) in mismatch_inputs.items()
        if not _diagnostic_values_match(name, expected, actual)
    }
    return CaseDiagnostics(
        case_id=case.id,
        question=case.question,
        expected_primary_intent=case.expected_intent.value,
        actual_primary_intent=decision.intent.value if decision else None,
        expected_decision_type=expected_decision,
        actual_decision_type=actual_decision,
        expected_requested_analyses=expected_analyses,
        actual_requested_analyses=actual_analyses,
        expected_extracted_entities=expected_entities,
        actual_extracted_entities=actual_entities,
        expected_extracted_arguments=expected_extracted_arguments,
        actual_extracted_arguments=actual_extracted_arguments,
        expected_tool_names=expected_tools,
        actual_tool_names=actual_tools,
        raw_scout_plan_tool_names=actual_tools,
        raw_scout_plan_arguments=raw_arguments,
        normalized_tool_names=normalized_names,
        normalized_tool_arguments=actual_normalized,
        expected_normalized_tool_arguments=expected_normalized_arguments,
        actual_normalized_tool_arguments=actual_normalized,
        expected_clarification=case.clarification_expected,
        actual_clarification=run.clarification_required,
        expected_unsupported=case.insufficient_evidence_expected,
        actual_unsupported=actual_unsupported,
        entity_resolution_outcome={
            "players": [item.model_dump(mode="json") for item in run.player_resolutions],
            "teams": [item.model_dump(mode="json") for item in run.team_resolutions],
        },
        run_status=run.status.value,
        planner_status=run.planner_status.value,
        provider=run.provider,
        model=run.model,
        prompt_version=run.prompt_version,
        latency=run.latency.model_dump(mode="json"),
        token_usage=run.usage.model_dump(mode="json") if run.usage else None,
        sanitized_failure=failure,
        mismatches=mismatches,
    )


def _expected_requested_analyses(case: GoldenPlannerCase) -> tuple[str, ...]:
    analyses: list[str] = []
    for tool in case.required_tools:
        analysis = _TOOL_ANALYSIS.get(tool)
        if analysis is not None and analysis != case.expected_intent.value:
            analyses.append(analysis)
    return tuple(dict.fromkeys(analyses))


def _expected_decision_type(case: GoldenPlannerCase) -> str:
    if case.clarification_expected:
        return "clarification_required"
    if case.insufficient_evidence_expected and not case.required_tools:
        return "unsupported"
    return "ready"


def _expected_entities(case: GoldenPlannerCase) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    for tool_name, arguments in case.expected_normalized_arguments.items():
        if "player_ids" in arguments:
            player_ids = list(arguments["player_ids"])
            if player_ids:
                entities["primary_player_id"] = player_ids[0]
            if len(player_ids) > 1:
                entities["secondary_player_id"] = player_ids[1]
        if "player_id" in arguments:
            entities["primary_player_id"] = arguments["player_id"]
        if "target_team_id" in arguments:
            entities["team_id"] = arguments["target_team_id"]
        elif "team_id" in arguments:
            entities["team_id"] = arguments["team_id"]
        if "team" in arguments:
            entities["team_name"] = arguments["team"]
        if "query" in arguments:
            key = (
                "search_query"
                if case.expected_intent.value == "player_search"
                else "primary_player_name"
            )
            entities[key] = arguments["query"]
        if tool_name == ToolName.GET_METHODOLOGY.value and "topic" in arguments:
            entities["methodology_topic"] = arguments["topic"]
    return entities


def _without_entity_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    entity_keys = {
        "player",
        "players",
        "player_id",
        "player_ids",
        "team",
        "team_id",
        "target_team",
        "target_team_id",
    }
    return {key: value for key, value in arguments.items() if key not in entity_keys}


def _actual_entities(run: AIScoutRunResult) -> dict[str, Any]:
    decision = run.provider_decision
    if decision is None:
        return {}
    values: dict[str, Any] = {}
    for key, value in (
        ("primary_player_id", decision.primary_player_id),
        ("primary_player_name", decision.primary_player_name.strip()),
        ("secondary_player_id", decision.secondary_player_id),
        ("secondary_player_name", decision.secondary_player_name.strip()),
        ("team_id", decision.team_id),
        ("team_name", decision.team_name.strip()),
        ("search_query", decision.search_query.strip()),
    ):
        if value not in (0, ""):
            values[key] = value
    if decision.methodology_topic.value != "none":
        values["methodology_topic"] = decision.methodology_topic.value
    return values


def _diagnostic_values_match(name: str, expected: Any, actual: Any) -> bool:
    if name in {"extracted_arguments", "normalized_tool_arguments"}:
        return all(
            tool in actual and is_subset(arguments, actual[tool])
            for tool, arguments in expected.items()
        )
    if name == "extracted_entities":
        return all(actual.get(key) == value for key, value in expected.items())
    return expected == actual


def _arguments_correct(case: GoldenPlannerCase, run: AIScoutRunResult) -> bool:
    if not case.expected_normalized_arguments:
        return True
    normalized_by_name = {
        call.name.value: call.arguments
        for call in (run.normalized_plan.calls if run.normalized_plan else [])
    }
    planned_by_name = {
        call.name.value: call.arguments.model_dump(mode="json")
        for call in (run.planned_plan.calls if run.planned_plan else [])
    }
    for name, expected in case.expected_normalized_arguments.items():
        actual = normalized_by_name.get(name, planned_by_name.get(name))
        if actual is None or not is_subset(expected, actual):
            return False
    return True


def _entity_resolution_correct(case: GoldenPlannerCase, run: AIScoutRunResult) -> bool:
    behavior = case.expected_entity_behavior
    if "ambiguous" in behavior or "clarify" in behavior or "duplicate" in behavior:
        return run.clarification_required
    if "not_found" in behavior:
        return run.status is AIScoutRunStatus.NOT_FOUND
    if "resolve" in behavior or "stable_id" in behavior or "use_stable" in behavior:
        return bool(run.player_resolutions or run.team_resolutions or run.normalized_plan)
    return run.status not in {
        AIScoutRunStatus.PLANNER_ERROR,
        AIScoutRunStatus.PLANNER_REFUSED,
        AIScoutRunStatus.INVALID_PLAN,
    }


def _actual_unsupported(run: AIScoutRunResult) -> bool:
    unsupported = run.status in {AIScoutRunStatus.UNSUPPORTED, AIScoutRunStatus.NOT_FOUND}
    if not unsupported:
        unsupported = any(
            record.result is not None
            and (record.result.get("total") == 0 or record.result.get("available") is False)
            for record in run.evidence
        )
    return unsupported


def _sum_optional(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None
