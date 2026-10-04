"""Deterministic Phase 8 scoring against golden expectations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.ai.content_hygiene import (
    detect_current_claim_authority,
    detect_role_fit_language,
)
from app.ai.evaluation.cases import (
    ArgumentAssertion,
    ArgumentOperator,
    GoldenEvalCase,
    WebExpectation,
)
from app.ai.evaluation.phase8_models import ArgumentFailure, DeterministicCaseEvaluation
from app.ai.observability import AIScoutRunTrace
from app.ai.request_guard import is_percentile_fabrication_request
from app.ai.schemas import ToolName
from app.ai.synthesis import GroundedAnswerStatus
from app.ai.workflow import AIScoutWorkflowResult

_AUTHORITY_ERROR = "current_world_claim_requires_web"
_UNSUPPORTED_CODES = {
    "uncalibrated_role_fit_band",
    "normative_role_fit_inference",
    "unsupported_role_fit_driver_language",
    "unsupported_role_fit_limitation_expansion",
    "unsupported_methodology_term",
    "unknown_evidence_id",
    "undeclared_inline_citation",
}
_SAFE_TERMINALS = {
    GroundedAnswerStatus.CLARIFICATION_REQUIRED.value,
    GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value,
    GroundedAnswerStatus.UNSUPPORTED.value,
}
_INTENT_OPTIONAL_TERMINALS = {
    GroundedAnswerStatus.CLARIFICATION_REQUIRED.value,
    GroundedAnswerStatus.UNSUPPORTED.value,
}
_SINGULAR_ENTITY_ID_KEYS = frozenset({"player_id", "team_id", "target_team_id"})
_PLURAL_ENTITY_ID_KEYS = frozenset({"player_ids", "team_ids"})
_ENTITY_DEPENDENT_DOWNSTREAM_TOOLS = {
    "compare_players",
    "get_player_dossier",
    "get_role_fit",
    "get_similar_players",
}
_ENTITY_RESOLUTION_TERMINAL_REASONS = {
    "preparation_clarification_required",
    "preparation_not_found",
}
_FRESHNESS_SENSITIVE_WEB_CATEGORIES = {
    "availability",
    "injury_status",
    "recent_news",
    "transfer_reporting",
}


def evaluate_deterministic_case(
    case: GoldenEvalCase,
    result: AIScoutWorkflowResult,
    trace: AIScoutRunTrace,
) -> DeterministicCaseEvaluation:
    conditional_status_reason = _conditional_current_status_reason(case, trace)
    status_correct = (
        trace.terminal_status == case.expected_status.value
        or conditional_status_reason is not None
    )
    actual_intent = trace.normalized_intent or trace.primary_intent

    planned_tools = tuple(trace.tools_planned)
    normalized_tools = tuple(call.tool_name for call in trace.normalized_tool_calls)
    executed_tools = tuple(trace.tools_executed)
    unreachable_tools = _unreachable_tools(
        trace,
        planned_tools=planned_tools,
        normalized_tools=normalized_tools,
        executed_tools=executed_tools,
    )
    actual_tools = normalized_tools or tuple(
        tool for tool in planned_tools if tool not in set(unreachable_tools)
    )
    actual_tool_set = set(actual_tools)
    required = {tool.value for tool in case.required_tools}
    optional = {tool.value for tool in case.optional_tools}
    forbidden = set(case.forbidden_tools)
    governed_preflight_tools = _governed_preflight_tools(
        case,
        trace,
        required=required,
        planned_tools=set(planned_tools),
        unreachable_tools=set(unreachable_tools),
    )
    selection_tool_set = actual_tool_set | governed_preflight_tools
    required_present = required <= selection_tool_set
    forbidden_absent = not bool(forbidden & actual_tool_set)
    unexpected = tuple(sorted(actual_tool_set - required - optional))
    tool_selection_correct = required_present and forbidden_absent and not unexpected
    required_recall = (
        len(required & selection_tool_set) / len(required) if required else None
    )
    intent_correct = _intent_correct(
        case,
        actual_intent=actual_intent,
        status_correct=status_correct,
        tool_selection_correct=tool_selection_correct,
        actual_tools=actual_tools,
    )

    calls = _actual_calls(trace, unreachable_tools=set(unreachable_tools))
    argument_failures = tuple(
        failure
        for assertion in case.argument_assertions
        if (failure := _check_argument(assertion, calls)) is not None
    )
    argument_correct = not argument_failures if case.argument_assertions else None
    entity_correct = _entity_resolution_correct(case, calls, trace)
    safe_terminal = _matching_safe_terminal(trace, status_correct=status_correct)
    methodology_correct = (
        None if safe_terminal else _methodology_correct(case, trace)
    )
    web_correct = _web_routing_correct(case, trace)
    evidence_correct = (
        None if safe_terminal else _evidence_categories_correct(case, trace, result)
    )
    citation_pass, valid_citations, total_citations = _citation_integrity(
        case,
        result,
        trace,
    )

    validation_codes = {
        finding.error_code for finding in result.diagnostics.validation_findings
    }
    validation_guarded = bool(
        result.diagnostics.validation_block_count
        or result.diagnostics.validation_repair_count
        or result.diagnostics.validation_repair_applied
    )
    grounding_guard_pass = not bool(
        validation_codes & {"unknown_evidence_id", "undeclared_inline_citation"}
    ) or validation_guarded
    authority_applicable = (
        case.web_expectation is WebExpectation.REQUIRED
        or "historical_current" in case.tags
        or _AUTHORITY_ERROR in case.forbidden_answer_behaviors
    )
    authority_pass = (
        (not bool(validation_codes & {_AUTHORITY_ERROR}) or validation_guarded)
        if authority_applicable
        else None
    )
    unsupported_applicable = bool(case.forbidden_answer_behaviors) or bool(
        validation_codes & _UNSUPPORTED_CODES
    )
    forbidden_text_present = any(
        behavior.casefold() in result.answer.answer_markdown.casefold()
        for behavior in case.forbidden_answer_behaviors
        if len(behavior) >= 8
    )
    unsafe_percentile_advice = _contains_unsupported_percentile_fabrication_advice(
        case.question,
        result.answer.answer_markdown,
    )
    role_fit_semantic_codes = {
        finding.error_code
        for finding in detect_role_fit_language(
            result.answer.answer_markdown,
            tuple(result.evidence.records),
        )
    }
    forbidden_role_fit_semantics = bool(
        set(case.forbidden_answer_behaviors) & role_fit_semantic_codes
    )
    unsupported_pass = (
        (
            not forbidden_text_present
            and not unsafe_percentile_advice
            and not forbidden_role_fit_semantics
            and (not bool(validation_codes & _UNSUPPORTED_CODES) or validation_guarded)
        )
        if unsupported_applicable
        else None
    )

    checks: dict[str, bool | None] = {
        "status": status_correct,
        "intent": intent_correct,
        "entity_resolution": entity_correct,
        "tool_selection": tool_selection_correct,
        "tool_arguments": argument_correct,
        "methodology": methodology_correct,
        "web_routing": web_correct,
        "evidence_categories": evidence_correct,
        "citation_integrity": citation_pass,
        "grounding_guard": grounding_guard_pass,
        "historical_current_authority": authority_pass,
        "unsupported_behavior": unsupported_pass,
    }
    failures = tuple(name for name, passed in checks.items() if passed is False)
    deterministic_pass = not failures
    return DeterministicCaseEvaluation(
        case_id=case.case_id,
        planned_tools=planned_tools,
        normalized_tools=normalized_tools,
        executed_tools=executed_tools,
        unreachable_tools=unreachable_tools,
        status_correct=status_correct,
        status_conditionally_accepted=conditional_status_reason is not None,
        status_acceptance_reason=conditional_status_reason,
        intent_correct=intent_correct,
        entity_resolution_correct=entity_correct,
        tool_selection_correct=tool_selection_correct,
        required_tools_present=required_present,
        required_tool_recall=required_recall,
        forbidden_tools_absent=forbidden_absent,
        unexpected_tools=unexpected,
        tool_argument_correct=argument_correct,
        argument_failures=argument_failures,
        methodology_retrieval_correct=methodology_correct,
        web_routing_correct=web_correct,
        evidence_category_correct=evidence_correct,
        citation_integrity_pass=citation_pass,
        citation_valid_references=valid_citations,
        citation_total_references=total_citations,
        grounding_guard_pass=grounding_guard_pass,
        historical_current_authority_pass=authority_pass,
        unsupported_behavior_pass=unsupported_pass,
        deterministic_pass=deterministic_pass,
        failure_reasons=failures,
    )


def _unreachable_tools(
    trace: AIScoutRunTrace,
    *,
    planned_tools: tuple[str, ...],
    normalized_tools: tuple[str, ...],
    executed_tools: tuple[str, ...],
) -> tuple[str, ...]:
    """Identify only entity-dependent calls blocked by deterministic resolution."""
    if trace.terminal_branch_reason not in _ENTITY_RESOLUTION_TERMINAL_REASONS:
        return ()
    reached = set(normalized_tools) | set(executed_tools)
    return tuple(
        dict.fromkeys(
            tool
            for tool in planned_tools
            if tool in _ENTITY_DEPENDENT_DOWNSTREAM_TOOLS and tool not in reached
        )
    )


def _matching_safe_terminal(
    trace: AIScoutRunTrace,
    *,
    status_correct: bool,
) -> bool:
    return (
        status_correct
        and trace.terminal_status in _SAFE_TERMINALS
    )


def _governed_preflight_tools(
    case: GoldenEvalCase,
    trace: AIScoutRunTrace,
    *,
    required: set[str],
    planned_tools: set[str],
    unreachable_tools: set[str],
) -> set[str]:
    """Credit only an explicit unsupported-capability halt for its planned tool."""
    if not (
        case.expected_status is GroundedAnswerStatus.INSUFFICIENT_EVIDENCE
        and trace.terminal_status == GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value
        and trace.preparation_reason_code == "unsupported_team_capability"
    ):
        return set()
    eligible = required & planned_tools & unreachable_tools
    return eligible if eligible == {ToolName.GET_ROLE_FIT.value} else set()


def _intent_correct(
    case: GoldenEvalCase,
    *,
    actual_intent: str | None,
    status_correct: bool,
    tool_selection_correct: bool,
    actual_tools: tuple[str, ...],
) -> bool | None:
    """Make intent N/A only for correct, tool-free terminal contracts.

    Answered analytics and terminal cases that require a governed tool remain
    subject to exact intent scoring.
    """
    if (
        case.expected_status.value in _INTENT_OPTIONAL_TERMINALS
        and status_correct
        and tool_selection_correct
        and not case.required_tools
        and not actual_tools
    ):
        return None
    return actual_intent == case.expected_primary_intent.value


def _conditional_current_status_reason(
    case: GoldenEvalCase,
    trace: AIScoutRunTrace,
) -> str | None:
    """Accept only a proven no-fresh-evidence terminal for live current-status cases."""
    if not (
        case.expected_status is GroundedAnswerStatus.ANSWERED
        and case.web_expectation is WebExpectation.REQUIRED
        and trace.terminal_status
        == GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value
        and trace.web_search_category in _FRESHNESS_SENSITIVE_WEB_CATEGORIES
        and trace.web_search_required
        and trace.web_search_executed
        and trace.web_failure_stage == "no_relevant_results"
        and trace.evidence_max_age_hours is not None
        and trace.web_result_count > 0
        and trace.web_selected_result_count == 0
        and trace.web_evidence_count == 0
        and not trace.provider_failure
        and trace.error_count == 0
        and trace.validation_block_count == 0
    ):
        return None
    return "freshness_filter_removed_all_live_web_results"


def _actual_calls(
    trace: AIScoutRunTrace,
    *,
    unreachable_tools: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = unreachable_tools or set()
    source = trace.normalized_tool_calls or trace.planned_tool_calls
    for call in source:
        if call.tool_name in excluded:
            continue
        calls[call.tool_name].append(call.safe_argument_summary)
    for call in trace.planned_tool_calls:
        if call.tool_name not in calls and call.tool_name not in excluded:
            calls[call.tool_name].append(call.safe_argument_summary)
    return dict(calls)


def _check_argument(
    assertion: ArgumentAssertion,
    calls: dict[str, list[dict[str, Any]]],
) -> ArgumentFailure | None:
    candidates = calls.get(assertion.tool_name.value, [])
    actual_values = [_nested_value(call, assertion.argument) for call in candidates]
    if any(_argument_matches(assertion, actual) for actual in actual_values):
        return None
    return ArgumentFailure(
        tool_name=assertion.tool_name.value,
        argument=assertion.argument,
        operator=assertion.operator.value,
        expected=(
            {"minimum": assertion.minimum, "maximum": assertion.maximum}
            if assertion.operator is ArgumentOperator.NUMERIC_RANGE
            else assertion.expected
        ),
        actual=actual_values[0] if len(actual_values) == 1 else actual_values,
    )


def _argument_matches(assertion: ArgumentAssertion, actual: Any) -> bool:
    if assertion.operator is ArgumentOperator.EXACT:
        return actual == assertion.expected
    if assertion.operator is ArgumentOperator.SET_EQUAL:
        return isinstance(actual, (list, tuple, set)) and set(actual) == set(assertion.expected)
    if assertion.operator is ArgumentOperator.CASE_INSENSITIVE:
        return isinstance(actual, str) and isinstance(assertion.expected, str) and (
            actual.casefold() == assertion.expected.casefold()
        )
    if assertion.operator is ArgumentOperator.NUMERIC_RANGE:
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and (
            assertion.minimum is None or actual >= assertion.minimum
        ) and (assertion.maximum is None or actual <= assertion.maximum)
    if assertion.operator is ArgumentOperator.IS_NULL:
        return actual is None
    if assertion.operator is ArgumentOperator.CONTAINS:
        try:
            return assertion.expected in actual
        except TypeError:
            return False
    return False


def _nested_value(arguments: dict[str, Any], path: str) -> Any:
    value: Any = arguments
    for segment in path.split("."):
        if not isinstance(value, dict) or segment not in value:
            return None
        value = value[segment]
    return value


def _entity_resolution_correct(
    case: GoldenEvalCase,
    calls: dict[str, list[dict[str, Any]]],
    trace: AIScoutRunTrace,
) -> bool | None:
    behavior = case.expected_entity_behavior.casefold()
    if "clarif" in behavior or "ambiguous" in behavior or "duplicate" in behavior:
        return trace.terminal_status == GroundedAnswerStatus.CLARIFICATION_REQUIRED.value
    if "not_found" in behavior:
        return trace.terminal_status == GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value
    flattened = [value for entries in calls.values() for value in entries]
    resolved_players = tuple(
        entity for entity in trace.entity_resolutions if entity.entity_type == "player"
    )
    resolved_teams = tuple(
        entity for entity in trace.entity_resolutions if entity.entity_type == "team"
    )
    if case.expected_player_id is not None:
        return any(
            _contains_value(arguments, "player", case.expected_player_id)
            for arguments in flattened
        ) or any(entity.stable_id == case.expected_player_id for entity in resolved_players)
    if case.expected_team_id is not None:
        return any(
            _contains_value(arguments, "team", case.expected_team_id)
            for arguments in flattened
        ) or any(entity.stable_id == case.expected_team_id for entity in resolved_teams)
    if case.expected_player is not None:
        return any(
            _contains_text(arguments, case.expected_player) for arguments in flattened
        ) or any(
            entity.display_name is not None
            and case.expected_player.casefold() in entity.display_name.casefold()
            for entity in resolved_players
        )
    if case.expected_team is not None:
        return any(
            _contains_text(arguments, case.expected_team) for arguments in flattened
        ) or any(
            entity.display_name is not None
            and case.expected_team.casefold() in entity.display_name.casefold()
            for entity in resolved_teams
        )
    if any(marker in behavior for marker in ("resolve", "stable_id", "use_stable")):
        return any(_contains_identifier(arguments) for arguments in flattened) or any(
            entity.stable_id is not None for entity in trace.entity_resolutions
        )
    return None


def _contains_value(value: Any, key_hint: str, expected: int) -> bool:
    if isinstance(value, dict):
        return any(
            (key_hint in str(key) and item == expected)
            or (
                key_hint in str(key)
                and isinstance(item, list)
                and expected in item
            )
            or _contains_value(item, key_hint, expected)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_value(item, key_hint, expected) for item in value)
    return False


def _contains_text(value: Any, expected: str) -> bool:
    if isinstance(value, str):
        return value.casefold() == expected.casefold()
    if isinstance(value, dict):
        return any(_contains_text(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_text(item, expected) for item in value)
    return False


def _contains_identifier(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (
                str(key) in _SINGULAR_ENTITY_ID_KEYS
                and _is_positive_entity_id(item)
            )
            or (
                str(key) in _PLURAL_ENTITY_ID_KEYS
                and isinstance(item, (list, tuple, set))
                and any(_is_positive_entity_id(identifier) for identifier in item)
            )
            or _contains_identifier(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_identifier(item) for item in value)
    return False


def _is_positive_entity_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _methodology_correct(case: GoldenEvalCase, trace: AIScoutRunTrace) -> bool | None:
    if (
        not case.required_methodology_topics
        and not case.optional_methodology_topics
        and not case.forbidden_methodology_topics
    ):
        return None
    actual = set(trace.methodology_topics_retrieved)
    required = {topic.value for topic in case.required_methodology_topics}
    optional = {topic.value for topic in case.optional_methodology_topics}
    forbidden = {topic.value for topic in case.forbidden_methodology_topics}
    if not required <= actual or forbidden & actual:
        return False
    if case.methodology_topics_exact:
        return actual == required
    return actual <= required | optional


def _web_routing_correct(case: GoldenEvalCase, trace: AIScoutRunTrace) -> bool | None:
    if case.web_expectation is WebExpectation.OPTIONAL:
        return None
    expected_execution = case.web_expectation is WebExpectation.REQUIRED
    if trace.web_search_executed != expected_execution:
        return False
    return not case.expected_web_category or trace.web_search_category == case.expected_web_category


def _evidence_categories_correct(
    case: GoldenEvalCase,
    trace: AIScoutRunTrace,
    result: AIScoutWorkflowResult,
) -> bool | None:
    if not case.required_evidence_categories and not case.forbidden_evidence_categories:
        return None
    counts = {
        "analytics": trace.analytics_evidence_count,
        "methodology": trace.methodology_evidence_count,
        "web": trace.web_evidence_count,
    }
    conditionally_absent_web = _conditionally_accept_absent_mixed_web_evidence(
        case,
        trace,
        result,
    )
    return all(
        counts.get(category, 0) > 0
        or (category == "web" and conditionally_absent_web)
        for category in case.required_evidence_categories
    ) and all(
        counts.get(category, 0) == 0 for category in case.forbidden_evidence_categories
    )


def _conditionally_accept_absent_mixed_web_evidence(
    case: GoldenEvalCase,
    trace: AIScoutRunTrace,
    result: AIScoutWorkflowResult,
) -> bool:
    """Credit a mixed answer only when fresh candidates were deterministically rejected."""
    required = set(case.required_evidence_categories)
    if not (
        case.expected_status is GroundedAnswerStatus.ANSWERED
        and case.web_expectation is WebExpectation.REQUIRED
        and {"analytics", "methodology", "web"} <= required
        and trace.terminal_status == GroundedAnswerStatus.ANSWERED.value
        and trace.web_search_required
        and trace.web_search_executed
        and (
            not case.expected_web_category
            or trace.web_search_category == case.expected_web_category
        )
        and trace.web_failure_stage == "no_relevant_results"
        and trace.web_result_count > 0
        and trace.web_selected_result_count == 0
        and trace.web_evidence_count == 0
        and trace.analytics_evidence_count > 0
        and trace.methodology_evidence_count > 0
        and trace.degraded_due_to_web_failure
        and trace.web_failure_type is None
        and not trace.provider_failure
        and trace.error_count == 0
        and trace.validation_block_count == 0
    ):
        return False
    return not detect_current_claim_authority(
        result.answer.answer_markdown,
        tuple(result.evidence.records),
    )


def _contains_unsupported_percentile_fabrication_advice(
    question: str,
    answer_markdown: str,
) -> bool:
    """Detect affirmative algorithmic advice for a governed no-imputation request."""
    if not is_percentile_fabrication_request(question):
        return False
    normalized = " ".join(answer_markdown.casefold().split())
    rejection_markers = (
        "do not impute",
        "should not be imputed",
        "not be imputed",
        "rather than imputed",
        "rather than be imputed",
        "remain missing",
        "preserved rather than scored",
    )
    unsupported_implementations = (
        "supervised model",
        "k-nearest",
        "knn",
        "position-median percentile",
        "position median percentile",
        "imputation flag",
        "uncertainty band",
    )
    for term in unsupported_implementations:
        start = 0
        while (index := normalized.find(term, start)) >= 0:
            prefix = normalized[max(0, index - 80) : index]
            if not any(
                negation in prefix
                for negation in (
                    "do not ",
                    "don't ",
                    "must not ",
                    "should not ",
                    "never ",
                    "avoid ",
                    "without ",
                )
            ):
                return True
            start = index + len(term)
    affirmative_imputation = (
        "impute missing percentile",
        "impute the missing percentile",
        "predict each missing metric's percentile",
        "predict each missing metric’s percentile",
        "fill in missing percentile",
    )
    return any(term in normalized for term in affirmative_imputation) and not any(
        marker in normalized for marker in rejection_markers
    )


def _citation_integrity(
    case: GoldenEvalCase,
    result: AIScoutWorkflowResult,
    trace: AIScoutRunTrace,
) -> tuple[bool | None, int, int]:
    if (
        result.answer.status is GroundedAnswerStatus.ERROR
        and case.expected_status is not GroundedAnswerStatus.ERROR
    ):
        return None, 0, 0
    cited = tuple(trace.evidence_ids_cited)
    if not cited and trace.evidence_supplied_count == 0:
        return None, 0, 0
    supplied = set(trace.evidence_ids_supplied)
    valid = sum(evidence_id in supplied for evidence_id in cited)
    coverage = not (
        result.answer.status is GroundedAnswerStatus.ANSWERED
        and trace.evidence_supplied_count > 0
        and not cited
    )
    return valid == len(cited) and coverage, valid, len(cited)
