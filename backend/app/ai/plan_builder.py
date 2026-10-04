"""Deterministic mapping from request understanding to strict internal ScoutPlan."""

from __future__ import annotations

import json
import re

from app.ai.methodology_guard import is_archetype_quality_hierarchy_request
from app.ai.plan_normalizer import ALLOWED_TOOLS_BY_INTENT
from app.ai.policy import (
    LEADERBOARD_MAX_RESULTS,
    MAX_TOOL_CALLS,
    PLAYER_SEARCH_MAX_RESULTS,
    RECOMMENDATIONS_MAX_RESULTS,
    SIMILAR_PLAYERS_MAX_RESULTS,
)
from app.ai.provider_schemas import (
    LLMLeaderboardMetric,
    LLMMethodologyTopic,
    LLMPlannerDecision,
    LLMPlayerSortField,
    LLMPositionGroup,
)
from app.ai.schemas import (
    MISSING_PLAYER_REFERENCE,
    SUBJECT_CURRENT_TEAM_REFERENCE,
    EntityRef,
    IntentKind,
    LeaderboardInput,
    LeaderboardIntent,
    MethodologyInput,
    MethodologyIntent,
    MethodologyTopic,
    PlannedComparePlayersCall,
    PlannedComparePlayersInput,
    PlannedLeaderboardCall,
    PlannedMethodologyCall,
    PlannedPlayerDossierCall,
    PlannedPlayerDossierInput,
    PlannedRoleFitCall,
    PlannedRoleFitInput,
    PlannedRoleRecommendationsCall,
    PlannedRoleRecommendationsInput,
    PlannedSearchPlayersCall,
    PlannedSimilarPlayersCall,
    PlannedSimilarPlayersInput,
    PlannedTeamIntelligenceCall,
    PlannedTeamIntelligenceInput,
    PlannedToolCall,
    PlannerDecision,
    PlayerComparisonIntent,
    PlayerDossierSection,
    PlayerProfileIntent,
    PlayerSearchIntent,
    RoleFitIntent,
    RoleRecommendationsIntent,
    ScoutIntent,
    ScoutPlan,
    SearchPlayersInput,
    SimilarPlayersIntent,
    TeamAnalysisIntent,
    TeamRef,
    ToolName,
)
from app.schemas import LeaderboardMetric, PlayerSortField, PositionGroup, SortOrder

_MISSING_PLAYER = EntityRef(player_name=MISSING_PLAYER_REFERENCE)
_MISSING_TEAM = TeamRef(team_name="unspecified team")
_ENTITY_RESOLVABLE_INTENTS = {
    IntentKind.PLAYER_PROFILE,
    IntentKind.PLAYER_COMPARISON,
    IntentKind.SIMILAR_PLAYERS,
    IntentKind.ROLE_FIT,
}
_EXPLICIT_POSITION_PATTERNS = (
    (PositionGroup.GK, re.compile(r"\b(?:goalkeepers?|goalies?|gk)\b", re.IGNORECASE)),
    (PositionGroup.DEF, re.compile(r"\b(?:defenders?|def)\b", re.IGNORECASE)),
    (PositionGroup.MID, re.compile(r"\b(?:midfielders?|midfield|mid)\b", re.IGNORECASE)),
    (
        PositionGroup.FWD,
        re.compile(
            r"\b(?:strikers?|forwards?|fwd)\b(?!\s+(?:pass(?:es|ing)?|distance|progression))",
            re.IGNORECASE,
        ),
    ),
)
_PLAYER_ROLE_REFERENCE = (
    r"(?:goalkeepers?|goalies?|gk|defenders?|centre[- ]backs?|center[- ]backs?|"
    r"full[- ]backs?|midfielders?|wingers?|strikers?|forwards?|players?)"
)
_PLACEHOLDER_PLAYER_REFERENCE = re.compile(
    rf"^(?:(?:a|an|another)\s+(?:(?:external|candidate|potential|prospective|other)\s+)?"
    rf"|(?:external|candidate|potential|prospective|other)\s+){_PLAYER_ROLE_REFERENCE}$",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_METHODOLOGY_SPLIT_TERMS = re.compile(
    r"\b(?:test set|training set|train(?:ing)?[/-]test split|evaluation split|"
    r"out[ -]of[ -]fold|oof|cross[ -]validation(?: predictions?)?|"
    r"grouped[ -](?:cv|cross[ -]validation)|production inference)\b",
    re.IGNORECASE,
)
_PROFILE_COMPOSITION_TERMS = re.compile(
    r"\b(?:what (?:is|are) (?:in|included in)|what (?:does|do) .+ contain|"
    r"composition|composed of|built from|made (?:from|of)|based (?:only )?on|"
    r"derived from|profiles? use|profile provenance)\b",
    re.IGNORECASE,
)
_REFLEXIVE_PLAYER_REFERENCES = frozenset(
    {"himself", "herself", "themself", "themselves", "the same player", "same player"}
)
_EXPLICIT_PLAYER_ID_LABEL = re.compile(
    r"\bplayer(?:\s+|_)id(?P<plural>s?)\b",
    re.IGNORECASE,
)
_COMPARISON_SCOPE_BOUNDARY = re.compile(
    r"\s+(?:on|using|across|by|based\s+on|in\s+terms\s+of|including)\s+",
    re.IGNORECASE,
)
_COMPARISON_NON_ENTITY_TERMS = re.compile(
    r"\b(?:pass(?:es|ing)?|shoot(?:ing|s)?|goal(?:s|scoring)?|metric(?:s)?|"
    r"stat(?:istic)?s?|profile|style|impact|value|performance|percentile(?:s)?|"
    r"explain|describe|show|tell)\b",
    re.IGNORECASE,
)


def build_scout_plan(
    source: LLMPlannerDecision,
    *,
    question: str | None = None,
) -> ScoutPlan:
    """Build only allowlisted calls; the model never emits tool names or arguments."""
    source = _apply_contract_corrections(source, question)
    primary = _primary_player_ref(source)
    secondary = _player_ref(source.secondary_player_id, source.secondary_player_name)
    team = _team_ref(source.team_id, source.team_name)
    position = _position(source.position_group) or _explicit_position(question)
    topic = _topic(source.methodology_topic)
    methodology_topics = _methodology_topics(source, question, topic)
    decision = source.decision
    clarification = source.clarification_message.strip()
    unsupported = source.unsupported_reason.strip()

    operations = _requested_operations(source, question)
    player_team_filter = _player_team_filter(source, operations, question)
    subject_position = _explicit_subject_position(question, primary, position)
    role_fit_team_preflight = _is_role_fit_team_capability_preflight(
        source,
        question,
        primary=primary,
        team=team,
    )
    requirement_error = next(
        (
            error
            for operation in operations
            if (
                error := _requirement_error(
                    operation,
                    primary=primary,
                    secondary=secondary,
                    team=team,
                    position=position,
                    topic=topic,
                    allow_missing_role_fit_player=role_fit_team_preflight,
                )
            )
        ),
        None,
    )
    if decision is PlannerDecision.READY and requirement_error:
        decision = PlannerDecision.CLARIFICATION_REQUIRED
        clarification = requirement_error

    unsupported_reason = next(
        (
            reason
            for operation in operations
            if (reason := _unsupported_scope(operation, position))
        ),
        None,
    )
    if decision is PlannerDecision.READY and unsupported_reason:
        decision = PlannerDecision.UNSUPPORTED
        unsupported = unsupported_reason

    intent = _build_intent(
        source,
        primary=primary,
        secondary=secondary,
        team=team,
        position=position,
        topic=topic,
    )

    # Provider clarification is advisory when deterministic contracts prove that
    # one safe plan is already executable. Product search remains authoritative for
    # named-player ambiguity; role recommendations require only one resolvable team
    # and one supported outfield position, with the governed limit default below.
    if (
        decision is PlannerDecision.CLARIFICATION_REQUIRED
        and requirement_error is None
        and unsupported_reason is None
        and (
            _can_deterministically_prepare_entity_intent(
                source.intent,
                primary=primary,
                secondary=secondary,
                team=team,
            )
            or _is_complete_single_path_role_recommendation(
                source,
                operations=operations,
                team=team,
                position=position,
            )
        )
    ):
        decision = PlannerDecision.READY
        clarification = ""

    if decision is PlannerDecision.UNSUPPORTED:
        calls = []
    elif decision is PlannerDecision.CLARIFICATION_REQUIRED:
        calls = _discovery_calls(
            source,
            primary,
            player_team_filter=player_team_filter,
            subject_position=subject_position,
        )
    else:
        calls = _build_ready_calls(
            source,
            operations=operations,
            primary=primary,
            secondary=secondary,
            team=team,
            position=position,
            topic=topic,
            methodology_topics=methodology_topics,
            player_team_filter=player_team_filter,
            subject_position=subject_position,
        )
        allowed = ALLOWED_TOOLS_BY_INTENT[source.intent]
        if any(call.name not in allowed for call in calls):
            decision = PlannerDecision.CLARIFICATION_REQUIRED
            clarification = "The requested combination of analyses is not supported in one plan."
            calls = []
        elif len(calls) > MAX_TOOL_CALLS:
            decision = PlannerDecision.CLARIFICATION_REQUIRED
            clarification = f"Please narrow the request to at most {MAX_TOOL_CALLS} tool operations."
            calls = []

    return ScoutPlan(
        decision=decision,
        intent=intent,
        calls=calls,
        clarification_message=(
            clarification if decision is PlannerDecision.CLARIFICATION_REQUIRED else None
        ),
        unsupported_reason=(unsupported if decision is PlannerDecision.UNSUPPORTED else None),
        limitations=_provider_limitations(source),
    )


def _provider_limitations(source: LLMPlannerDecision) -> list[str]:
    """Keep provider prose out of governed or misleading user-facing limitations."""
    limitation = source.limitation.strip()
    if not limitation:
        return []
    normalized = limitation.casefold()
    if any(term in normalized for term in ("verify", "verified", "verification")) and any(
        term in normalized
        for term in ("current", "external", "latest", "public", "update", "web")
    ):
        # Retrieval supplies current public reporting; it does not verify truth by itself.
        return []
    operations = {source.intent, *source.requested_analyses}
    is_role_fit_methodology = (
        source.intent is IntentKind.METHODOLOGY
        and source.methodology_topic is LLMMethodologyTopic.ROLE_FIT
    )
    if (
        is_role_fit_methodology
        or IntentKind.ROLE_FIT in operations
        or IntentKind.ROLE_RECOMMENDATIONS in operations
    ):
        return []
    return [limitation]


def _apply_contract_corrections(
    source: LLMPlannerDecision,
    question: str | None,
) -> LLMPlannerDecision:
    """Correct only unambiguous request/tool-contract mismatches."""
    if not question:
        return source

    if _has_invalid_explicit_player_id(source, question):
        return _clarification_without_player_calls(
            source,
            "Player IDs must be positive whole numbers.",
        )

    if (
        source.intent is IntentKind.PLAYER_COMPARISON
        and (_explicit_comparison_reference_count(question) or 0) > 2
    ):
        return _clarification_without_player_calls(
            source,
            "FootyScout compares exactly two players at a time. Which two players "
            "should it compare?",
        )

    mixed_role_fit = _mixed_role_fit_transfer_contract(source, question)
    if mixed_role_fit is not None:
        source = mixed_role_fit

    filtered_search = _filtered_player_search_contract(source, question)
    if filtered_search is not None:
        return filtered_search

    direct_query = _direct_player_search_query(source, question)
    if direct_query is not None:
        source = source.model_copy(
            update={
                "decision": PlannerDecision.READY,
                "intent": IntentKind.PLAYER_SEARCH,
                "requested_analyses": [],
                "search_query": direct_query,
                "clarification_message": "",
                "unsupported_reason": "",
            }
        )

    reflexive_comparison = _reflexive_comparison_contract(source)
    if reflexive_comparison is not None:
        source = reflexive_comparison

    role_fit_comparison = _explicit_role_fit_comparison_contract(source, question)
    if role_fit_comparison is not None:
        source = role_fit_comparison

    archetype_profile = _archetype_profile_contract(source, question)
    if archetype_profile is not None:
        source = archetype_profile

    team_role = _team_role_description_contract(source, question)
    if team_role is not None:
        source = team_role

    similarity = _similarity_contract(source, question)
    if similarity is not None:
        source = similarity

    percentile_methodology = _safe_percentile_methodology_contract(source, question)
    if percentile_methodology is not None:
        source = percentile_methodology

    pressure_passing = _pressure_passing_profile_contract(source, question)
    if pressure_passing is not None:
        source = pressure_passing

    if _is_direct_pass_attempt_ranking(question):
        explicit_limit = _explicit_rank_limit(question)
        return source.model_copy(
            update={
                "decision": PlannerDecision.READY,
                "intent": IntentKind.PLAYER_SEARCH,
                "requested_analyses": [],
                "search_query": "",
                "player_sort_by": LLMPlayerSortField.PASS_ATTEMPTS,
                "sort_order": SortOrder.DESC,
                "limit": explicit_limit or source.limit,
                "clarification_message": "",
                "unsupported_reason": "",
            }
        )

    if source.intent is IntentKind.SIMILAR_PLAYERS:
        explicit_limit = _explicit_similarity_limit(question)
        if explicit_limit is not None:
            source = source.model_copy(update={"limit": explicit_limit})

    recommendation = _candidate_ranking_contract(source, question)
    if recommendation is not None:
        source = recommendation

    team_only_role_fit = _team_only_role_fit_contract(source, question)
    if team_only_role_fit is not None:
        source = team_only_role_fit

    operational_role_fit = _player_specific_role_fit_contract(source, question)
    if operational_role_fit is not None:
        source = operational_role_fit

    if source.intent is IntentKind.METHODOLOGY and _METHODOLOGY_SPLIT_TERMS.search(question):
        source = source.model_copy(
            update={"methodology_topic": LLMMethodologyTopic.XPASS}
        )
    return source


def _clarification_without_player_calls(
    source: LLMPlannerDecision,
    message: str,
) -> LLMPlannerDecision:
    """Return a deterministic input clarification with no discoverable player slot."""
    return source.model_copy(
        update={
            "decision": PlannerDecision.CLARIFICATION_REQUIRED,
            "requested_analyses": [],
            "primary_player_name": "",
            "primary_player_id": 0,
            "secondary_player_name": "",
            "secondary_player_id": 0,
            "search_query": "",
            "clarification_message": message,
            "unsupported_reason": "",
        }
    )


def _has_invalid_explicit_player_id(
    source: LLMPlannerDecision,
    question: str,
) -> bool:
    """Reject malformed/non-positive explicit stable IDs before name resolution."""
    if source.intent not in {*_ENTITY_RESOLVABLE_INTENTS, IntentKind.PLAYER_SEARCH}:
        return False
    labels = list(_EXPLICIT_PLAYER_ID_LABEL.finditer(question))
    if not labels:
        return False
    for label in labels:
        suffix = question[label.end() :]
        if label.group("plural"):
            match = re.match(
                r"\s*(?:[:=#]\s*)?(?:are\s+)?"
                r"(?P<values>[+-]?\d+(?:\s*(?:,|and)\s*[+-]?\d+)*)",
                suffix,
                re.IGNORECASE,
            )
            if match is None:
                return True
            values = re.findall(r"[+-]?\d+", match.group("values"))
            if not values or any(int(value) <= 0 for value in values):
                return True
            continue
        match = re.match(
            r"\s*(?:[:=#]\s*)?(?:is\s+)?(?P<value>[^\s,;?!]+)",
            suffix,
            re.IGNORECASE,
        )
        if match is None:
            return True
        value = match.group("value").rstrip(".").removesuffix("'s").removesuffix("’s")
        if re.fullmatch(r"\+?\d+", value) is None or int(value) <= 0:
            return True
    return False


def _explicit_comparison_reference_count(question: str) -> int | None:
    """Count only clear, compact player lists in an explicit comparison request."""
    match = re.search(r"\bcompare\s+(?P<body>[^?!.]+)", question, re.IGNORECASE)
    if match is None:
        match = re.search(
            r"\bhow\s+do(?:es)?\s+(?P<body>[^?!.]+?)\s+compare\b",
            question,
            re.IGNORECASE,
        )
    if match is None:
        return None
    body = _COMPARISON_SCOPE_BOUNDARY.split(match.group("body"), maxsplit=1)[0]
    body = re.split(
        r"\s+and\s+(?:explain|describe|show|tell)\b",
        body,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    raw_parts = re.split(
        r"\s*,\s*|\s+(?:and|with|versus|vs\.?|&)\s+",
        body,
        flags=re.IGNORECASE,
    )
    parts = [
        re.sub(r"^(?:and|with|versus|vs\.?|&)\s+", "", part, flags=re.IGNORECASE)
        .strip(" ,")
        for part in raw_parts
    ]
    parts = [part for part in parts if part]
    if len(parts) <= 2:
        return len(parts)
    if any(
        len(part.split()) > 5 or _COMPARISON_NON_ENTITY_TERMS.search(part)
        for part in parts[2:]
    ):
        return None
    return len(parts)


def _pressure_passing_profile_contract(
    source: LLMPlannerDecision,
    question: str,
) -> LLMPlannerDecision | None:
    """Keep a pressure-passing profile request on the passing-only tool contract."""
    if source.intent is not IntentKind.PLAYER_PROFILE:
        return None
    if not _is_pressure_passing_question(question):
        return None
    return source.model_copy(
        update={
            "decision": PlannerDecision.READY,
            "requested_sections": [PlayerDossierSection.PASSING],
            "clarification_message": "",
            "unsupported_reason": "",
        }
    )


def _is_pressure_passing_question(question: str) -> bool:
    return bool(
        re.search(r"\b(?:pressure|pressured)\b", question, re.IGNORECASE)
        and re.search(r"\bpass(?:es|ing|er)?\b", question, re.IGNORECASE)
    )


def _mixed_role_fit_transfer_contract(
    source: LLMPlannerDecision,
    question: str,
) -> LLMPlannerDecision | None:
    """Preserve supported Role Fit analytics beside governed transfer context."""
    if not re.search(
        r"\b(?:latest|recent)\s+transfer\b|"
        r"\btransfer\s+(?:news|rumou?rs?|report(?:s|ing)?)\b",
        question,
        re.IGNORECASE,
    ):
        return None
    fit_match = re.search(
        r"\bfit\s+(?P<team>[^,?]+?)(?=\s*,?\s+and\s+|[?.!]|$)",
        question,
        re.IGNORECASE,
    )
    if fit_match is None:
        return None

    player_name = source.primary_player_name.strip()
    player_id = source.primary_player_id
    if player_id <= 0 and not player_name:
        player_id_match = re.search(r"\bplayer\s+(?P<id>\d+)\b", question, re.IGNORECASE)
        if player_id_match is not None:
            player_id = int(player_id_match.group("id"))
        else:
            player_name_match = re.search(
                r"\bhow\s+does\s+(?P<name>.+?)\s+fit\s+",
                question,
                re.IGNORECASE,
            )
            if player_name_match is not None:
                player_name = player_name_match.group("name").strip()
    team_name = source.team_name.strip() or fit_match.group("team").strip()
    if (player_id <= 0 and not player_name) or (
        source.team_id <= 0 and not team_name
    ):
        return None

    return source.model_copy(
        update={
            "decision": PlannerDecision.READY,
            "intent": IntentKind.ROLE_FIT,
            "requested_analyses": [],
            "primary_player_id": player_id,
            "primary_player_name": player_name if player_id <= 0 else "",
            "team_name": team_name if source.team_id <= 0 else source.team_name,
            "methodology_topic": LLMMethodologyTopic.NONE,
            "clarification_message": "",
            "unsupported_reason": "",
        }
    )


def _direct_player_search_query(
    source: LLMPlannerDecision,
    question: str,
) -> str | None:
    """Extract only a bare lookup term from an unambiguous search command."""
    if source.intent is not IntentKind.PLAYER_SEARCH:
        return None
    match = re.fullmatch(
        r"\s*(?:find|search(?:\s+for)?|look\s+up)\s+(?P<query>.+?)\s*[.?!]*\s*",
        question,
        re.IGNORECASE,
    )
    if not match:
        return None
    query = match.group("query").strip(" .?!")
    if not query or re.search(
        r"\b(?:players?|goalkeepers?|defenders?|midfielders?|forwards?|"
        r"similar|role fit|at least|with|who)\b",
        query,
        re.IGNORECASE,
    ):
        return None
    return query


def _filtered_player_search_contract(
    source: LLMPlannerDecision,
    question: str,
) -> LLMPlannerDecision | None:
    """Recover explicit position/sample filters from plural player searches."""
    if not re.search(r"\b(?:find|list|show|identify)\b", question, re.IGNORECASE):
        return None
    position = _explicit_position(question)
    attempts = re.search(
        r"\bat least\s+(?P<count>\d{1,6})\s+(?:pass(?:es)?|pass attempts?)\b",
        question,
        re.IGNORECASE,
    )
    if position is None or attempts is None:
        return None
    return source.model_copy(
        update={
            "decision": PlannerDecision.READY,
            "intent": IntentKind.PLAYER_SEARCH,
            "requested_analyses": [],
            "search_query": "",
            "position_group": LLMPositionGroup(position.value),
            "min_pass_attempts": int(attempts.group("count")),
            "player_sort_by": LLMPlayerSortField.NONE,
            "sort_order": SortOrder.ASC,
            "clarification_message": "",
            "unsupported_reason": "",
        }
    )


def _reflexive_comparison_contract(
    source: LLMPlannerDecision,
) -> LLMPlannerDecision | None:
    if source.intent is not IntentKind.PLAYER_COMPARISON:
        return None
    secondary = source.secondary_player_name.strip().casefold()
    if secondary not in _REFLEXIVE_PLAYER_REFERENCES:
        return None
    return source.model_copy(
        update={
            "decision": PlannerDecision.READY,
            "secondary_player_id": source.primary_player_id,
            "secondary_player_name": source.primary_player_name,
            "clarification_message": "",
            "unsupported_reason": "",
        }
    )


def _explicit_role_fit_comparison_contract(
    source: LLMPlannerDecision,
    question: str,
) -> LLMPlannerDecision | None:
    """Keep explicit Role Fit comparisons in the Role Fit capability boundary."""
    if not re.search(r"\bcompar(?:e|ed|ison)\b", question, re.IGNORECASE):
        return None
    explicit_role_fit = re.search(r"\brole[ -]fit\b", question, re.IGNORECASE)
    governed_team_fit = re.search(
        r"\bhow\s+does\s+.+?\s+fit\s+.+?\s+compar(?:e|ed)\s+(?:with|to)\b",
        question,
        re.IGNORECASE,
    )
    if explicit_role_fit is None and governed_team_fit is None:
        return None
    player_name = source.primary_player_name.strip() or _possessive_role_fit_player(
        question
    )
    if not player_name and governed_team_fit is not None:
        match = re.search(
            r"\bhow\s+does\s+(?P<name>.+?)\s+fit\s+",
            question,
            re.IGNORECASE,
        )
        player_name = match.group("name").strip() if match is not None else ""
    if source.primary_player_id <= 0 and not player_name:
        return None
    return source.model_copy(
        update={
            "intent": IntentKind.ROLE_FIT,
            "requested_analyses": [],
            "primary_player_name": player_name,
            "secondary_player_id": 0,
            "secondary_player_name": "",
            "methodology_topic": LLMMethodologyTopic.NONE,
        }
    )


def _archetype_profile_contract(
    source: LLMPlannerDecision,
    question: str,
) -> LLMPlannerDecision | None:
    if not re.search(r"\barchetype\b", question, re.IGNORECASE):
        return None
    if is_archetype_quality_hierarchy_request(question):
        return None
    player_name = source.primary_player_name.strip() or source.search_query.strip()
    if not player_name:
        match = re.search(
            r"\barchetype\s+(?:is|for)\s+(?P<name>[^?!.]+)",
            question,
            re.IGNORECASE,
        )
        player_name = match.group("name").strip() if match else ""
    if source.primary_player_id <= 0 and not player_name:
        return None
    return source.model_copy(
        update={
            "decision": PlannerDecision.READY,
            "intent": IntentKind.PLAYER_PROFILE,
            "requested_analyses": [],
            "primary_player_name": player_name,
            "team_name": "",
            "team_id": 0,
            "requested_sections": [PlayerDossierSection.INTELLIGENCE],
            "clarification_message": "",
            "unsupported_reason": "",
        }
    )


def _team_role_description_contract(
    source: LLMPlannerDecision,
    question: str,
) -> LLMPlannerDecision | None:
    match = re.search(
        r"\b(?:describe|explain|show)\s+(?P<team>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,2})"
        r"['’]s\s+(?:observed\s+)?(?:goalkeeper|defender|midfielder|forward)\s+role\b",
        question,
        re.IGNORECASE,
    )
    position = _explicit_position(question)
    if match is None or position is None:
        return None
    return source.model_copy(
        update={
            "decision": PlannerDecision.READY,
            "intent": IntentKind.TEAM_ANALYSIS,
            "requested_analyses": [],
            "primary_player_name": "",
            "primary_player_id": 0,
            "team_name": source.team_name.strip() or match.group("team").strip(),
            "position_group": LLMPositionGroup(position.value),
            "clarification_message": "",
            "unsupported_reason": "",
        }
    )


def _similarity_contract(
    source: LLMPlannerDecision,
    question: str,
) -> LLMPlannerDecision | None:
    if not re.search(r"\bsimilar(?:ity|\s+players?|\s+to)\b", question, re.IGNORECASE):
        return None
    if re.search(
        r"\b(?:regardless of|across|without regard to)\s+(?:the\s+)?positions?\b",
        question,
        re.IGNORECASE,
    ):
        return source.model_copy(
            update={
                "decision": PlannerDecision.UNSUPPORTED,
                "intent": IntentKind.SIMILAR_PLAYERS,
                "requested_analyses": [],
                "clarification_message": "",
                "unsupported_reason": (
                    "Playing-style similarity is available only within the same "
                    "position group."
                ),
            }
        )
    player_name = source.primary_player_name.strip() or source.search_query.strip()
    if not player_name:
        for pattern in (
            r"\bsimilar\s+to\s+(?P<name>[^?!.]+)",
            r"\bas\s+good\s+as\s+(?P<name>[^?!.]+)",
        ):
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                player_name = match.group("name").strip()
                break
    if source.primary_player_id <= 0 and not player_name:
        return None
    return source.model_copy(
        update={
            "decision": PlannerDecision.READY,
            "intent": IntentKind.SIMILAR_PLAYERS,
            "requested_analyses": [],
            "primary_player_name": player_name,
            "team_name": "",
            "team_id": 0,
            "clarification_message": "",
            "unsupported_reason": "",
        }
    )


def _safe_percentile_methodology_contract(
    source: LLMPlannerDecision,
    question: str,
) -> LLMPlannerDecision | None:
    if not re.search(r"\b(?:invent|fabricate|impute|fill in)\b", question, re.IGNORECASE):
        return None
    if not re.search(r"\bpercentiles?\b", question, re.IGNORECASE):
        return None
    return source.model_copy(
        update={
            "decision": PlannerDecision.READY,
            "intent": IntentKind.METHODOLOGY,
            "requested_analyses": [],
            "methodology_topic": LLMMethodologyTopic.PERCENTILES,
            "clarification_message": "",
            "unsupported_reason": "",
        }
    )


def _is_direct_pass_attempt_ranking(question: str) -> bool:
    normalized = question.casefold()
    direct_field = bool(
        re.search(r"\bpass attempts?\b|\bpasses attempted\b|\bmost passes\b", normalized)
    )
    ranking = bool(
        re.search(r"\b(?:top|rank|sort|most|highest|players? with)\b", normalized)
    )
    return direct_field and ranking and "completion above" not in normalized


def _explicit_rank_limit(question: str) -> int | None:
    match = re.search(r"\btop\s+(\d{1,4})\b", question, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _explicit_similarity_limit(question: str) -> int | None:
    normalized = question.casefold()
    if re.search(
        r"\b(?:the\s+)?most similar player\b|\b(?:closest|nearest)\s+(?:style\s+)?"
        r"(?:neighbor|neighbour|player)\b",
        normalized,
    ):
        return 1
    number = r"(?P<count>\d{1,4}|" + "|".join(_NUMBER_WORDS) + r")"
    match = re.search(
        rf"\b(?:top\s+)?{number}\s+(?:most\s+)?(?:similar\s+)?players?\b",
        normalized,
    )
    if not match:
        return None
    value = match.group("count")
    return int(value) if value.isdigit() else _NUMBER_WORDS[value]


def _candidate_ranking_contract(
    source: LLMPlannerDecision,
    question: str,
) -> LLMPlannerDecision | None:
    normalized = question.casefold()
    candidate_request = bool(
        re.search(
            r"\b(?:which|what)\s+players\b|\brank\b.*\b(?:players|candidates)\b|"
            r"\bbest candidates?\b|\bwho fits?\b|"
            r"\b(?:find|show|list|identify|recommend)\b[^?.!]{0,100}"
            r"\b(?:goalkeepers?|defenders?|midfielders?|forwards?|players?|candidates?)\b"
            r"[^?.!]{0,100}\b(?:role fit|fit(?:s|ting)?)\b",
            normalized,
        )
    )
    if not candidate_request:
        return None
    position = _position(source.position_group) or _explicit_position(question)
    team_name = source.team_name.strip() or _candidate_team_name(question, position)
    if source.team_id <= 0 and not team_name:
        return None
    if position not in {PositionGroup.DEF, PositionGroup.MID, PositionGroup.FWD}:
        return None
    return source.model_copy(
        update={
            "decision": PlannerDecision.READY,
            "intent": IntentKind.ROLE_RECOMMENDATIONS,
            "requested_analyses": [],
            "primary_player_name": "",
            "primary_player_id": 0,
            "secondary_player_name": "",
            "secondary_player_id": 0,
            "team_name": team_name,
            "position_group": LLMPositionGroup(position.value),
            "clarification_message": "",
            "unsupported_reason": "",
        }
    )


def _candidate_team_name(
    question: str,
    position: PositionGroup | None,
) -> str:
    if position is None:
        return ""
    position_words = {
        PositionGroup.DEF: r"defen(?:ce|se|der|ders|sive)|def",
        PositionGroup.MID: r"midfield(?:er|ers)?|mid",
        PositionGroup.FWD: r"forward(?:s)?|striker(?:s)?|fwd",
        PositionGroup.GK: r"goalkeeper(?:s)?|goalie(?:s)?|gk",
    }[position]
    proper_team = r"(?P<team>[A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,2})"
    patterns = (
        rf"\b[Rr]ank\s+{proper_team}\s+(?:{position_words})\s+(?:players|candidates)\b",
        rf"\b[Ff]it\s+{proper_team}['’]s\s+(?:{position_words})\s+role\b",
        rf"\b[Ff]or\s+{proper_team}['’]s?\s+(?:{position_words})\s+role\b",
        rf"\b(?:role\s+)?[Ff]it\s+for\s+{proper_team}(?:[.?!]|$)",
        rf"\b(?:who\s+)?[Ff]it\s+{proper_team}(?:[.?!]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group("team").strip()
    return ""


def _player_specific_role_fit_contract(
    source: LLMPlannerDecision,
    question: str,
) -> LLMPlannerDecision | None:
    if not re.search(r"\brole fit\b", question, re.IGNORECASE):
        return None
    if not re.search(
        r"\b(?:own events?|include[sd]?|calculated|computed|calculation|"
        r"leave[ -]self[ -]out|current[ -]team)\b",
        question,
        re.IGNORECASE,
    ):
        return None
    player_name = source.primary_player_name.strip() or _possessive_role_fit_player(question)
    if not player_name:
        return None
    current_team = bool(re.search(r"\bcurrent[ -]team\b", question, re.IGNORECASE))
    if source.team_id <= 0 and not source.team_name.strip() and not current_team:
        return None
    return source.model_copy(
        update={
            "decision": PlannerDecision.READY,
            "intent": IntentKind.ROLE_FIT,
            "requested_analyses": [],
            "primary_player_name": player_name,
            "team_name": (
                SUBJECT_CURRENT_TEAM_REFERENCE
                if current_team and source.team_id <= 0
                else source.team_name
            ),
            "methodology_topic": LLMMethodologyTopic.NONE,
            "clarification_message": "",
            "unsupported_reason": "",
        }
    )


def _team_only_role_fit_contract(
    source: LLMPlannerDecision,
    question: str,
) -> LLMPlannerDecision | None:
    """Allow team-support resolution to precede missing-player clarification."""
    if source.intent is not IntentKind.ROLE_FIT:
        return None
    if source.primary_player_id > 0:
        return None
    if source.team_id <= 0 and not source.team_name.strip():
        return None
    if not re.fullmatch(
        r"\s*(?:(?:calculate|show|check|give me)\s+)?role fit\s+for\s+.+?[.?!]?\s*",
        question,
        re.IGNORECASE,
    ):
        return None
    player_name = source.primary_player_name.strip()
    team_name = source.team_name.strip()
    if player_name and player_name.casefold() != team_name.casefold():
        return None
    return source.model_copy(
        update={
            "decision": PlannerDecision.READY,
            "primary_player_name": "",
            "search_query": "",
            "clarification_message": "",
            "unsupported_reason": "",
        }
    )


def _possessive_role_fit_player(question: str) -> str:
    match = re.search(
        r"\b(?P<name>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,3})['’]s\s+"
        r"(?:current[ -]team\s+)?Role Fit\b",
        question,
    )
    if not match:
        return ""
    tokens = match.group("name").split()
    while tokens and tokens[0].casefold() in {"does", "is", "what", "how"}:
        tokens.pop(0)
    return " ".join(tokens)


def _methodology_topics(
    source: LLMPlannerDecision,
    question: str | None,
    fallback: MethodologyInput | None,
) -> list[MethodologyInput]:
    if source.intent is not IntentKind.METHODOLOGY or fallback is None:
        return [fallback] if fallback is not None else []
    topics: list[MethodologyInput] = []
    if question and _METHODOLOGY_SPLIT_TERMS.search(question):
        if (
            re.search(r"\b(?:player\s+)?profiles?\b", question, re.IGNORECASE)
            and _PROFILE_COMPOSITION_TERMS.search(question)
        ):
            topics.append(MethodologyInput(topic=MethodologyTopic.PLAYER_PROFILES))
        topics.append(MethodologyInput(topic=MethodologyTopic.XPASS))
    else:
        topics.append(fallback)
    unique: dict[MethodologyTopic, MethodologyInput] = {}
    for item in topics:
        unique.setdefault(item.topic, item)
    return list(unique.values())


def _primary_player_ref(source: LLMPlannerDecision) -> EntityRef | None:
    primary = _player_ref(source.primary_player_id, source.primary_player_name)
    if primary is not None or source.intent not in _ENTITY_RESOLVABLE_INTENTS:
        return primary
    query = source.search_query.strip()
    return _player_ref(0, query)


def _can_deterministically_prepare_entity_intent(
    intent: IntentKind,
    *,
    primary: EntityRef | None,
    secondary: EntityRef | None,
    team: TeamRef | None,
) -> bool:
    """Let governed resolution decide complete ID/name references, not provider doubt."""
    if intent not in _ENTITY_RESOLVABLE_INTENTS or primary is None:
        return False
    if intent is IntentKind.PLAYER_COMPARISON:
        return secondary is not None
    if intent is IntentKind.ROLE_FIT:
        return team is not None
    return True


def is_placeholder_player_reference(value: str) -> bool:
    """Return whether a phrase denotes a missing player slot rather than a name."""
    normalized = re.sub(r"\s+", " ", value.strip(" .?!")).strip()
    return bool(normalized and _PLACEHOLDER_PLAYER_REFERENCE.fullmatch(normalized))


def _explicit_subject_position(
    question: str | None,
    primary: EntityRef | None,
    position: PositionGroup | None,
) -> PositionGroup | None:
    """Keep candidate/cohort filters off subject lookup unless explicitly attached."""
    if not question or primary is None or primary.player_name is None or position is None:
        return None
    name = re.escape(primary.player_name.strip())
    role_pattern = next(
        pattern.pattern
        for candidate, pattern in _EXPLICIT_POSITION_PATTERNS
        if candidate is position
    )
    if re.search(rf"\b(?:{role_pattern})\s+{name}\b", question, re.IGNORECASE):
        return position
    if re.search(
        rf"\b{name}\b\s*(?:,|\bis\b|\bas\b)\s*(?:a|an)?\s*(?:{role_pattern})\b",
        question,
        re.IGNORECASE,
    ):
        return position
    return None


def _is_complete_single_path_role_recommendation(
    source: LLMPlannerDecision,
    *,
    operations: list[IntentKind],
    team: TeamRef | None,
    position: PositionGroup | None,
) -> bool:
    """Recover a provider clarification only when one recommendation call is sufficient."""
    return bool(
        source.intent is IntentKind.ROLE_RECOMMENDATIONS
        and operations == [IntentKind.ROLE_RECOMMENDATIONS]
        and team is not None
        and position in {PositionGroup.DEF, PositionGroup.MID, PositionGroup.FWD}
    )


def _explicit_position(question: str | None) -> PositionGroup | None:
    """Recover only explicit position nouns using the governed four-group vocabulary."""
    if not question:
        return None
    matches = [
        (match.start(), position)
        for position, pattern in _EXPLICIT_POSITION_PATTERNS
        if (match := pattern.search(question)) is not None
    ]
    return min(matches, default=(0, None), key=lambda item: item[0])[1]


def _requirement_error(
    intent: IntentKind,
    *,
    primary: EntityRef | None,
    secondary: EntityRef | None,
    team: TeamRef | None,
    position: PositionGroup | None,
    topic: MethodologyInput | None,
    allow_missing_role_fit_player: bool = False,
) -> str | None:
    if intent in {
        IntentKind.PLAYER_PROFILE,
        IntentKind.SIMILAR_PLAYERS,
        IntentKind.ROLE_FIT,
    } and primary is None and not (
        intent is IntentKind.ROLE_FIT and allow_missing_role_fit_player
    ):
        return "Which player should FootyScout analyze?"
    if intent is IntentKind.PLAYER_COMPARISON:
        if primary is None or secondary is None:
            return "Which two players should FootyScout compare?"
        if primary == secondary:
            return "Comparison requires two distinct players."
    if intent in {
        IntentKind.TEAM_ANALYSIS,
        IntentKind.ROLE_FIT,
        IntentKind.ROLE_RECOMMENDATIONS,
    } and team is None:
        return "Which team should FootyScout analyze?"
    if intent is IntentKind.ROLE_RECOMMENDATIONS and position is None:
        return "Which outfield position should FootyScout use for role recommendations?"
    if intent is IntentKind.METHODOLOGY and topic is None:
        return "Which FootyScout methodology topic should be explained?"
    return None


def _is_role_fit_team_capability_preflight(
    source: LLMPlannerDecision,
    question: str | None,
    *,
    primary: EntityRef | None,
    team: TeamRef | None,
) -> bool:
    """Resolve governed team coverage before asking for a missing Role Fit subject."""
    return bool(
        question
        and source.intent is IntentKind.ROLE_FIT
        and primary is None
        and team is not None
        and re.search(r"\brole fit\b", question, re.IGNORECASE)
    )


def _unsupported_scope(intent: IntentKind, position: PositionGroup | None) -> str | None:
    if position is not PositionGroup.GK:
        return None
    if intent is IntentKind.TEAM_ANALYSIS:
        return "Goalkeeper team-role intelligence is not available."
    if intent is IntentKind.ROLE_FIT:
        return "Goalkeeper Role Fit is not available."
    if intent is IntentKind.ROLE_RECOMMENDATIONS:
        return "Goalkeeper role recommendations are not available."
    return None


def _build_intent(
    source: LLMPlannerDecision,
    *,
    primary: EntityRef | None,
    secondary: EntityRef | None,
    team: TeamRef | None,
    position: PositionGroup | None,
    topic: MethodologyInput | None,
) -> ScoutIntent:
    if source.intent is IntentKind.PLAYER_SEARCH:
        return PlayerSearchIntent(
            query=source.search_query.strip() or None,
            team=source.team_name.strip() or None,
            position_group=position,
            min_pass_attempts=source.min_pass_attempts,
            sort_by=_sort_field(source.player_sort_by),
            sort_order=source.sort_order,
            limit=_limit(source.limit, 10, PLAYER_SEARCH_MAX_RESULTS),
        )
    if source.intent is IntentKind.PLAYER_PROFILE:
        return PlayerProfileIntent(
            player=primary or _MISSING_PLAYER,
            sections=_sections(source),
        )
    if source.intent is IntentKind.PLAYER_COMPARISON:
        comparison_secondary = secondary
        if comparison_secondary is None or comparison_secondary == primary:
            comparison_secondary = EntityRef(player_name="other player")
        return PlayerComparisonIntent(
            players=(primary or _MISSING_PLAYER, comparison_secondary)
        )
    if source.intent is IntentKind.SIMILAR_PLAYERS:
        return SimilarPlayersIntent(
            player=primary or _MISSING_PLAYER,
            candidate_position_group=position,
            limit=_limit(source.limit, 6, SIMILAR_PLAYERS_MAX_RESULTS),
        )
    if source.intent is IntentKind.LEADERBOARD:
        return LeaderboardIntent(
            metric=_metric(source.leaderboard_metric),
            position_group=position,
            team=source.team_name.strip() or None,
            limit=_limit(source.limit, 10, LEADERBOARD_MAX_RESULTS),
        )
    if source.intent is IntentKind.TEAM_ANALYSIS:
        return TeamAnalysisIntent(team=team or _MISSING_TEAM, position_group=position)
    if source.intent is IntentKind.ROLE_FIT:
        return RoleFitIntent(player=primary or _MISSING_PLAYER, target_team=team or _MISSING_TEAM)
    if source.intent is IntentKind.ROLE_RECOMMENDATIONS:
        return RoleRecommendationsIntent(
            team=team or _MISSING_TEAM,
            position_group=(
                position if position not in {None, PositionGroup.GK} else PositionGroup.MID
            ),
            limit=_limit(source.limit, 10, RECOMMENDATIONS_MAX_RESULTS),
        )
    return MethodologyIntent(
        topic=(topic.topic if topic is not None else _metric_topic_fallback()),
        question=None,
    )


def _build_ready_calls(
    source: LLMPlannerDecision,
    *,
    operations: list[IntentKind],
    primary: EntityRef | None,
    secondary: EntityRef | None,
    team: TeamRef | None,
    position: PositionGroup | None,
    topic: MethodologyInput | None,
    methodology_topics: list[MethodologyInput],
    player_team_filter: str | None,
    subject_position: PositionGroup | None,
) -> list[PlannedToolCall]:
    calls: list[PlannedToolCall] = []
    for operation in operations:
        calls.extend(
            _calls_for_operation(
                operation,
                source,
                primary=primary,
                secondary=secondary,
                team=team,
                position=position,
                topic=topic,
                methodology_topics=methodology_topics,
                player_team_filter=player_team_filter,
                subject_position=subject_position,
            )
        )
    return _deduplicate_calls(calls)


def _calls_for_operation(
    operation: IntentKind,
    source: LLMPlannerDecision,
    *,
    primary: EntityRef | None,
    secondary: EntityRef | None,
    team: TeamRef | None,
    position: PositionGroup | None,
    topic: MethodologyInput | None,
    methodology_topics: list[MethodologyInput],
    player_team_filter: str | None,
    subject_position: PositionGroup | None,
) -> list[PlannedToolCall]:
    if operation is IntentKind.PLAYER_SEARCH:
        return [
            _search_call(
                source,
                team_filter=player_team_filter,
                position=position,
            )
        ]
    if operation is IntentKind.PLAYER_PROFILE and primary is not None:
        calls = _named_player_search(
            source,
            primary,
            team_filter=player_team_filter,
            position=subject_position,
        )
        calls.append(
            PlannedPlayerDossierCall(
                name=ToolName.GET_PLAYER_DOSSIER,
                arguments=PlannedPlayerDossierInput(
                    player=primary,
                    sections=_sections(source),
                ),
            )
        )
        return calls
    if operation is IntentKind.PLAYER_COMPARISON and primary and secondary:
        calls = _named_player_search(
            source,
            primary,
            team_filter=player_team_filter,
            position=subject_position,
        )
        calls.append(
            PlannedComparePlayersCall(
                name=ToolName.COMPARE_PLAYERS,
                arguments=PlannedComparePlayersInput(players=(primary, secondary)),
            )
        )
        return calls
    if operation is IntentKind.SIMILAR_PLAYERS and primary is not None:
        calls = _named_player_search(
            source,
            primary,
            team_filter=player_team_filter,
            position=subject_position,
        )
        calls.append(
            PlannedSimilarPlayersCall(
                name=ToolName.GET_SIMILAR_PLAYERS,
                arguments=PlannedSimilarPlayersInput(
                    player=primary,
                    candidate_position_group=position,
                    limit=_limit(source.limit, 6, SIMILAR_PLAYERS_MAX_RESULTS),
                ),
            )
        )
        return calls
    if operation is IntentKind.LEADERBOARD:
        return [
            PlannedLeaderboardCall(
                name=ToolName.GET_LEADERBOARD,
                arguments=LeaderboardInput(
                    metric=_metric(source.leaderboard_metric),
                    position_group=position,
                    team=source.team_name.strip() or None,
                    limit=_limit(source.limit, 10, LEADERBOARD_MAX_RESULTS),
                ),
            )
        ]
    if operation is IntentKind.TEAM_ANALYSIS and team is not None:
        return [
            PlannedTeamIntelligenceCall(
                name=ToolName.GET_TEAM_INTELLIGENCE,
                arguments=PlannedTeamIntelligenceInput(team=team, position_group=position),
            )
        ]
    if operation is IntentKind.ROLE_FIT and team is not None:
        player = primary or _MISSING_PLAYER
        calls = (
            _named_player_search(
                source,
                player,
                team_filter=player_team_filter,
                position=subject_position,
            )
            if primary is not None
            else []
        )
        calls.append(
            PlannedRoleFitCall(
                name=ToolName.GET_ROLE_FIT,
                arguments=PlannedRoleFitInput(player=player, target_team=team),
            )
        )
        return calls
    if operation is IntentKind.ROLE_RECOMMENDATIONS and team is not None and position is not None:
        return [
            PlannedRoleRecommendationsCall(
                name=ToolName.GET_ROLE_RECOMMENDATIONS,
                arguments=PlannedRoleRecommendationsInput(
                    team=team,
                    position_group=position,
                    limit=_limit(source.limit, 10, RECOMMENDATIONS_MAX_RESULTS),
                ),
            )
        ]
    if operation is IntentKind.METHODOLOGY and topic is not None:
        return [
            PlannedMethodologyCall(
                name=ToolName.GET_METHODOLOGY,
                arguments=methodology_topic,
            )
            for methodology_topic in methodology_topics
        ]
    return []


def _discovery_calls(
    source: LLMPlannerDecision,
    primary: EntityRef | None,
    *,
    player_team_filter: str | None,
    subject_position: PositionGroup | None,
) -> list[PlannedToolCall]:
    if primary is None or primary.player_name is None:
        return []
    if source.intent not in {
        IntentKind.PLAYER_PROFILE,
        IntentKind.PLAYER_COMPARISON,
        IntentKind.SIMILAR_PLAYERS,
        IntentKind.ROLE_FIT,
    }:
        return []
    return _named_player_search(
        source,
        primary,
        team_filter=player_team_filter,
        position=subject_position,
    )


def _search_call(
    source: LLMPlannerDecision,
    *,
    team_filter: str | None,
    position: PositionGroup | None,
) -> PlannedSearchPlayersCall:
    return PlannedSearchPlayersCall(
        name=ToolName.SEARCH_PLAYERS,
        arguments=SearchPlayersInput(
            query=source.search_query.strip() or None,
            team=team_filter,
            position_group=position,
            min_pass_attempts=source.min_pass_attempts,
            sort_by=_sort_field(source.player_sort_by),
            sort_order=source.sort_order,
            limit=_limit(source.limit, 10, PLAYER_SEARCH_MAX_RESULTS),
        ),
    )


def _named_player_search(
    source: LLMPlannerDecision,
    player: EntityRef,
    *,
    team_filter: str | None,
    position: PositionGroup | None = None,
) -> list[PlannedToolCall]:
    if player.player_name is None:
        return []
    return [
        PlannedSearchPlayersCall(
            name=ToolName.SEARCH_PLAYERS,
            arguments=SearchPlayersInput(
                query=player.player_name,
                team=team_filter,
                position_group=position,
                limit=10,
            ),
        )
    ]


def _requested_operations(
    source: LLMPlannerDecision,
    question: str | None,
) -> list[IntentKind]:
    operations = list(dict.fromkeys([source.intent, *source.requested_analyses]))
    if (
        IntentKind.ROLE_FIT in operations
        and source.intent is not IntentKind.PLAYER_PROFILE
        and IntentKind.PLAYER_PROFILE in operations
        and not _explicit_profile_requested(question)
    ):
        operations.remove(IntentKind.PLAYER_PROFILE)
    return operations


def _explicit_profile_requested(question: str | None) -> bool:
    if not question:
        return False
    return bool(
        re.search(
            r"\b(?:player|playing|passing|shooting|attacking|intelligence) profile\b|"
            r"\bplaying style\b|\barchetype\b|"
            r"\b(?:passing|shooting|attacking) (?:metric|metrics|analysis)\b|"
            r"\bprofile comparison\b",
            question,
            re.IGNORECASE,
        )
    )


def _player_team_filter(
    source: LLMPlannerDecision,
    operations: list[IntentKind],
    question: str | None,
) -> str | None:
    if IntentKind.ROLE_FIT not in operations:
        return source.team_name.strip() or None
    return _explicit_candidate_team(question, source.primary_player_name)


def _explicit_candidate_team(question: str | None, player_name: str) -> str | None:
    if not question or not player_name.strip():
        return None
    position = r"(?:midfielder|defender|forward|winger|striker|player)"
    team = r"(?P<team>[A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,2})"
    match = re.search(
        rf"\b(?:the\s+)?{team}\s+{position}\s+{re.escape(player_name.strip())}\b",
        question,
    )
    return match.group("team").strip() if match else None


def _deduplicate_calls(calls: list[PlannedToolCall]) -> list[PlannedToolCall]:
    unique: list[PlannedToolCall] = []
    seen: set[str] = set()
    for call in calls:
        key = json.dumps(call.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            unique.append(call)
    return unique


def _player_ref(player_id: int, player_name: str) -> EntityRef | None:
    if player_id > 0:
        return EntityRef(player_id=player_id)
    name = player_name.strip()
    return EntityRef(player_name=name) if name and not is_placeholder_player_reference(name) else None


def _team_ref(team_id: int, team_name: str) -> TeamRef | None:
    if team_id > 0:
        return TeamRef(team_id=team_id)
    name = team_name.strip()
    return TeamRef(team_name=name) if name else None


def _position(value: LLMPositionGroup) -> PositionGroup | None:
    return None if value is LLMPositionGroup.NONE else PositionGroup(value.value)


def _metric(value: LLMLeaderboardMetric) -> LeaderboardMetric:
    if value is LLMLeaderboardMetric.NONE:
        return LeaderboardMetric.COMPLETION_ABOVE_EXPECTED_PP
    return LeaderboardMetric(value.value)


def _sort_field(value: LLMPlayerSortField) -> PlayerSortField:
    if value is LLMPlayerSortField.NONE:
        return PlayerSortField.PLAYER_NAME
    return PlayerSortField(value.value)


def _topic(value: LLMMethodologyTopic) -> MethodologyInput | None:
    if value is LLMMethodologyTopic.NONE:
        return None
    return MethodologyInput(topic=value.value)


def _metric_topic_fallback() -> MethodologyTopic:
    return MethodologyTopic.XPASS


def _sections(source: LLMPlannerDecision) -> list[PlayerDossierSection]:
    return list(dict.fromkeys(source.requested_sections)) or [PlayerDossierSection.INTELLIGENCE]


def _limit(value: int, default: int, maximum: int) -> int:
    return min(value or default, maximum)
