"""Deterministic validation and entity resolution for untrusted ScoutPlan output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.ai.entity_resolution import resolve_player, resolve_team
from app.ai.policy import MAX_TOOL_CALLS
from app.ai.schemas import (
    MISSING_PLAYER_REFERENCE,
    SUBJECT_CURRENT_TEAM_REFERENCE,
    EntityRef,
    EntityResolutionStatus,
    IntentKind,
    NormalizedPlan,
    NormalizedToolCall,
    PlannedComparePlayersCall,
    PlannedLeaderboardCall,
    PlannedMethodologyCall,
    PlannedPlayerDossierCall,
    PlannedRoleFitCall,
    PlannedRoleRecommendationsCall,
    PlannedSearchPlayersCall,
    PlannedSimilarPlayersCall,
    PlannedTeamIntelligenceCall,
    PlannerDecision,
    PlanPreparationReasonCode,
    PlanPreparationResult,
    PlanPreparationStatus,
    PlayerResolution,
    PositionGroup,
    ResolvedEntity,
    ScoutPlan,
    SearchPlayersInput,
    TeamRef,
    TeamResolution,
    ToolName,
)
from app.ai.tools.registry import get_tool_definition
from app.services.players import normalize_player_name

ALLOWED_TOOLS_BY_INTENT: dict[IntentKind, frozenset[ToolName]] = {
    IntentKind.PLAYER_SEARCH: frozenset({ToolName.SEARCH_PLAYERS}),
    IntentKind.PLAYER_PROFILE: frozenset(
        {ToolName.SEARCH_PLAYERS, ToolName.GET_PLAYER_DOSSIER, ToolName.GET_METHODOLOGY}
    ),
    IntentKind.PLAYER_COMPARISON: frozenset(
        {
            ToolName.SEARCH_PLAYERS,
            ToolName.COMPARE_PLAYERS,
            ToolName.GET_PLAYER_DOSSIER,
            ToolName.GET_METHODOLOGY,
        }
    ),
    IntentKind.SIMILAR_PLAYERS: frozenset(
        {
            ToolName.SEARCH_PLAYERS,
            ToolName.GET_SIMILAR_PLAYERS,
            ToolName.GET_ROLE_FIT,
            ToolName.GET_ROLE_RECOMMENDATIONS,
            ToolName.GET_TEAM_INTELLIGENCE,
            ToolName.GET_METHODOLOGY,
        }
    ),
    IntentKind.LEADERBOARD: frozenset(
        {ToolName.GET_LEADERBOARD, ToolName.GET_PLAYER_DOSSIER, ToolName.GET_METHODOLOGY}
    ),
    IntentKind.TEAM_ANALYSIS: frozenset({ToolName.GET_TEAM_INTELLIGENCE, ToolName.GET_METHODOLOGY}),
    IntentKind.ROLE_FIT: frozenset(
        {
            ToolName.SEARCH_PLAYERS,
            ToolName.GET_ROLE_FIT,
            ToolName.GET_TEAM_INTELLIGENCE,
            ToolName.GET_PLAYER_DOSSIER,
            ToolName.GET_METHODOLOGY,
        }
    ),
    IntentKind.ROLE_RECOMMENDATIONS: frozenset(
        {
            ToolName.GET_ROLE_RECOMMENDATIONS,
            ToolName.GET_TEAM_INTELLIGENCE,
            ToolName.GET_LEADERBOARD,
            ToolName.GET_ROLE_FIT,
            ToolName.GET_SIMILAR_PLAYERS,
            ToolName.GET_PLAYER_DOSSIER,
            ToolName.GET_METHODOLOGY,
        }
    ),
    IntentKind.METHODOLOGY: frozenset({ToolName.GET_METHODOLOGY}),
}


@dataclass
class _PreparationContext:
    session: Session
    player_resolutions: list[PlayerResolution]
    team_resolutions: list[TeamResolution]
    resolved_entities: list[ResolvedEntity]
    player_cache: dict[tuple[str, str | int], PlayerResolution]
    team_cache: dict[tuple[str, str | int], TeamResolution]
    player_searches: dict[str, SearchPlayersInput]


class _PreparationHalt(Exception):
    def __init__(self, result: PlanPreparationResult) -> None:
        self.result = result


def prepare_plan(session: Session, plan: ScoutPlan) -> PlanPreparationResult:
    """Resolve entities and produce only registry-valid, policy-approved tool calls."""
    if plan.decision is PlannerDecision.CLARIFICATION_REQUIRED:
        return PlanPreparationResult(
            status=PlanPreparationStatus.CLARIFICATION_REQUIRED,
            clarification_message=plan.clarification_message,
            limitations=plan.limitations,
        )
    if plan.decision is PlannerDecision.UNSUPPORTED:
        return PlanPreparationResult(
            status=PlanPreparationStatus.UNSUPPORTED,
            unsupported_reason=plan.unsupported_reason,
            limitations=plan.limitations,
        )
    if len(plan.calls) > MAX_TOOL_CALLS:
        return PlanPreparationResult(
            status=PlanPreparationStatus.INVALID,
            error_message=f"Plans may contain at most {MAX_TOOL_CALLS} calls.",
        )

    allowed = ALLOWED_TOOLS_BY_INTENT[plan.intent.kind]
    context = _PreparationContext(session, [], [], [], {}, {}, {})
    normalized_arguments: list[tuple[ToolName, dict[str, Any]]] = []
    try:
        for call in plan.calls:
            if call.name not in allowed:
                return _result(
                    context,
                    PlanPreparationStatus.INVALID,
                    error_message=(
                        f"Tool {call.name.value!r} is not valid for intent "
                        f"{plan.intent.kind.value!r}."
                    ),
                    limitations=plan.limitations,
                )
            normalized_arguments.append((call.name, _normalize_call(context, call)))
    except _PreparationHalt as halt:
        return halt.result.model_copy(
            update={
                "limitations": list(dict.fromkeys([*plan.limitations, *halt.result.limitations]))
            }
        )

    unique_calls: list[tuple[ToolName, dict[str, Any]]] = []
    seen: set[str] = set()
    duplicate_removed = False
    for name, arguments in normalized_arguments:
        key = f"{name.value}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'))}"
        if key in seen:
            duplicate_removed = True
            continue
        seen.add(key)
        unique_calls.append((name, arguments))

    limitations = list(plan.limitations)
    if duplicate_removed:
        limitations.append("An exact duplicate tool call was removed during normalization.")
    normalized_plan = NormalizedPlan(
        intent=plan.intent,
        calls=[
            NormalizedToolCall(call_id=f"call-{index}", name=name, arguments=arguments)
            for index, (name, arguments) in enumerate(unique_calls, start=1)
        ],
        resolved_entities=context.resolved_entities,
        limitations=list(dict.fromkeys(limitations)),
    )
    return _result(
        context,
        PlanPreparationStatus.READY,
        normalized_plan=normalized_plan,
        limitations=normalized_plan.limitations,
    )


def _normalize_call(context: _PreparationContext, call: Any) -> dict[str, Any]:
    if isinstance(call, PlannedSearchPlayersCall):
        arguments = call.arguments.model_dump(mode="json")
        for field in ("query", "team"):
            if isinstance(arguments.get(field), str):
                arguments[field] = sanitize_reference_text(arguments[field])
    elif isinstance(call, PlannedPlayerDossierCall):
        player_id = _resolve_player(context, call.arguments.player)
        arguments = {
            "player_id": player_id,
            "sections": [section.value for section in call.arguments.sections],
        }
    elif isinstance(call, PlannedComparePlayersCall):
        player_ids = tuple(_resolve_player(context, ref) for ref in call.arguments.players)
        if player_ids[0] == player_ids[1]:
            _halt(
                context,
                PlanPreparationStatus.CLARIFICATION_REQUIRED,
                clarification_message="Comparison requires two distinct players.",
            )
        arguments = {"player_ids": list(player_ids)}
    elif isinstance(call, PlannedSimilarPlayersCall):
        player_id = _resolve_player(context, call.arguments.player)
        player = next(
            resolution.player
            for resolution in reversed(context.player_resolutions)
            if resolution.player_id == player_id
        )
        candidate_position = call.arguments.candidate_position_group
        if (
            player is not None
            and candidate_position is not None
            and player.position_group != candidate_position.value
        ):
            _halt(
                context,
                PlanPreparationStatus.NOT_FOUND,
                reason_code=PlanPreparationReasonCode.CANDIDATE_POSITION_INCOMPATIBLE,
                error_message=(
                    "Playing-style similarity is available only within the subject's "
                    f"{player.position_group} position group; no "
                    f"{candidate_position.value} candidate cohort applies."
                ),
            )
        arguments = {
            "player_id": player_id,
            "limit": call.arguments.limit,
        }
    elif isinstance(call, PlannedLeaderboardCall):
        arguments = call.arguments.model_dump(mode="json")
    elif isinstance(call, PlannedTeamIntelligenceCall):
        arguments = {
            "team_id": _resolve_qualified_team(context, call.arguments.team),
            "position_group": (
                call.arguments.position_group.value
                if call.arguments.position_group is not None
                else None
            ),
        }
        if call.arguments.position_group == PositionGroup.GK:
            _halt(
                context,
                PlanPreparationStatus.UNSUPPORTED,
                unsupported_reason="Goalkeeper team-role intelligence is not available.",
            )
    elif isinstance(call, PlannedRoleFitCall):
        target_team = call.arguments.target_team
        if target_team.team_name == SUBJECT_CURRENT_TEAM_REFERENCE:
            # A current-team reference depends on the resolved subject, so this
            # special path intentionally preserves player-first resolution.
            player_id = _resolve_player(context, call.arguments.player)
            player = next(
                resolution.player
                for resolution in reversed(context.player_resolutions)
                if resolution.player_id == player_id
            )
            if player is None:
                _halt(
                    context,
                    PlanPreparationStatus.NOT_FOUND,
                    error_message="The player's current loaded team could not be established.",
                )
            target_team_id = _resolve_qualified_team(
                context,
                TeamRef(team_id=player.team_id),
            )
        else:
            # Resolve production team coverage first. This lets a clear request
            # for an unsupported team terminate as insufficient evidence before
            # asking for an otherwise-missing player.
            target_team_id = _resolve_qualified_team(context, target_team)
            if call.arguments.player.player_name == MISSING_PLAYER_REFERENCE:
                _halt(
                    context,
                    PlanPreparationStatus.CLARIFICATION_REQUIRED,
                    clarification_message="Which player should FootyScout analyze?",
                )
            player_id = _resolve_player(context, call.arguments.player)
            player = next(
                resolution.player
                for resolution in reversed(context.player_resolutions)
                if resolution.player_id == player_id
            )
        if player is not None and player.position_group == PositionGroup.GK.value:
            _halt(
                context,
                PlanPreparationStatus.UNSUPPORTED,
                unsupported_reason="Goalkeeper Role Fit is not available.",
            )
        arguments = {
            "player_id": player_id,
            "target_team_id": target_team_id,
        }
    elif isinstance(call, PlannedRoleRecommendationsCall):
        if call.arguments.position_group == PositionGroup.GK:
            _halt(
                context,
                PlanPreparationStatus.UNSUPPORTED,
                unsupported_reason="Goalkeeper role recommendations are not available.",
            )
        arguments = {
            "team_id": _resolve_qualified_team(context, call.arguments.team),
            "position_group": call.arguments.position_group.value,
            "limit": call.arguments.limit,
        }
    elif isinstance(call, PlannedMethodologyCall):
        arguments = call.arguments.model_dump(mode="json")
    else:  # pragma: no cover - discriminated schema makes this unreachable
        raise TypeError(f"Unsupported planned call type: {type(call)!r}")

    definition = get_tool_definition(call.name)
    try:
        validated = definition.input_model.model_validate(arguments)
    except ValidationError as exc:
        _halt(
            context,
            PlanPreparationStatus.INVALID,
            error_message=f"Normalized arguments for {call.name.value} are invalid: {exc}",
        )
    if isinstance(call, PlannedSearchPlayersCall) and validated.query:
        context.player_searches[normalize_player_name(validated.query)] = validated
    return validated.model_dump(mode="json")


def _resolve_player(context: _PreparationContext, reference: EntityRef) -> int:
    if reference.player_name is not None:
        reference = reference.model_copy(
            update={"player_name": sanitize_reference_text(reference.player_name)}
        )
    key = (
        ("id", reference.player_id)
        if reference.player_id is not None
        else ("name", normalize_player_name(reference.player_name))
    )
    resolution = context.player_cache.get(key)
    if resolution is None:
        resolution = resolve_player(
            context.session,
            reference,
            search_request=(
                context.player_searches.get(normalize_player_name(reference.player_name))
                if reference.player_name is not None
                else None
            ),
        )
        context.player_cache[key] = resolution
        context.player_resolutions.append(resolution)
    if resolution.status is EntityResolutionStatus.NOT_FOUND:
        _halt(
            context,
            PlanPreparationStatus.NOT_FOUND,
            error_message=f"Player {resolution.query!r} was not found.",
        )
    if resolution.status is EntityResolutionStatus.AMBIGUOUS:
        _halt(
            context,
            PlanPreparationStatus.CLARIFICATION_REQUIRED,
            clarification_message=_player_clarification_message(resolution),
        )
    assert resolution.player_id is not None and resolution.player is not None
    _record_entity(
        context,
        ResolvedEntity(
            entity_type="player",
            query=resolution.query,
            stable_id=resolution.player_id,
            display_name=resolution.player.player_name,
        ),
    )
    return resolution.player_id


def _player_clarification_message(resolution: PlayerResolution) -> str:
    candidates = resolution.candidates[:5]
    details = "; ".join(
        f"{candidate.player_name} — {candidate.team_name}, {candidate.position}"
        for candidate in candidates
    )
    suffix = "" if len(resolution.candidates) <= len(candidates) else "; additional matches"
    return (
        f"I found multiple players matching {resolution.query!r}. Which one do you mean? "
        f"{details}{suffix}"
    )


def _resolve_qualified_team(context: _PreparationContext, reference: TeamRef) -> int:
    if reference.team_name is not None:
        reference = reference.model_copy(
            update={"team_name": sanitize_reference_text(reference.team_name)}
        )
    key = (
        ("id", reference.team_id)
        if reference.team_id is not None
        else ("name", reference.team_name.casefold())
    )
    resolution = context.team_cache.get(key)
    if resolution is None:
        resolution = resolve_team(context.session, reference)
        context.team_cache[key] = resolution
        context.team_resolutions.append(resolution)
    if resolution.status is EntityResolutionStatus.NOT_FOUND:
        _halt(
            context,
            PlanPreparationStatus.NOT_FOUND,
            error_message=f"Team {resolution.query!r} was not found in the loaded cohort.",
        )
    if resolution.status is EntityResolutionStatus.AMBIGUOUS:
        _halt(
            context,
            PlanPreparationStatus.CLARIFICATION_REQUIRED,
            clarification_message=f"Team {resolution.query!r} is ambiguous; choose a candidate.",
        )
    if not resolution.analytics_supported:
        _halt(
            context,
            PlanPreparationStatus.NOT_FOUND,
            reason_code=PlanPreparationReasonCode.UNSUPPORTED_TEAM_CAPABILITY,
            error_message=(
                f"Team {resolution.query!r} exists in the loaded cohort but does not have "
                "qualified production team-intelligence or Role Fit support."
            ),
        )
    assert resolution.team_id is not None and resolution.team is not None
    _record_entity(
        context,
        ResolvedEntity(
            entity_type="team",
            query=resolution.query,
            stable_id=resolution.team_id,
            display_name=resolution.team.team_name,
        ),
    )
    return resolution.team_id


def _record_entity(context: _PreparationContext, entity: ResolvedEntity) -> None:
    key = (entity.entity_type, entity.stable_id)
    if all((item.entity_type, item.stable_id) != key for item in context.resolved_entities):
        context.resolved_entities.append(entity)


def _halt(
    context: _PreparationContext,
    status: PlanPreparationStatus,
    *,
    reason_code: PlanPreparationReasonCode | None = None,
    clarification_message: str | None = None,
    unsupported_reason: str | None = None,
    error_message: str | None = None,
) -> None:
    raise _PreparationHalt(
        _result(
            context,
            status,
            reason_code=reason_code,
            clarification_message=clarification_message,
            unsupported_reason=unsupported_reason,
            error_message=error_message,
        )
    )


def _result(
    context: _PreparationContext,
    status: PlanPreparationStatus,
    *,
    reason_code: PlanPreparationReasonCode | None = None,
    normalized_plan: NormalizedPlan | None = None,
    clarification_message: str | None = None,
    unsupported_reason: str | None = None,
    error_message: str | None = None,
    limitations: list[str] | None = None,
) -> PlanPreparationResult:
    return PlanPreparationResult(
        status=status,
        reason_code=reason_code,
        normalized_plan=normalized_plan,
        player_resolutions=context.player_resolutions,
        team_resolutions=context.team_resolutions,
        clarification_message=clarification_message,
        unsupported_reason=unsupported_reason,
        error_message=error_message,
        limitations=limitations or [],
    )


def sanitize_reference_text(value: str) -> str:
    """Remove obvious trailing sentence punctuation without changing internal names."""
    normalized = value.strip()
    if not normalized:
        return normalized
    if normalized.endswith(("?", "!")):
        return normalized.rstrip("?!").rstrip()
    if normalized.endswith(".") and normalized.count(".") == 1 and len(normalized) > 3:
        return normalized[:-1].rstrip()
    return normalized
