"""Read-only player services shared by REST endpoints and deterministic tools."""

import unicodedata
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session, aliased

from app.intelligence import get_intelligence_response
from app.models import (
    AttackingAction,
    Pass,
    Player,
    PlayerAttackingProfile,
    PlayerProfile,
    PlayerShootingProfile,
    PlayerSimilarity,
    Shot,
)
from app.presenters import player_identity, player_profile, player_summary
from app.schemas import (
    AttackingActionListResponse,
    AttackingActionResponse,
    AttackingProfileResponse,
    PassListResponse,
    PassResponse,
    PlayerIntelligenceResponse,
    PlayerListResponse,
    PlayerProfileResponse,
    PlayerSortField,
    PositionGroup,
    ShootingProfileResponse,
    ShotListResponse,
    ShotResponse,
    SimilarPlayerResponse,
    SimilarPlayersResponse,
    SortOrder,
)

SIMILARITY_VERSION = "V3.3B"
SIMILARITY_REQUIREMENTS = [
    "outfield player",
    "at least 50 passes",
    "at least 29 carries",
]

PLAYER_SORT_COLUMNS = {
    PlayerSortField.PLAYER_NAME: Player.player_name,
    PlayerSortField.PASS_ATTEMPTS: Player.pass_attempts,
    PlayerSortField.ACTUAL_COMPLETION_RATE: PlayerProfile.actual_completion_rate,
    PlayerSortField.EXPECTED_COMPLETION_RATE: PlayerProfile.expected_completion_rate,
    PlayerSortField.COMPLETION_ABOVE_EXPECTED_PP: (PlayerProfile.completion_above_expected_pp),
    PlayerSortField.PROGRESSIVE_PASS_RATE: PlayerProfile.progressive_pass_rate,
    PlayerSortField.PRESSURE_ABOVE_EXPECTED_PP: (PlayerProfile.pressure_above_expected_pp),
    PlayerSortField.FINAL_THIRD_ENTRIES_PER_100_PASSES: (
        PlayerProfile.final_third_entries_per_100_passes
    ),
}

_APOSTROPHES = frozenset({"'", "’", "ʼ", "＇"})


def normalize_player_name(value: str) -> str:
    """Normalize a player name for deterministic matching without changing storage."""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    characters: list[str] = []
    for character in normalized:
        if character in _APOSTROPHES:
            continue
        if character.isalnum() or unicodedata.category(character).startswith("M"):
            characters.append(character)
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def find_players_by_normalized_name(
    session: Session,
    query: str,
    *,
    limit: int,
) -> list[Player]:
    """Return only the best deterministic player-name match class."""
    normalized_query = normalize_player_name(query)
    query_tokens = frozenset(normalized_query.split())
    if not normalized_query or not query_tokens:
        return []

    ranked: list[tuple[int, str, str, int, Player]] = []
    players = session.scalars(
        select(Player).order_by(Player.player_name.asc(), Player.player_id.asc())
    ).all()
    for player in players:
        normalized_name = normalize_player_name(player.player_name)
        name_tokens = frozenset(normalized_name.split())
        if normalized_name == normalized_query:
            match_class = 0
        elif normalized_query in normalized_name:
            match_class = 1
        elif query_tokens.issubset(name_tokens):
            match_class = 2
        else:
            continue
        ranked.append(
            (
                match_class,
                normalized_name,
                player.player_name.casefold(),
                player.player_id,
                player,
            )
        )

    if not ranked:
        return []
    ranked.sort(key=lambda item: item[:4])
    best_class = ranked[0][0]
    return [item[4] for item in ranked if item[0] == best_class][:limit]


def player_filters(
    search: str | None,
    team: str | None,
    position_group: PositionGroup | None,
    min_pass_attempts: int,
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if search:
        filters.append(Player.player_name.icontains(search, autoescape=True))
    if team:
        filters.append(func.lower(Player.team_name) == team.casefold())
    if position_group:
        filters.append(Player.position_group == position_group.value)
    filters.append(Player.pass_attempts >= min_pass_attempts)
    return filters


def get_player_profile_record(
    session: Session,
    player_id: int,
) -> tuple[Player, PlayerProfile]:
    row = session.execute(
        select(Player, PlayerProfile)
        .join(PlayerProfile, PlayerProfile.player_id == Player.player_id)
        .where(Player.player_id == player_id)
    ).one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player {player_id} was not found.",
        )
    return row[0], row[1]


def search_players(
    session: Session,
    *,
    search: str | None = None,
    team: str | None = None,
    position_group: PositionGroup | None = None,
    min_pass_attempts: int = 0,
    sort_by: PlayerSortField = PlayerSortField.PLAYER_NAME,
    sort_order: SortOrder = SortOrder.ASC,
    limit: int = 50,
    offset: int = 0,
) -> PlayerListResponse:
    filters = player_filters(search, team, position_group, min_pass_attempts)
    total = (
        session.scalar(
            select(func.count())
            .select_from(Player)
            .join(PlayerProfile, PlayerProfile.player_id == Player.player_id)
            .where(*filters)
        )
        or 0
    )

    sort_column = PLAYER_SORT_COLUMNS[sort_by]
    order_expression: Any = (
        sort_column.desc().nulls_last()
        if sort_order is SortOrder.DESC
        else sort_column.asc().nulls_last()
    )
    rows = session.execute(
        select(Player, PlayerProfile)
        .join(PlayerProfile, PlayerProfile.player_id == Player.player_id)
        .where(*filters)
        .order_by(order_expression, Player.player_id.asc())
        .limit(limit)
        .offset(offset)
    ).all()
    return PlayerListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[player_summary(player, profile) for player, profile in rows],
    )


def get_player_profile(session: Session, player_id: int) -> PlayerProfileResponse:
    return player_profile(*get_player_profile_record(session, player_id))


def get_player_intelligence(
    session: Session,
    player_id: int,
) -> PlayerIntelligenceResponse:
    player = session.get(Player, player_id)
    if player is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Player {player_id} was not found.",
        )
    response = get_intelligence_response(session, player)
    if response is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No intelligence profile is available for player {player_id}.",
        )
    return response


def get_similar_players(
    session: Session,
    player_id: int,
    *,
    limit: int = 6,
) -> SimilarPlayersResponse:
    source = session.get(Player, player_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player {player_id} was not found.",
        )

    similar_player = aliased(Player)
    total = (
        session.scalar(
            select(func.count())
            .select_from(PlayerSimilarity)
            .where(PlayerSimilarity.player_id == player_id)
        )
        or 0
    )
    rows = session.execute(
        select(PlayerSimilarity, similar_player)
        .join(
            similar_player,
            PlayerSimilarity.similar_player_id == similar_player.player_id,
        )
        .where(PlayerSimilarity.player_id == player_id)
        .order_by(PlayerSimilarity.rank.asc())
        .limit(limit)
    ).all()
    items = [
        SimilarPlayerResponse(
            similar_player_id=candidate.player_id,
            similar_player_name=candidate.player_name,
            similar_team_name=candidate.team_name,
            similar_position=candidate.position,
            similar_position_group=candidate.position_group,
            rank=similarity.rank,
            rms_distance=similarity.rms_distance,
            similarity_score=similarity.similarity_score,
            same_position_group=similarity.same_position_group,
            closest_feature_1=similarity.closest_feature_1,
            closest_feature_2=similarity.closest_feature_2,
            closest_feature_3=similarity.closest_feature_3,
            closest_style_dimensions=[
                similarity.closest_feature_1,
                similarity.closest_feature_2,
                similarity.closest_feature_3,
            ],
            query_matches_observed=similarity.query_matches_observed,
            candidate_matches_observed=similarity.candidate_matches_observed,
            pair_support_matches=similarity.pair_support_matches,
            sample_support=similarity.sample_support,
            sample_support_explanation=similarity.sample_support_explanation,
            methodology_version=similarity.methodology_version,
        )
        for similarity, candidate in rows
    ]
    return SimilarPlayersResponse(
        source_player=player_identity(source),
        available=bool(items),
        unavailable_reason=(
            None
            if items
            else "V3.3B similarity requires an outfield player with at least 50 passes and 29 carries."
        ),
        methodology_version=SIMILARITY_VERSION,
        query_matches_observed=source.matches_observed,
        query_sample_support="higher" if source.matches_observed >= 3 else "limited",
        eligibility_requirements=SIMILARITY_REQUIREMENTS,
        total=total,
        limit=limit,
        items=items,
    )


def get_player_passes(
    session: Session,
    player_id: int,
    *,
    match_id: int | None = None,
    completed: bool | None = None,
    under_pressure: bool | None = None,
    progressive: bool | None = None,
    min_expected_completion: float | None = None,
    max_expected_completion: float | None = None,
    limit: int = 100,
    offset: int = 0,
) -> PassListResponse:
    if (
        min_expected_completion is not None
        and max_expected_completion is not None
        and min_expected_completion > max_expected_completion
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "min_expected_completion must be less than or equal to max_expected_completion."
            ),
        )
    _require_player(session, player_id)

    filters: list[ColumnElement[bool]] = [Pass.player_id == player_id]
    if match_id is not None:
        filters.append(Pass.match_id == match_id)
    if completed is not None:
        filters.append(Pass.completed.is_(completed))
    if under_pressure is not None:
        filters.append(Pass.under_pressure.is_(under_pressure))
    if progressive is not None:
        filters.append(Pass.progressive.is_(progressive))
    if min_expected_completion is not None:
        filters.append(Pass.expected_completion >= min_expected_completion)
    if max_expected_completion is not None:
        filters.append(Pass.expected_completion <= max_expected_completion)

    total = session.scalar(select(func.count()).select_from(Pass).where(*filters)) or 0
    rows = session.execute(
        select(Pass, AttackingAction)
        .outerjoin(AttackingAction, AttackingAction.pass_index == Pass.pass_index)
        .where(*filters)
        .order_by(Pass.match_id.asc(), Pass.pass_index.asc())
        .limit(limit)
        .offset(offset)
    ).all()
    return PassListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[
            PassResponse.model_validate(pass_row).model_copy(
                update={
                    "attacking_value": action.attacking_value if action else None,
                    "state_value_before": action.state_value_before if action else None,
                    "state_value_after": action.state_value_after if action else None,
                    "pass_risk": action.pass_risk if action else None,
                    "risk_reward_category": (action.risk_reward_category if action else None),
                }
            )
            for pass_row, action in rows
        ],
    )


def get_player_shooting(
    session: Session,
    player_id: int,
) -> ShootingProfileResponse:
    _require_player(session, player_id)
    profile = session.get(PlayerShootingProfile, player_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No shooting profile is available for player {player_id}.",
        )
    return ShootingProfileResponse.model_validate(profile)


def get_player_shots(
    session: Session,
    player_id: int,
    *,
    match_id: int | None = None,
    goal: bool | None = None,
    model_eligible: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> ShotListResponse:
    _require_player(session, player_id)
    filters: list[ColumnElement[bool]] = [Shot.player_id == player_id]
    if match_id is not None:
        filters.append(Shot.match_id == match_id)
    if goal is not None:
        filters.append(Shot.goal.is_(goal))
    if model_eligible is not None:
        filters.append(Shot.model_eligible.is_(model_eligible))
    total = session.scalar(select(func.count()).select_from(Shot).where(*filters)) or 0
    shots = session.scalars(
        select(Shot)
        .where(*filters)
        .order_by(
            Shot.match_id.asc(),
            Shot.period.asc(),
            Shot.minute.asc(),
            Shot.second.asc(),
        )
        .limit(limit)
        .offset(offset)
    ).all()
    return ShotListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[ShotResponse.model_validate(shot) for shot in shots],
    )


def get_player_attacking(
    session: Session,
    player_id: int,
) -> AttackingProfileResponse:
    _require_player(session, player_id)
    profile = session.get(PlayerAttackingProfile, player_id)
    if profile is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No attacking-value profile is available for player {player_id}.",
        )
    return AttackingProfileResponse.model_validate(profile)


def get_player_actions(
    session: Session,
    player_id: int,
    *,
    action_type: str | None = None,
    match_id: int | None = None,
    positive_only: bool = False,
    under_pressure: bool | None = None,
    progressive: bool | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    limit: int = 100,
    offset: int = 0,
) -> AttackingActionListResponse:
    if min_value is not None and max_value is not None and min_value > max_value:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "min_value must not exceed max_value",
        )
    _require_player(session, player_id)
    filters: list[ColumnElement[bool]] = [AttackingAction.player_id == player_id]
    if action_type is not None:
        filters.append(AttackingAction.action_type == action_type)
    if match_id is not None:
        filters.append(AttackingAction.match_id == match_id)
    if positive_only:
        filters.append(AttackingAction.attacking_value > 0)
    if under_pressure is not None:
        filters.append(AttackingAction.under_pressure.is_(under_pressure))
    if progressive is not None:
        filters.append(AttackingAction.progressive.is_(progressive))
    if min_value is not None:
        filters.append(AttackingAction.attacking_value >= min_value)
    if max_value is not None:
        filters.append(AttackingAction.attacking_value <= max_value)
    total = session.scalar(select(func.count()).select_from(AttackingAction).where(*filters)) or 0
    rows = session.scalars(
        select(AttackingAction)
        .where(*filters)
        .order_by(AttackingAction.match_id, AttackingAction.event_index)
        .limit(limit)
        .offset(offset)
    ).all()
    return AttackingActionListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[AttackingActionResponse.model_validate(row) for row in rows],
    )


def _require_player(session: Session, player_id: int) -> Player:
    player = session.get(Player, player_id)
    if player is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player {player_id} was not found.",
        )
    return player
