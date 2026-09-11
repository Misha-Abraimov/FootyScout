"""Player discovery, profile, similarity, and pass-map endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session, aliased

from app.dependencies import get_db
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
SIMILARITY_REQUIREMENTS = ["outfield player", "at least 50 passes", "at least 29 carries"]

router = APIRouter(prefix="/api/players", tags=["players"])

PLAYER_SORT_COLUMNS = {
    PlayerSortField.PLAYER_NAME: Player.player_name,
    PlayerSortField.PASS_ATTEMPTS: Player.pass_attempts,
    PlayerSortField.ACTUAL_COMPLETION_RATE: PlayerProfile.actual_completion_rate,
    PlayerSortField.EXPECTED_COMPLETION_RATE: PlayerProfile.expected_completion_rate,
    PlayerSortField.COMPLETION_ABOVE_EXPECTED_PP: (
        PlayerProfile.completion_above_expected_pp
    ),
    PlayerSortField.PROGRESSIVE_PASS_RATE: PlayerProfile.progressive_pass_rate,
    PlayerSortField.PRESSURE_ABOVE_EXPECTED_PP: PlayerProfile.pressure_above_expected_pp,
    PlayerSortField.FINAL_THIRD_ENTRIES_PER_100_PASSES: (
        PlayerProfile.final_third_entries_per_100_passes
    ),
}


def _player_filters(
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


def get_player_profile_or_404(
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


@router.get("", response_model=PlayerListResponse)
def list_players(
    session: Annotated[Session, Depends(get_db)],
    search: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    team: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    position_group: PositionGroup | None = None,
    min_pass_attempts: Annotated[int, Query(ge=0)] = 0,
    sort_by: PlayerSortField = PlayerSortField.PLAYER_NAME,
    sort_order: SortOrder = SortOrder.ASC,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=100_000)] = 0,
) -> PlayerListResponse:
    """Search and filter players using a fixed, injection-safe sort allowlist."""
    filters = _player_filters(search, team, position_group, min_pass_attempts)
    total = session.scalar(
        select(func.count())
        .select_from(Player)
        .join(PlayerProfile, PlayerProfile.player_id == Player.player_id)
        .where(*filters)
    ) or 0

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


@router.get("/{player_id}", response_model=PlayerProfileResponse)
def get_player(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> PlayerProfileResponse:
    player, profile = get_player_profile_or_404(session, player_id)
    return player_profile(player, profile)


@router.get("/{player_id}/intelligence", response_model=PlayerIntelligenceResponse)
def get_player_intelligence(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> PlayerIntelligenceResponse:
    player = session.get(Player, player_id)
    if player is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Player {player_id} was not found.")
    response = get_intelligence_response(session, player)
    if response is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No intelligence profile is available for player {player_id}.",
        )
    return response


@router.get("/{player_id}/similar", response_model=SimilarPlayersResponse)
def get_similar_players(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=10)] = 6,
) -> SimilarPlayersResponse:
    source = session.get(Player, player_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player {player_id} was not found.",
        )

    similar_player = aliased(Player)
    total = session.scalar(
        select(func.count())
        .select_from(PlayerSimilarity)
        .where(PlayerSimilarity.player_id == player_id)
    ) or 0
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


@router.get("/{player_id}/passes", response_model=PassListResponse)
def get_player_passes(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
    match_id: int | None = None,
    completed: bool | None = None,
    under_pressure: bool | None = None,
    progressive: bool | None = None,
    min_expected_completion: Annotated[float | None, Query(ge=0, le=1)] = None,
    max_expected_completion: Annotated[float | None, Query(ge=0, le=1)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
) -> PassListResponse:
    if (
        min_expected_completion is not None
        and max_expected_completion is not None
        and min_expected_completion > max_expected_completion
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "min_expected_completion must be less than or equal to "
                "max_expected_completion."
            ),
        )
    if session.get(Player, player_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player {player_id} was not found.",
        )

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

    total = session.scalar(
        select(func.count()).select_from(Pass).where(*filters)
    ) or 0
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
                    "risk_reward_category": action.risk_reward_category if action else None,
                }
            )
            for pass_row, action in rows
        ],
    )


@router.get("/{player_id}/shooting", response_model=ShootingProfileResponse)
def get_player_shooting(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> ShootingProfileResponse:
    if session.get(Player, player_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player {player_id} was not found.",
        )
    profile = session.get(PlayerShootingProfile, player_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No shooting profile is available for player {player_id}.",
        )
    return ShootingProfileResponse.model_validate(profile)


@router.get("/{player_id}/shots", response_model=ShotListResponse)
def get_player_shots(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
    match_id: int | None = None,
    goal: bool | None = None,
    model_eligible: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
) -> ShotListResponse:
    if session.get(Player, player_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player {player_id} was not found.",
        )
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
        .order_by(Shot.match_id.asc(), Shot.period.asc(), Shot.minute.asc(), Shot.second.asc())
        .limit(limit)
        .offset(offset)
    ).all()
    return ShotListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[ShotResponse.model_validate(shot) for shot in shots],
    )


@router.get("/{player_id}/attacking", response_model=AttackingProfileResponse)
def get_player_attacking(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> AttackingProfileResponse:
    if session.get(Player, player_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Player {player_id} was not found.")
    profile = session.get(PlayerAttackingProfile, player_id)
    if profile is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No attacking-value profile is available for player {player_id}.",
        )
    return AttackingProfileResponse.model_validate(profile)


@router.get("/{player_id}/actions", response_model=AttackingActionListResponse)
def get_player_actions(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
    action_type: Annotated[str | None, Query(pattern="^(Pass|Carry)$")] = None,
    match_id: int | None = None,
    positive_only: bool = False,
    under_pressure: bool | None = None,
    progressive: bool | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
) -> AttackingActionListResponse:
    if min_value is not None and max_value is not None and min_value > max_value:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "min_value must not exceed max_value")
    if session.get(Player, player_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Player {player_id} was not found.")
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
