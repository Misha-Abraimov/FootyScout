"""Player discovery, profile, similarity, and pass-map endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas import (
    AttackingActionListResponse,
    AttackingProfileResponse,
    PassListResponse,
    PlayerIntelligenceResponse,
    PlayerListResponse,
    PlayerProfileResponse,
    PlayerSortField,
    PositionGroup,
    ShootingProfileResponse,
    ShotListResponse,
    SimilarPlayersResponse,
    SortOrder,
)
from app.services import players as player_service

router = APIRouter(prefix="/api/players", tags=["players"])


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
    return player_service.search_players(
        session,
        search=search,
        team=team,
        position_group=position_group,
        min_pass_attempts=min_pass_attempts,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
    )


@router.get("/{player_id}", response_model=PlayerProfileResponse)
def get_player(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> PlayerProfileResponse:
    return player_service.get_player_profile(session, player_id)


@router.get("/{player_id}/intelligence", response_model=PlayerIntelligenceResponse)
def get_player_intelligence(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> PlayerIntelligenceResponse:
    return player_service.get_player_intelligence(session, player_id)


@router.get("/{player_id}/similar", response_model=SimilarPlayersResponse)
def get_similar_players(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=10)] = 6,
) -> SimilarPlayersResponse:
    return player_service.get_similar_players(session, player_id, limit=limit)


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
    return player_service.get_player_passes(
        session,
        player_id,
        match_id=match_id,
        completed=completed,
        under_pressure=under_pressure,
        progressive=progressive,
        min_expected_completion=min_expected_completion,
        max_expected_completion=max_expected_completion,
        limit=limit,
        offset=offset,
    )


@router.get("/{player_id}/shooting", response_model=ShootingProfileResponse)
def get_player_shooting(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> ShootingProfileResponse:
    return player_service.get_player_shooting(session, player_id)


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
    return player_service.get_player_shots(
        session,
        player_id,
        match_id=match_id,
        goal=goal,
        model_eligible=model_eligible,
        limit=limit,
        offset=offset,
    )


@router.get("/{player_id}/attacking", response_model=AttackingProfileResponse)
def get_player_attacking(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> AttackingProfileResponse:
    return player_service.get_player_attacking(session, player_id)


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
    return player_service.get_player_actions(
        session,
        player_id,
        action_type=action_type,
        match_id=match_id,
        positive_only=positive_only,
        under_pressure=under_pressure,
        progressive=progressive,
        min_value=min_value,
        max_value=max_value,
        limit=limit,
        offset=offset,
    )
