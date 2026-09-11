"""Production V4 team intelligence, Role Fit, and scouting endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.models import Player, PlayerRoleFit, TeamRoleProfile, TeamStyleProfile
from app.schemas import (
    PlayerRoleFitResponse,
    PositionGroup,
    ScoutingRecommendationsResponse,
    TeamIntelligenceResponse,
    TeamRoleResponse,
    TeamRolesResponse,
)
from app.team_intelligence import (
    get_team_intelligence,
    get_team_roles,
    player_role_fit_response,
    scouting_recommendations_response,
    team_role_response,
)

router = APIRouter(tags=["team intelligence"])
TARGET_TEAM_ID = 904
TARGET_TEAM_NAME = "Bayer Leverkusen"
UNAVAILABLE_DETAIL = (
    "Full team intelligence is currently available for Bayer Leverkusen, the club "
    "with complete 34-match product-sample coverage."
)


def _team_or_unavailable(session: Session, team_id: int) -> TeamStyleProfile:
    profile = session.get(TeamStyleProfile, team_id)
    if profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, UNAVAILABLE_DETAIL)
    return profile


def _outfield_role(position_group: PositionGroup) -> str:
    if position_group is PositionGroup.GK:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Goalkeeper Role Fit is not available in the current outfield style space.",
        )
    return position_group.value


@router.get("/api/teams/{team_id}/intelligence", response_model=TeamIntelligenceResponse)
def team_intelligence(
    team_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> TeamIntelligenceResponse:
    profile = _team_or_unavailable(session, team_id)
    return get_team_intelligence(session, profile)


@router.get("/api/teams/{team_id}/roles", response_model=TeamRolesResponse)
def team_roles(
    team_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> TeamRolesResponse:
    profile = _team_or_unavailable(session, team_id)
    return get_team_roles(session, team_id, profile.team_name)


@router.get(
    "/api/teams/{team_id}/roles/{position_group}", response_model=TeamRoleResponse
)
def team_role(
    team_id: int,
    position_group: PositionGroup,
    session: Annotated[Session, Depends(get_db)],
) -> TeamRoleResponse:
    _team_or_unavailable(session, team_id)
    group = _outfield_role(position_group)
    role = session.get(TeamRoleProfile, (team_id, group))
    if role is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The requested role is unavailable.")
    return team_role_response(role)


@router.get("/api/players/{player_id}/role-fit", response_model=PlayerRoleFitResponse)
def player_role_fit(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
    target_team_id: Annotated[int, Query(ge=1)] = TARGET_TEAM_ID,
) -> PlayerRoleFitResponse:
    player = session.get(Player, player_id)
    if player is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Player {player_id} was not found.")
    target = _team_or_unavailable(session, target_team_id)
    fit = session.get(PlayerRoleFit, (target_team_id, player_id))
    return player_role_fit_response(player, fit, target_team_id, target.team_name)


@router.get(
    "/api/teams/{team_id}/roles/{position_group}/recommendations",
    response_model=ScoutingRecommendationsResponse,
)
def scouting_recommendations(
    team_id: int,
    position_group: PositionGroup,
    session: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=50)] = 6,
) -> ScoutingRecommendationsResponse:
    _team_or_unavailable(session, team_id)
    group = _outfield_role(position_group)
    role = session.get(TeamRoleProfile, (team_id, group))
    if role is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The requested role is unavailable.")
    filters = (
        PlayerRoleFit.target_team_id == team_id,
        PlayerRoleFit.position_group == group,
        PlayerRoleFit.is_target_team_player.is_(False),
    )
    total = session.scalar(select(func.count()).select_from(PlayerRoleFit).where(*filters)) or 0
    fits = session.scalars(
        select(PlayerRoleFit)
        .where(*filters)
        .order_by(PlayerRoleFit.role_distance.asc(), PlayerRoleFit.player_id.asc())
        .limit(limit)
    ).all()
    return scouting_recommendations_response(session, role, list(fits), total, limit)
