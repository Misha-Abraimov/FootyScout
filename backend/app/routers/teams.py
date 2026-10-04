"""Production V4 team intelligence, Role Fit, and scouting endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas import (
    PlayerRoleFitResponse,
    PositionGroup,
    ScoutingRecommendationsResponse,
    TeamIntelligenceResponse,
    TeamRoleResponse,
    TeamRolesResponse,
)
from app.services import teams as team_service

router = APIRouter(tags=["team intelligence"])
TARGET_TEAM_ID = team_service.TARGET_TEAM_ID
TARGET_TEAM_NAME = team_service.TARGET_TEAM_NAME


@router.get("/api/teams/{team_id}/intelligence", response_model=TeamIntelligenceResponse)
def team_intelligence(
    team_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> TeamIntelligenceResponse:
    return team_service.get_team_intelligence(session, team_id)


@router.get("/api/teams/{team_id}/roles", response_model=TeamRolesResponse)
def team_roles(
    team_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> TeamRolesResponse:
    return team_service.get_team_roles(session, team_id)


@router.get("/api/teams/{team_id}/roles/{position_group}", response_model=TeamRoleResponse)
def team_role(
    team_id: int,
    position_group: PositionGroup,
    session: Annotated[Session, Depends(get_db)],
) -> TeamRoleResponse:
    return team_service.get_team_role(session, team_id, position_group)


@router.get("/api/players/{player_id}/role-fit", response_model=PlayerRoleFitResponse)
def player_role_fit(
    player_id: int,
    session: Annotated[Session, Depends(get_db)],
    target_team_id: Annotated[int, Query(ge=1)] = TARGET_TEAM_ID,
) -> PlayerRoleFitResponse:
    return team_service.get_role_fit(
        session,
        player_id,
        target_team_id=target_team_id,
    )


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
    return team_service.get_role_recommendations(
        session,
        team_id,
        position_group,
        limit=limit,
    )
