"""Deterministic team intelligence and Role Fit tool adapters."""

from sqlalchemy.orm import Session

from app.ai.schemas import (
    RoleFitInput,
    RoleRecommendationsInput,
    TeamIntelligenceInput,
    TeamIntelligenceToolResponse,
)
from app.schemas import PlayerRoleFitResponse, ScoutingRecommendationsResponse
from app.services import teams


def get_team_intelligence(
    session: Session,
    request: TeamIntelligenceInput,
) -> TeamIntelligenceToolResponse:
    if request.position_group is not None:
        return TeamIntelligenceToolResponse(
            role=teams.get_team_role(
                session,
                request.team_id,
                request.position_group,
            )
        )
    return TeamIntelligenceToolResponse(
        intelligence=teams.get_team_intelligence(session, request.team_id)
    )


def get_role_fit(
    session: Session,
    request: RoleFitInput,
) -> PlayerRoleFitResponse:
    return teams.get_role_fit(
        session,
        request.player_id,
        target_team_id=request.target_team_id,
    )


def get_role_recommendations(
    session: Session,
    request: RoleRecommendationsInput,
) -> ScoutingRecommendationsResponse:
    return teams.get_role_recommendations(
        session,
        request.team_id,
        request.position_group,
        limit=request.limit,
    )
