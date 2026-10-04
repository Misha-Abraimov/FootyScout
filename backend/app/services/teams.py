"""Read-only team intelligence and Role Fit services."""

from dataclasses import dataclass

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

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
    get_team_intelligence as build_team_intelligence,
)
from app.team_intelligence import (
    get_team_roles as build_team_roles,
)
from app.team_intelligence import (
    player_role_fit_response,
    scouting_recommendations_response,
    team_role_response,
)

TARGET_TEAM_ID = 904
TARGET_TEAM_NAME = "Bayer Leverkusen"
UNAVAILABLE_DETAIL = (
    "Full team intelligence is currently available for Bayer Leverkusen, the club "
    "with complete 34-match product-sample coverage."
)


@dataclass(frozen=True)
class RoleFitCohort:
    """Fields that make Role Fit distances directly comparable."""

    target_team_id: int
    position_group: str
    calculation_scope: str
    is_target_team_player: bool


def role_fit_cohort(fit: PlayerRoleFit) -> RoleFitCohort:
    return RoleFitCohort(
        target_team_id=fit.target_team_id,
        position_group=fit.position_group,
        calculation_scope=fit.calculation_scope,
        is_target_team_player=fit.is_target_team_player,
    )


def ranked_role_fit_cohort(
    session: Session,
    cohort: RoleFitCohort,
) -> list[PlayerRoleFit]:
    """Return one comparable cohort with deterministic ordinal tie-breaking."""
    return list(
        session.scalars(
            select(PlayerRoleFit)
            .where(
                PlayerRoleFit.target_team_id == cohort.target_team_id,
                PlayerRoleFit.position_group == cohort.position_group,
                PlayerRoleFit.calculation_scope == cohort.calculation_scope,
                PlayerRoleFit.is_target_team_player.is_(
                    cohort.is_target_team_player
                ),
            )
            .order_by(PlayerRoleFit.role_distance.asc(), PlayerRoleFit.player_id.asc())
        ).all()
    )


def get_team_profile(session: Session, team_id: int) -> TeamStyleProfile:
    profile = session.get(TeamStyleProfile, team_id)
    if profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, UNAVAILABLE_DETAIL)
    return profile


def find_team_profiles_by_name(
    session: Session,
    team_name: str,
) -> list[TeamStyleProfile]:
    """Return qualified team profiles with an exact case-insensitive name."""
    return list(
        session.scalars(
            select(TeamStyleProfile)
            .where(func.lower(TeamStyleProfile.team_name) == team_name.casefold())
            .order_by(TeamStyleProfile.team_id.asc())
        ).all()
    )


def find_loaded_teams_by_name(
    session: Session,
    team_name: str,
) -> list[tuple[int, str]]:
    """Return distinct loaded-cohort teams using case-insensitive substring matching."""
    rows = session.execute(
        select(Player.team_id, Player.team_name)
        .where(Player.team_name.icontains(team_name, autoescape=True))
        .distinct()
        .order_by(Player.team_name.asc(), Player.team_id.asc())
    ).all()
    return [(team_id, name) for team_id, name in rows]


def require_outfield_role(position_group: PositionGroup) -> str:
    if position_group is PositionGroup.GK:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Goalkeeper Role Fit is not available in the current outfield style space.",
        )
    return position_group.value


def get_team_intelligence(
    session: Session,
    team_id: int,
) -> TeamIntelligenceResponse:
    profile = get_team_profile(session, team_id)
    return build_team_intelligence(session, profile)


def get_team_roles(session: Session, team_id: int) -> TeamRolesResponse:
    profile = get_team_profile(session, team_id)
    return build_team_roles(session, team_id, profile.team_name)


def get_team_role(
    session: Session,
    team_id: int,
    position_group: PositionGroup,
) -> TeamRoleResponse:
    get_team_profile(session, team_id)
    group = require_outfield_role(position_group)
    role = session.get(TeamRoleProfile, (team_id, group))
    if role is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "The requested role is unavailable.",
        )
    return team_role_response(role)


def get_role_fit(
    session: Session,
    player_id: int,
    *,
    target_team_id: int = TARGET_TEAM_ID,
) -> PlayerRoleFitResponse:
    player = session.get(Player, player_id)
    if player is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Player {player_id} was not found.",
        )
    target = get_team_profile(session, target_team_id)
    fit = session.get(PlayerRoleFit, (target_team_id, player_id))
    cohort_rank: int | None = None
    cohort_size = 0
    if fit is not None:
        comparable = ranked_role_fit_cohort(session, role_fit_cohort(fit))
        cohort_size = len(comparable)
        cohort_rank = next(
            index
            for index, candidate in enumerate(comparable, start=1)
            if candidate.player_id == fit.player_id
        )
    return player_role_fit_response(
        player,
        fit,
        target_team_id,
        target.team_name,
        cohort_rank=cohort_rank,
        cohort_size=cohort_size,
    )


def get_role_recommendations(
    session: Session,
    team_id: int,
    position_group: PositionGroup,
    *,
    limit: int = 6,
) -> ScoutingRecommendationsResponse:
    get_team_profile(session, team_id)
    group = require_outfield_role(position_group)
    role = session.get(TeamRoleProfile, (team_id, group))
    if role is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "The requested role is unavailable.",
        )
    cohort = RoleFitCohort(
        target_team_id=team_id,
        position_group=group,
        calculation_scope="full_target_role",
        is_target_team_player=False,
    )
    comparable = ranked_role_fit_cohort(session, cohort)
    total = len(comparable)
    fits = comparable[:limit]
    return scouting_recommendations_response(
        session,
        role,
        list(fits),
        total,
        limit,
    )
