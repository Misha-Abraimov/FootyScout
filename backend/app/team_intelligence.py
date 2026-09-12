"""Database presentation helpers for production V4 team intelligence."""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Player, PlayerRoleFit, TeamRoleProfile, TeamStyleProfile
from app.presenters import player_identity
from app.schemas import (
    PlayerRoleFitResponse,
    ScoutingRecommendationResponse,
    ScoutingRecommendationsResponse,
    TeamIntelligenceResponse,
    TeamRoleDimensionResponse,
    TeamRoleResponse,
    TeamRolesResponse,
    TeamStyleResponse,
)
from app.team_feature_contract import DESCRIPTIVE_FEATURES, FIT_FEATURES

FEATURE_LABELS = {
    "expected_completion_rate": "Expected completion",
    "pressure_pass_rate": "Under-pressure pass rate",
    "progressive_pass_rate": "Progressive-pass rate",
    "long_pass_rate": "Long-pass rate",
    "positive_forward_distance_per_100_passes": "Positive forward distance / 100 passes",
    "carry_share_of_actions": "Carry share",
}
FIT_INTERPRETATION = (
    "Role Fit measures how closely a player's observed playing style resembles "
    "Bayer Leverkusen's observed positional-role style. It does not predict transfer "
    "success or future performance."
)
RECOMMENDATION_DEFINITION = (
    "Scouting Recommendations rank eligible players by Role Fit to Bayer Leverkusen's "
    "observed positional-role profile."
)


def team_style_response(profile: TeamStyleProfile) -> TeamStyleResponse:
    return TeamStyleResponse(
        team_id=profile.team_id,
        team_name=profile.team_name,
        methodology_version=profile.methodology_version,
        sample_scope=profile.sample_scope,
        matches_observed=profile.matches_observed,
        contributors=profile.contributors,
        passes=profile.passes,
        carries=profile.carries,
        actions=profile.actions,
        shots=profile.shots,
        metrics={feature: float(getattr(profile, feature)) for feature in DESCRIPTIVE_FEATURES},
    )


def team_role_response(profile: TeamRoleProfile) -> TeamRoleResponse:
    return TeamRoleResponse(
        team_id=profile.team_id,
        team_name=profile.team_name,
        position_group=profile.position_group,
        methodology_version=profile.methodology_version,
        aggregation_method=profile.aggregation_method,
        matches_observed=profile.matches_observed,
        contributor_count=profile.contributor_count,
        contributors=json.loads(profile.contributors),
        passes=profile.passes,
        carries=profile.carries,
        actions=profile.actions,
        shots=profile.shots,
        support_level=profile.support_level,
        support_message=profile.support_message,
        dimensions=[
            TeamRoleDimensionResponse(
                feature_name=feature,
                label=FEATURE_LABELS[feature],
                raw_value=float(getattr(profile, feature)),
                position_z=float(getattr(profile, f"{feature}_z")),
            )
            for feature in FIT_FEATURES
        ],
        position_context=(
            f"Relative to eligible {profile.position_group} players in the FootyScout "
            "player cohort. Positive and negative values describe style, not quality."
        ),
    )


def get_team_intelligence(
    session: Session, profile: TeamStyleProfile
) -> TeamIntelligenceResponse:
    roles = session.scalars(
        select(TeamRoleProfile)
        .where(TeamRoleProfile.team_id == profile.team_id)
        .order_by(TeamRoleProfile.position_group)
    ).all()
    return TeamIntelligenceResponse(
        team=team_style_response(profile),
        roles=[team_role_response(role) for role in roles],
        fit_definition=FIT_INTERPRETATION,
    )


def get_team_roles(session: Session, team_id: int, team_name: str) -> TeamRolesResponse:
    roles = session.scalars(
        select(TeamRoleProfile)
        .where(TeamRoleProfile.team_id == team_id)
        .order_by(TeamRoleProfile.position_group)
    ).all()
    return TeamRolesResponse(
        team_id=team_id,
        team_name=team_name,
        roles=[team_role_response(role) for role in roles],
    )


def player_role_fit_response(
    player: Player,
    fit: PlayerRoleFit | None,
    target_team_id: int,
    target_team_name: str,
) -> PlayerRoleFitResponse:
    if fit is None:
        if player.position_group == "GK":
            reason = "Role Fit is currently available for outfield DEF, MID, and FWD players only."
        else:
            reason = "Role Fit requires at least 50 passes and 29 carries in the product sample."
        return PlayerRoleFitResponse(
            player=player_identity(player), available=False, unavailable_reason=reason,
            target_team_id=target_team_id, target_team_name=target_team_name,
            position_group=player.position_group, is_target_team_player=False,
            calculation_scope=None, role_distance=None, closest_dimensions=[],
            largest_difference=None, feature_gaps={}, distance_contributions={},
            player_matches_observed=player.matches_observed, sample_support=None,
            sample_support_message=None, role_matches_observed=None,
            role_contributor_count=None, role_actions=None, role_support_message=None,
            methodology_version="V4.3", interpretation=FIT_INTERPRETATION,
        )
    return PlayerRoleFitResponse(
        player=player_identity(player), available=True, unavailable_reason=None,
        target_team_id=fit.target_team_id, target_team_name=fit.target_team_name,
        position_group=fit.position_group,
        is_target_team_player=fit.is_target_team_player,
        calculation_scope=fit.calculation_scope, role_distance=fit.role_distance,
        closest_dimensions=[
            fit.closest_feature_1, fit.closest_feature_2, fit.closest_feature_3
        ],
        largest_difference=fit.largest_difference,
        feature_gaps=json.loads(fit.feature_gaps),
        distance_contributions=json.loads(fit.distance_contributions),
        player_matches_observed=fit.player_matches_observed,
        sample_support=fit.sample_support,
        sample_support_message=fit.sample_support_message,
        role_matches_observed=fit.role_matches_observed,
        role_contributor_count=fit.role_contributor_count,
        role_actions=fit.role_actions,
        role_support_message=fit.role_support_message,
        methodology_version=fit.methodology_version,
        interpretation=FIT_INTERPRETATION,
    )


def scouting_recommendations_response(
    session: Session,
    role: TeamRoleProfile,
    fits: list[PlayerRoleFit],
    total: int,
    limit: int,
) -> ScoutingRecommendationsResponse:
    players = {
        player.player_id: player
        for player in session.scalars(
            select(Player).where(Player.player_id.in_([fit.player_id for fit in fits]))
        ).all()
    }
    return ScoutingRecommendationsResponse(
        target_team_id=role.team_id,
        target_team_name=role.team_name,
        position_group=role.position_group,
        methodology_version="V4.4",
        definition=RECOMMENDATION_DEFINITION,
        disclaimer=(
            "Recommendations describe observed style fit and are not predictions of "
            "transfer success."
        ),
        role_support_message=role.support_message,
        total=total,
        limit=limit,
        items=[
            ScoutingRecommendationResponse(
                rank=int(fit.recommendation_rank),
                player=player_identity(players[fit.player_id]),
                role_distance=fit.role_distance,
                closest_dimensions=[
                    fit.closest_feature_1, fit.closest_feature_2, fit.closest_feature_3
                ],
                largest_difference=fit.largest_difference,
                sample_support=fit.sample_support,
                sample_support_message=fit.sample_support_message,
                player_matches_observed=fit.player_matches_observed,
                archetype_name=fit.archetype_name,
            )
            for fit in fits
        ],
    )
