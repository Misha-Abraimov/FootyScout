"""Canonical cohort-relative Role Fit ordering contracts."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Player, PlayerRoleFit, TeamRoleProfile, TeamStyleProfile
from app.schemas import PositionGroup
from app.services.teams import get_role_fit, get_role_recommendations
from app.team_intelligence import FIT_INTERPRETATION


def _player(player_id: int, position_group: str, *, target: bool = False) -> Player:
    return Player(
        player_id=player_id,
        player_name=f"Player {player_id}",
        team_id=904 if target else 100 + player_id,
        team_name="Bayer Leverkusen" if target else f"Club {player_id}",
        position="Center Midfield" if position_group == "MID" else "Center Back",
        position_group=position_group,
        matches_observed=5,
        pass_attempts=100,
        overall_reliable=True,
    )


def _fit(
    player: Player,
    distance: float,
    *,
    target_team_id: int = 904,
    scope: str = "full_target_role",
    is_target_player: bool = False,
    recommendation_rank: int | None = None,
) -> PlayerRoleFit:
    return PlayerRoleFit(
        target_team_id=target_team_id,
        player_id=player.player_id,
        target_team_name=(
            "Bayer Leverkusen" if target_team_id == 904 else "Other Target"
        ),
        player_name=player.player_name,
        player_team_name=player.team_name,
        position=player.position,
        position_group=player.position_group,
        is_target_team_player=is_target_player,
        calculation_scope=scope,
        role_distance=distance,
        recommendation_rank=recommendation_rank,
        closest_feature_1="pressure_pass_rate",
        closest_feature_2="progressive_pass_rate",
        closest_feature_3="expected_completion_rate",
        largest_difference="long_pass_rate",
        feature_gaps='{"long_pass_rate": 0.4}',
        distance_contributions='{"long_pass_rate": 0.16}',
        player_matches_observed=5,
        player_pass_attempts=100,
        player_carries=40,
        sample_support="higher",
        sample_support_message="Observed across five matches.",
        role_matches_observed=34,
        role_contributor_count=10,
        role_actions=10000,
        role_support_message="Observed role support.",
        archetype_id=None,
        archetype_name=None,
        methodology_version="V4.3",
    )


def _team_profile(team_id: int, name: str) -> TeamStyleProfile:
    return TeamStyleProfile(
        team_id=team_id,
        team_name=name,
        methodology_version="V4.1",
        sample_scope="Test scope.",
        matches_observed=34,
        contributors=20,
        passes=1000,
        carries=800,
        actions=1800,
        shots=200,
        expected_completion_rate=0.8,
        pressure_pass_rate=0.1,
        progressive_pass_rate=0.1,
        long_pass_rate=0.1,
        positive_forward_distance_per_100_passes=500.0,
        carry_share_of_actions=0.4,
        average_forward_distance=5.0,
        final_third_entries_per_100_passes=10.0,
        progressive_carry_rate=0.1,
        progressive_action_rate=0.1,
        pressure_action_rate=0.1,
        shots_per_match=10.0,
        xg_per_shot=0.1,
        xg_per_match=1.0,
        attacking_value_per_100_actions=0.2,
    )


def _mid_role() -> TeamRoleProfile:
    return TeamRoleProfile(
        team_id=904,
        team_name="Bayer Leverkusen",
        position_group="MID",
        methodology_version="V4.2",
        aggregation_method="pooled_events_actions",
        matches_observed=34,
        contributor_count=10,
        contributors="[]",
        passes=1000,
        carries=800,
        actions=1800,
        shots=20,
        support_level="established",
        support_message="Observed role support.",
        expected_completion_rate=0.8,
        pressure_pass_rate=0.1,
        progressive_pass_rate=0.1,
        long_pass_rate=0.1,
        positive_forward_distance_per_100_passes=500.0,
        carry_share_of_actions=0.4,
        expected_completion_rate_z=0.0,
        pressure_pass_rate_z=0.0,
        progressive_pass_rate_z=0.0,
        long_pass_rate_z=0.0,
        positive_forward_distance_per_100_passes_z=0.0,
        carry_share_of_actions_z=0.0,
    )


@pytest.fixture
def role_fit_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with session_factory() as session:
        external = [_player(player_id, "MID") for player_id in (10, 11, 12)]
        target_player = _player(13, "MID", target=True)
        defender = _player(14, "DEF")
        other_target = _player(15, "MID")
        unavailable = _player(16, "MID")
        players = [*external, target_player, defender, other_target, unavailable]
        session.add_all(players)
        session.add_all(
            [
                _team_profile(904, "Bayer Leverkusen"),
                _team_profile(905, "Other Target"),
                _mid_role(),
            ]
        )
        session.add_all(
            [
                _fit(external[0], 0.2, recommendation_rank=1),
                _fit(external[1], 0.2, recommendation_rank=2),
                _fit(external[2], 0.4, recommendation_rank=3),
                _fit(
                    target_player,
                    0.1,
                    scope="leave_self_out_target_role",
                    is_target_player=True,
                ),
                _fit(defender, 0.05, recommendation_rank=1),
                _fit(
                    other_target,
                    0.01,
                    target_team_id=905,
                    recommendation_rank=1,
                ),
            ]
        )
        session.commit()
        yield session
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_role_fit_rank_uses_distance_then_player_id_tie_break(
    role_fit_session: Session,
) -> None:
    first = get_role_fit(role_fit_session, 10)
    tied_second = get_role_fit(role_fit_session, 11)
    third = get_role_fit(role_fit_session, 12)
    assert (first.cohort_rank, tied_second.cohort_rank, third.cohort_rank) == (1, 2, 3)
    assert first.cohort_size == tied_second.cohort_size == third.cohort_size == 3
    assert first.role_distance == pytest.approx(0.2)
    assert "not player quality" in (first.ranking_interpretation or "")


def test_role_fit_cohort_filters_scope_status_position_and_target_team(
    role_fit_session: Session,
) -> None:
    current = get_role_fit(role_fit_session, 13)
    defender = get_role_fit(role_fit_session, 14)
    other_target = get_role_fit(role_fit_session, 15, target_team_id=905)
    unavailable = get_role_fit(role_fit_session, 16)
    assert (current.cohort_rank, current.cohort_size) == (1, 1)
    assert (defender.cohort_rank, defender.cohort_size) == (1, 1)
    assert (other_target.cohort_rank, other_target.cohort_size) == (1, 1)
    assert unavailable.available is False
    assert unavailable.cohort_rank is None
    assert unavailable.cohort_size == 0


def test_recommendation_order_uses_the_same_canonical_cohort_ranking(
    role_fit_session: Session,
) -> None:
    recommendations = get_role_recommendations(
        role_fit_session,
        904,
        PositionGroup.MID,
        limit=10,
    )
    assert recommendations.total == 3
    assert [item.player.player_id for item in recommendations.items] == [10, 11, 12]
    assert [item.rank for item in recommendations.items] == [1, 2, 3]
    for item in recommendations.items:
        fit = get_role_fit(role_fit_session, item.player.player_id)
        assert fit.cohort_rank == item.rank
        assert fit.cohort_size == recommendations.total
        assert fit.interpretation == FIT_INTERPRETATION
