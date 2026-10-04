"""Isolated deterministic database fixtures for AI Scout contract tests."""

from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.dependencies import get_db
from app.main import app
from app.models import (
    Player,
    PlayerAttackingProfile,
    PlayerProfile,
    PlayerRoleFit,
    PlayerSimilarity,
    TeamRoleProfile,
    TeamStyleProfile,
)


def _profile(player_id: int, **overrides: Any) -> PlayerProfile:
    values: dict[str, Any] = {
        "player_id": player_id,
        "passes_completed": 100,
        "actual_completion_rate": 0.80,
        "expected_completions": 97.0,
        "expected_completion_rate": 0.78,
        "completions_above_expected": 3.0,
        "completion_above_expected_pp": 2.0,
        "pressure_attempts": 30,
        "pressure_completed": 22,
        "pressure_actual_completion_rate": 0.73,
        "pressure_expected_completion_rate": 0.68,
        "pressure_completions_above_expected": 1.5,
        "pressure_above_expected_pp": 5.0,
        "pressure_pass_rate": 0.20,
        "progressive_attempts": 20,
        "progressive_completed": 14,
        "progressive_actual_completion_rate": 0.70,
        "progressive_expected_completion_rate": 0.66,
        "progressive_completions_above_expected": 0.8,
        "progressive_above_expected_pp": 4.0,
        "progressive_pass_rate": 0.15,
        "long_pass_attempts": 15,
        "long_pass_completed": 8,
        "long_pass_actual_completion_rate": 0.53,
        "long_pass_expected_completion_rate": 0.50,
        "long_pass_completions_above_expected": 0.45,
        "long_pass_above_expected_pp": 3.0,
        "average_forward_distance": 7.0,
        "net_forward_distance_per_100_passes": 700.0,
        "positive_forward_distance_per_100_passes": 850.0,
        "final_third_entries": 18,
        "final_third_entries_per_100_passes": 12.0,
        "pressure_reliable": True,
        "progressive_reliable": True,
        "long_pass_reliable": True,
    }
    values.update(overrides)
    return PlayerProfile(**values)


def _team_role(position_group: str) -> TeamRoleProfile:
    return TeamRoleProfile(
        team_id=904,
        team_name="Bayer Leverkusen",
        position_group=position_group,
        methodology_version="V4.2",
        aggregation_method="pooled_events_actions",
        matches_observed=34,
        contributor_count=4,
        contributors="[]",
        passes=1000,
        carries=800,
        actions=1800,
        shots=20,
        support_level="established",
        support_message="Observed role support.",
        expected_completion_rate=0.86,
        pressure_pass_rate=0.15,
        progressive_pass_rate=0.12,
        long_pass_rate=0.10,
        positive_forward_distance_per_100_passes=600.0,
        carry_share_of_actions=0.45,
        expected_completion_rate_z=0.1,
        pressure_pass_rate_z=-0.2,
        progressive_pass_rate_z=0.3,
        long_pass_rate_z=-0.4,
        positive_forward_distance_per_100_passes_z=0.5,
        carry_share_of_actions_z=-0.1,
    )


def _seed(session: Session) -> None:
    session.add_all(
        [
            Player(
                player_id=1,
                player_name="Alice Playmaker",
                team_id=100,
                team_name="Alpha FC",
                position="Center Midfield",
                position_group="MID",
                matches_observed=12,
                pass_attempts=150,
                overall_reliable=True,
            ),
            Player(
                player_id=2,
                player_name="Bob Defender",
                team_id=100,
                team_name="Alpha FC",
                position="Center Back",
                position_group="DEF",
                matches_observed=11,
                pass_attempts=120,
                overall_reliable=True,
            ),
            Player(
                player_id=4,
                player_name="Dana Midfielder",
                team_id=300,
                team_name="Gamma City",
                position="Left Midfield",
                position_group="MID",
                matches_observed=2,
                pass_attempts=130,
                overall_reliable=True,
            ),
            Player(
                player_id=5,
                player_name="Alice Runner",
                team_id=400,
                team_name="Delta Town",
                position="Right Midfield",
                position_group="MID",
                matches_observed=3,
                pass_attempts=40,
                overall_reliable=False,
            ),
            Player(
                player_id=6,
                player_name="Gina Keeper",
                team_id=500,
                team_name="Epsilon FC",
                position="Goalkeeper",
                position_group="GK",
                matches_observed=8,
                pass_attempts=90,
                overall_reliable=False,
            ),
            Player(
                player_id=7,
                player_name="Alex Williams",
                team_id=600,
                team_name="Zeta United",
                position="Center Midfield",
                position_group="MID",
                matches_observed=4,
                pass_attempts=110,
                overall_reliable=True,
            ),
            Player(
                player_id=8,
                player_name="Jordan Williams",
                team_id=700,
                team_name="Eta Athletic",
                position="Center Midfield",
                position_group="MID",
                matches_observed=5,
                pass_attempts=115,
                overall_reliable=True,
            ),
        ]
    )
    session.add_all(
        [
            _profile(1),
            _profile(2, completion_above_expected_pp=4.0),
            _profile(4, completion_above_expected_pp=1.0),
            _profile(5, completion_above_expected_pp=-1.0),
            _profile(6, completion_above_expected_pp=0.0),
            _profile(7, completion_above_expected_pp=0.5),
            _profile(8, completion_above_expected_pp=0.75),
        ]
    )
    session.add(
        PlayerAttackingProfile(
            player_id=1,
            matches_observed=12,
            actions=100,
            passes=70,
            carries=30,
            total_attacking_value=0.5,
            attacking_value_per_100_actions=0.5,
            total_pass_value=0.4,
            pass_value_per_100_passes=0.571,
            total_carry_value=0.1,
            carry_value_per_100_carries=0.333,
            positive_value_actions=55,
            positive_value_action_rate=0.55,
            progressive_action_value=0.25,
            progressive_value_per_100_actions=0.25,
            pressure_action_value=0.15,
            pressure_value_per_100_actions=0.15,
            attacking_value_reliable=True,
            pass_value_reliable=True,
            carry_value_reliable=True,
        )
    )
    session.add(
        PlayerSimilarity(
            player_id=1,
            similar_player_id=4,
            rank=1,
            rms_distance=0.2,
            similarity_score=90.0,
            same_position_group=True,
            position_group="MID",
            similar_position_group="MID",
            closest_feature_1="pressure_pass_rate",
            closest_feature_2="progressive_pass_rate",
            closest_feature_3="expected_completion_rate",
            query_matches_observed=12,
            candidate_matches_observed=2,
            pair_support_matches=2,
            sample_support="limited",
            sample_support_explanation="Limited sample for one profile.",
            methodology_version="V3.3B",
            distance_contributions="{}",
        )
    )
    session.add(
        TeamStyleProfile(
            team_id=904,
            team_name="Bayer Leverkusen",
            methodology_version="V4.1",
            sample_scope="Observed team style across the 34-match product sample.",
            matches_observed=34,
            contributors=24,
            passes=24244,
            carries=20141,
            actions=43697,
            shots=623,
            expected_completion_rate=0.87,
            pressure_pass_rate=0.14,
            progressive_pass_rate=0.13,
            long_pass_rate=0.11,
            positive_forward_distance_per_100_passes=677.0,
            carry_share_of_actions=0.46,
            average_forward_distance=5.0,
            final_third_entries_per_100_passes=10.0,
            progressive_carry_rate=0.2,
            progressive_action_rate=0.15,
            pressure_action_rate=0.14,
            shots_per_match=18.3,
            xg_per_shot=0.11,
            xg_per_match=2.0,
            attacking_value_per_100_actions=0.5,
        )
    )
    session.add_all([_team_role(group) for group in ("DEF", "MID", "FWD")])
    session.add_all(
        [
            PlayerRoleFit(
                target_team_id=904,
                player_id=1,
                target_team_name="Bayer Leverkusen",
                player_name="Alice Playmaker",
                player_team_name="Alpha FC",
                position="Center Midfield",
                position_group="MID",
                is_target_team_player=True,
                calculation_scope="leave_self_out_target_role",
                role_distance=0.62,
                recommendation_rank=None,
                closest_feature_1="pressure_pass_rate",
                closest_feature_2="progressive_pass_rate",
                closest_feature_3="expected_completion_rate",
                largest_difference="long_pass_rate",
                feature_gaps='{"long_pass_rate": 1.0}',
                distance_contributions='{"long_pass_rate": 1.0}',
                player_matches_observed=12,
                player_pass_attempts=150,
                player_carries=40,
                sample_support="higher",
                sample_support_message="Observed across 12 matches.",
                role_matches_observed=34,
                role_contributor_count=9,
                role_actions=12000,
                role_support_message="Observed role support.",
                archetype_id="direct_progressor",
                archetype_name="Direct Progressor",
                methodology_version="V4.3",
            ),
            PlayerRoleFit(
                target_team_id=904,
                player_id=4,
                target_team_name="Bayer Leverkusen",
                player_name="Dana Midfielder",
                player_team_name="Gamma City",
                position="Left Midfield",
                position_group="MID",
                is_target_team_player=False,
                calculation_scope="full_target_role",
                role_distance=0.39,
                recommendation_rank=1,
                closest_feature_1="progressive_pass_rate",
                closest_feature_2="carry_share_of_actions",
                closest_feature_3="expected_completion_rate",
                largest_difference="long_pass_rate",
                feature_gaps='{"long_pass_rate": 0.9}',
                distance_contributions='{"long_pass_rate": 1.0}',
                player_matches_observed=2,
                player_pass_attempts=130,
                player_carries=35,
                sample_support="limited",
                sample_support_message="Limited sample: 2 matches.",
                role_matches_observed=34,
                role_contributor_count=10,
                role_actions=19867,
                role_support_message="Observed role support.",
                archetype_id=None,
                archetype_name=None,
                methodology_version="V4.3",
            ),
        ]
    )
    session.commit()


@pytest.fixture(scope="module")
def ai_environment() -> Generator[
    tuple[TestClient, sessionmaker[Session]],
    None,
    None,
]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    with session_factory() as session:
        _seed(session)

    def override_get_db() -> Generator[Session, None, None]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client, session_factory
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="module")
def ai_client(
    ai_environment: tuple[TestClient, sessionmaker[Session]],
) -> TestClient:
    return ai_environment[0]


@pytest.fixture
def ai_session(
    ai_environment: tuple[TestClient, sessionmaker[Session]],
) -> Generator[Session, None, None]:
    with ai_environment[1]() as session:
        yield session
