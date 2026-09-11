"""Read-only API tests using deterministic SQLAlchemy fixture data."""

from collections.abc import Generator
from pathlib import Path
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
    AttackingAction,
    Pass,
    Player,
    PlayerArchetype,
    PlayerAttackingProfile,
    PlayerIntelligenceProfile,
    PlayerPercentile,
    PlayerProfile,
    PlayerRoleFit,
    PlayerShootingProfile,
    PlayerSimilarity,
    Shot,
    TeamRoleProfile,
    TeamStyleProfile,
)
from app.routers import model_info


def profile(player_id: int, **overrides: Any) -> PlayerProfile:
    values: dict[str, Any] = {
        "player_id": player_id,
        "passes_completed": 100,
        "actual_completion_rate": 0.8,
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
        "pressure_pass_rate": 0.2,
        "progressive_attempts": 20,
        "progressive_completed": 14,
        "progressive_actual_completion_rate": 0.7,
        "progressive_expected_completion_rate": 0.66,
        "progressive_completions_above_expected": 0.8,
        "progressive_above_expected_pp": 4.0,
        "progressive_pass_rate": 0.15,
        "long_pass_attempts": 15,
        "long_pass_completed": 8,
        "long_pass_actual_completion_rate": 0.53,
        "long_pass_expected_completion_rate": 0.5,
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


def pass_row(
    pass_index: int,
    player_id: int,
    *,
    completed: bool,
    expected_completion: float,
    under_pressure: bool,
    progressive: bool,
    match_id: int = 10,
) -> Pass:
    return Pass(
        pass_index=pass_index,
        match_id=match_id,
        player_id=player_id,
        team_id=100,
        position="Center Midfield",
        completed=completed,
        expected_completion=expected_completion,
        fold=1,
        start_x=20.0,
        start_y=30.0,
        end_x=40.0,
        end_y=35.0,
        pass_length=20.62,
        pass_angle=0.24,
        forward_distance=20.0,
        lateral_distance=5.0,
        distance_to_goal_before=100.5,
        distance_to_goal_after=80.15,
        distance_toward_goal=20.35,
        under_pressure=under_pressure,
        progressive=progressive,
        pass_height="Ground Pass",
        body_part="Right Foot",
        pass_type="Regular",
        start_zone="defensive_centre",
        end_zone="middle_centre",
    )


def team_role(position_group: str, contributors: int = 4) -> TeamRoleProfile:
    warning = (
        "Limited contributor diversity: this role profile is based on three contributing forwards."
        if position_group == "FWD" else "Observed role support."
    )
    return TeamRoleProfile(
        team_id=904, team_name="Bayer Leverkusen", position_group=position_group,
        methodology_version="V4.2", aggregation_method="pooled_events_actions",
        matches_observed=33 if position_group == "FWD" else 34,
        contributor_count=3 if position_group == "FWD" else contributors,
        contributors="[]", passes=1000, carries=800, actions=1800, shots=20,
        support_level="limited_contributor_diversity" if position_group == "FWD" else "established",
        support_message=warning, expected_completion_rate=0.86,
        pressure_pass_rate=0.15, progressive_pass_rate=0.12, long_pass_rate=0.10,
        positive_forward_distance_per_100_passes=600.0, carry_share_of_actions=0.45,
        expected_completion_rate_z=0.1, pressure_pass_rate_z=-0.2,
        progressive_pass_rate_z=0.3, long_pass_rate_z=-0.4,
        positive_forward_distance_per_100_passes_z=0.5,
        carry_share_of_actions_z=-0.1,
    )


def seed_database(session_factory: sessionmaker[Session]) -> None:
    with session_factory.begin() as session:
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
                    player_id=3,
                    player_name="Carol Forward",
                    team_id=200,
                    team_name="Beta United",
                    position="Center Forward",
                    position_group="FWD",
                    matches_observed=5,
                    pass_attempts=50,
                    overall_reliable=False,
                ),
                Player(
                    player_id=4,
                    player_name="Dana Midfielder",
                    team_id=300,
                    team_name="Gamma City",
                    position="Left Midfield",
                    position_group="MID",
                    matches_observed=10,
                    pass_attempts=130,
                    overall_reliable=True,
                ),
            ]
        )
        session.add(
            TeamStyleProfile(
                team_id=904, team_name="Bayer Leverkusen", methodology_version="V4.1",
                sample_scope="Observed team style across the 34-match product sample.",
                matches_observed=34, contributors=24, passes=24244, carries=20141,
                actions=43697, shots=623, expected_completion_rate=0.87,
                pressure_pass_rate=0.14, progressive_pass_rate=0.13,
                long_pass_rate=0.11, positive_forward_distance_per_100_passes=677.0,
                carry_share_of_actions=0.46, average_forward_distance=5.0,
                final_third_entries_per_100_passes=10.0, progressive_carry_rate=0.2,
                progressive_action_rate=0.15, pressure_action_rate=0.14,
                shots_per_match=18.3, xg_per_shot=0.11, xg_per_match=2.0,
                attacking_value_per_100_actions=0.5,
            )
        )
        session.add_all([team_role("DEF"), team_role("MID"), team_role("FWD")])
        session.add_all(
            [
                PlayerRoleFit(
                    target_team_id=904, player_id=1, target_team_name="Bayer Leverkusen",
                    player_name="Alice Playmaker", player_team_name="Alpha FC",
                    position="Center Midfield", position_group="MID",
                    is_target_team_player=True,
                    calculation_scope="leave_self_out_target_role", role_distance=0.62,
                    recommendation_rank=None, closest_feature_1="pressure_pass_rate",
                    closest_feature_2="progressive_pass_rate",
                    closest_feature_3="expected_completion_rate",
                    largest_difference="long_pass_rate", feature_gaps='{"long_pass_rate": 1.0}',
                    distance_contributions='{"long_pass_rate": 1.0}',
                    player_matches_observed=12, player_pass_attempts=150, player_carries=40,
                    sample_support="higher", sample_support_message="Observed across 12 matches.",
                    role_matches_observed=34, role_contributor_count=9, role_actions=12000,
                    role_support_message="Observed role support.", archetype_id="direct_progressor",
                    archetype_name="Direct Progressor", methodology_version="V4.3",
                ),
                PlayerRoleFit(
                    target_team_id=904, player_id=4, target_team_name="Bayer Leverkusen",
                    player_name="Dana Midfielder", player_team_name="Gamma City",
                    position="Left Midfield", position_group="MID", is_target_team_player=False,
                    calculation_scope="full_target_role", role_distance=0.39,
                    recommendation_rank=1, closest_feature_1="progressive_pass_rate",
                    closest_feature_2="carry_share_of_actions",
                    closest_feature_3="expected_completion_rate",
                    largest_difference="long_pass_rate", feature_gaps='{"long_pass_rate": 0.9}',
                    distance_contributions='{"long_pass_rate": 1.0}',
                    player_matches_observed=2, player_pass_attempts=130, player_carries=35,
                    sample_support="limited", sample_support_message="Limited sample: 2 matches.",
                    role_matches_observed=34, role_contributor_count=10, role_actions=19867,
                    role_support_message="Observed role support.", archetype_id=None,
                    archetype_name=None, methodology_version="V4.3",
                ),
            ]
        )
        session.add_all(
            [
                PlayerIntelligenceProfile(
                    player_id=player_id,
                    position_group=position_group,
                    matches_observed=10,
                )
                for player_id, position_group in ((1, "MID"), (2, "DEF"), (3, "FWD"), (4, "MID"))
            ]
        )
        session.add_all(
            [
                PlayerPercentile(
                    player_id=1,
                    metric_name=metric_name,
                    family=family,
                    raw_value=raw_value,
                    percentile=percentile,
                    peer_position_group="MID",
                    peer_count=12,
                    sample_count=150,
                    eligible=True,
                    eligibility_reason="eligible",
                )
                for metric_name, family, raw_value, percentile in (
                    ("progressive_pass_rate", "style", 0.15, 75.0),
                    ("pressure_pass_rate", "style", 0.2, 60.0),
                    ("completion_above_expected_pp", "performance", 2.0, 70.0),
                )
            ]
        )
        session.add(
            PlayerShootingProfile(
                player_id=1,
                shots=25,
                goals=4,
                total_xg=3.2,
                xg_per_shot=0.128,
                goals_minus_xg=0.8,
                goals_per_shot=0.16,
                matches_observed=10,
                shooting_reliable=True,
            )
        )
        session.add(
            Shot(
                shot_id="00000000-0000-0000-0000-000000000001",
                match_id=10,
                player_id=1,
                team_id=100,
                period=1,
                minute=8,
                second=12,
                start_x=108.0,
                start_y=40.0,
                distance=12.0,
                angle=0.64,
                goal=True,
                expected_goal=0.31,
                body_part="Right Foot",
                shot_type="Open Play",
                technique="Normal",
                play_pattern="Regular Play",
                under_pressure=False,
                first_time=False,
                one_on_one=False,
                open_goal=False,
                penalty=False,
                penalty_shootout=False,
                model_eligible=True,
            )
        )
        session.add_all(
            [
                profile(1),
                profile(
                    2,
                    completion_above_expected_pp=4.0,
                    pressure_above_expected_pp=10.0,
                    pressure_reliable=False,
                ),
                profile(
                    3,
                    pressure_attempts=0,
                    pressure_completed=0,
                    pressure_actual_completion_rate=None,
                    pressure_expected_completion_rate=None,
                    pressure_completions_above_expected=None,
                    pressure_above_expected_pp=None,
                    pressure_pass_rate=0.0,
                    pressure_reliable=False,
                    progressive_reliable=False,
                    long_pass_reliable=False,
                ),
                profile(4, completion_above_expected_pp=1.0, pressure_above_expected_pp=3.0),
            ]
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
                query_matches_observed=5,
                candidate_matches_observed=2,
                pair_support_matches=2,
                sample_support="limited",
                sample_support_explanation="Limited sample for one profile.",
                methodology_version="V3.3B",
                distance_contributions="{}",
            )
        )
        session.add_all(
            [
                pass_row(
                    0,
                    1,
                    completed=True,
                    expected_completion=0.2,
                    under_pressure=True,
                    progressive=False,
                ),
                pass_row(
                    1,
                    1,
                    completed=False,
                    expected_completion=0.5,
                    under_pressure=False,
                    progressive=True,
                ),
                pass_row(
                    2,
                    1,
                    completed=True,
                    expected_completion=0.8,
                    under_pressure=True,
                    progressive=True,
                    match_id=11,
                ),
                pass_row(
                    3,
                    1,
                    completed=True,
                    expected_completion=0.9,
                    under_pressure=False,
                    progressive=False,
                    match_id=11,
                ),
                pass_row(
                    4,
                    2,
                    completed=True,
                    expected_completion=0.7,
                    under_pressure=False,
                    progressive=False,
                ),
            ]
        )
        session.add(
            PlayerAttackingProfile(
                player_id=1,
                matches_observed=10,
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
            PlayerArchetype(
                player_id=1,
                archetype_id="direct_progressor",
                archetype_name="Direct Progressor",
                raw_cluster_id=0,
                position_group="MID",
                centroid_distance=0.7,
                second_centroid_distance=2.1,
                separation_margin=2 / 3,
                eligible=True,
                model_version="V3.2C",
                expected_completion_rate_position_z=-0.5,
                pressure_pass_rate_position_z=-0.1,
                progressive_pass_rate_position_z=0.8,
                long_pass_rate_position_z=0.6,
                positive_forward_distance_per_100_passes_position_z=0.7,
                carry_share_of_actions_position_z=-0.3,
            )
        )
        session.add(
            AttackingAction(
                action_id="00000000-0000-0000-0000-000000000010",
                pass_index=0,
                match_id=10,
                possession_id=5,
                event_index=20,
                player_id=1,
                team_id=100,
                action_type="Pass",
                start_x=20.0,
                start_y=30.0,
                end_x=40.0,
                end_y=35.0,
                state_value_before=0.02,
                state_value_after=0.08,
                attacking_value=0.06,
                success=True,
                under_pressure=True,
                progressive=False,
                expected_completion=0.2,
                pass_risk=0.8,
                risk_reward_category="difficult_high_value",
                fold=1,
            )
        )


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    test_engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )
    Base.metadata.create_all(test_engine)
    seed_database(testing_session)

    def override_get_db() -> Generator[Session, None, None]:
        session = testing_session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(test_engine)
    test_engine.dispose()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_uses_current_brand(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"] == {
        "title": "FootyScout API",
        "description": "Football scouting and analytics API for FootyScout.",
        "version": "0.1.0",
    }


def test_cors_allows_only_configured_frontend(client: TestClient) -> None:
    allowed = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    denied = client.options(
        "/health",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in denied.headers


def test_player_listing_and_pagination(client: TestClient) -> None:
    response = client.get("/api/players", params={"limit": 2, "offset": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert body["limit"] == 2
    assert body["offset"] == 1
    assert [item["player_name"] for item in body["items"]] == [
        "Bob Defender",
        "Carol Forward",
    ]


def test_player_search(client: TestClient) -> None:
    response = client.get("/api/players", params={"search": "pLaYmAk"})
    assert response.status_code == 200
    assert [item["player_id"] for item in response.json()["items"]] == [1]


def test_player_team_position_and_attempt_filters(client: TestClient) -> None:
    team = client.get("/api/players", params={"team": "alpha fc"}).json()
    position = client.get("/api/players", params={"position_group": "MID"}).json()
    attempts = client.get("/api/players", params={"min_pass_attempts": 130}).json()
    assert {item["player_id"] for item in team["items"]} == {1, 2}
    assert {item["player_id"] for item in position["items"]} == {1, 4}
    assert {item["player_id"] for item in attempts["items"]} == {1, 4}


def test_player_safe_sorting_and_invalid_sort(client: TestClient) -> None:
    response = client.get(
        "/api/players",
        params={"sort_by": "pass_attempts", "sort_order": "desc"},
    )
    assert [item["player_id"] for item in response.json()["items"]] == [1, 4, 2, 3]
    invalid = client.get("/api/players", params={"sort_by": "drop_table"})
    assert invalid.status_code == 422


def test_invalid_player_filters_and_pagination(client: TestClient) -> None:
    assert client.get("/api/players", params={"position_group": "WING"}).status_code == 422
    assert client.get("/api/players", params={"limit": 0}).status_code == 422
    assert client.get("/api/players", params={"offset": -1}).status_code == 422


def test_player_profile_and_unknown_player(client: TestClient) -> None:
    response = client.get("/api/players/1")
    assert response.status_code == 200
    body = response.json()
    assert body["player_name"] == "Alice Playmaker"
    assert body["passes_completed"] == 100
    assert body["pressure_reliable"] is True
    assert client.get("/api/players/999").status_code == 404


def test_similar_player_and_empty_recommendations(client: TestClient) -> None:
    response = client.get("/api/players/1/similar")
    assert response.status_code == 200
    assert response.json()["limit"] == 6
    assert response.json()["methodology_version"] == "V3.3B"
    assert response.json()["items"][0] == {
        "similar_player_id": 4,
        "similar_player_name": "Dana Midfielder",
        "similar_team_name": "Gamma City",
        "similar_position": "Left Midfield",
        "similar_position_group": "MID",
        "rank": 1,
        "rms_distance": 0.2,
        "similarity_score": 90.0,
        "same_position_group": True,
        "closest_feature_1": "pressure_pass_rate",
        "closest_feature_2": "progressive_pass_rate",
        "closest_feature_3": "expected_completion_rate",
        "closest_style_dimensions": [
            "pressure_pass_rate",
            "progressive_pass_rate",
            "expected_completion_rate",
        ],
        "query_matches_observed": 5,
        "candidate_matches_observed": 2,
        "pair_support_matches": 2,
        "sample_support": "limited",
        "sample_support_explanation": "Limited sample for one profile.",
        "methodology_version": "V3.3B",
    }
    empty = client.get("/api/players/3/similar")
    assert empty.status_code == 200
    assert empty.json()["total"] == 0
    assert empty.json()["items"] == []
    assert empty.json()["available"] is False
    assert "50 passes" in empty.json()["unavailable_reason"]
    assert client.get("/api/players/999/similar").status_code == 404


def test_pass_retrieval_and_pagination(client: TestClient) -> None:
    response = client.get("/api/players/1/passes", params={"limit": 2, "offset": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert [item["pass_index"] for item in body["items"]] == [1, 2]
    assert set(body["items"][0]) == {
        "pass_index",
        "match_id",
        "player_id",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "pass_length",
        "pass_angle",
        "forward_distance",
        "lateral_distance",
        "distance_to_goal_before",
        "distance_to_goal_after",
        "distance_toward_goal",
        "completed",
        "expected_completion",
        "under_pressure",
        "progressive",
        "pass_height",
        "body_part",
        "pass_type",
        "start_zone",
        "end_zone",
            "fold",
            "attacking_value",
            "state_value_before",
            "state_value_after",
            "pass_risk",
            "risk_reward_category",
        }


@pytest.mark.parametrize(
    ("params", "expected_indices"),
    [
        ({"completed": False}, [1]),
        ({"under_pressure": True}, [0, 2]),
        ({"progressive": True}, [1, 2]),
        ({"match_id": 11}, [2, 3]),
        ({"min_expected_completion": 0.4, "max_expected_completion": 0.8}, [1, 2]),
    ],
)
def test_pass_filters(
    client: TestClient,
    params: dict[str, Any],
    expected_indices: list[int],
) -> None:
    response = client.get("/api/players/1/passes", params=params)
    assert response.status_code == 200
    assert [item["pass_index"] for item in response.json()["items"]] == expected_indices


def test_invalid_pass_probability_range_and_unknown_player(client: TestClient) -> None:
    reversed_range = client.get(
        "/api/players/1/passes",
        params={"min_expected_completion": 0.8, "max_expected_completion": 0.2},
    )
    outside_range = client.get(
        "/api/players/1/passes",
        params={"min_expected_completion": 1.1},
    )
    assert reversed_range.status_code == 422
    assert outside_range.status_code == 422
    assert client.get("/api/players/999/passes").status_code == 404


@pytest.mark.artifacts
def test_valid_comparison(client: TestClient) -> None:
    response = client.get("/api/compare", params={"player_ids": "2,1"})
    assert response.status_code == 200
    body = response.json()
    assert body["comparison"] == {
        "player_ids": [2, 1],
        "same_position_group": False,
    }
    assert [player["player_id"] for player in body["players"]] == [2, 1]
    assert body["attacking"][0] is None
    assert body["attacking"][1]["player_id"] == 1


def test_player_shooting_profile_and_paginated_shots(client: TestClient) -> None:
    profile = client.get("/api/players/1/shooting")
    assert profile.status_code == 200
    assert profile.json()["shots"] == 25
    assert profile.json()["shooting_reliable"] is True

    shots = client.get("/api/players/1/shots", params={"limit": 1, "offset": 0})
    assert shots.status_code == 200
    assert shots.json()["total"] == 1
    assert shots.json()["items"][0]["shot_id"].endswith("1")
    assert shots.json()["items"][0]["expected_goal"] == pytest.approx(0.31)
    assert client.get("/api/players/2/shooting").status_code == 404


def test_attacking_profile_actions_and_filters(client: TestClient) -> None:
    profile_response = client.get("/api/players/1/attacking")
    assert profile_response.status_code == 200
    assert profile_response.json()["attacking_value_per_100_actions"] == pytest.approx(0.5)
    assert profile_response.json()["carry_value_reliable"] is True

    actions_response = client.get(
        "/api/players/1/actions",
        params={
            "action_type": "Pass",
            "match_id": 10,
            "positive_only": True,
            "under_pressure": True,
            "progressive": False,
            "min_value": 0.05,
            "max_value": 0.07,
        },
    )
    assert actions_response.status_code == 200
    body = actions_response.json()
    assert body["total"] == 1
    assert body["items"][0]["pass_index"] == 0
    assert body["items"][0]["attacking_value"] == pytest.approx(0.06)

    assert client.get("/api/players/2/attacking").status_code == 404
    assert client.get("/api/players/999/actions").status_code == 404
    assert client.get(
        "/api/players/1/actions", params={"min_value": 1, "max_value": 0}
    ).status_code == 422


def test_pass_endpoint_exposes_linked_value_without_changing_xpass(client: TestClient) -> None:
    item = client.get("/api/players/1/passes", params={"limit": 1}).json()["items"][0]
    assert item["expected_completion"] == pytest.approx(0.2)
    assert item["attacking_value"] == pytest.approx(0.06)
    assert item["state_value_before"] == pytest.approx(0.02)
    assert item["state_value_after"] == pytest.approx(0.08)
    assert item["pass_risk"] == pytest.approx(0.8)


@pytest.mark.artifacts
def test_player_intelligence_serializes_grouped_percentiles(client: TestClient) -> None:
    response = client.get("/api/players/1/intelligence")
    assert response.status_code == 200
    body = response.json()
    assert body["player"]["player_id"] == 1
    assert body["position_group"] == "MID"
    assert [metric["metric_name"] for metric in body["style_metrics"]] == [
        "pressure_pass_rate", "progressive_pass_rate"
    ]
    assert body["performance_metrics"][0]["percentile"] == 70.0
    assert body["style_metrics"][0]["peer_count"] == 12
    assert len(body["radar_metrics"]) == 3
    assert "not ratings" in body["percentile_context"]
    assert body["archetype"]["id"] == "direct_progressor"
    assert body["archetype"]["name"] == "Direct Progressor"
    assert body["archetype"]["eligible"] is True
    assert body["archetype"]["separation_margin"] == pytest.approx(2 / 3)
    assert len(body["archetype"]["style_dimensions"]) == 6
    assert "not a probability" in body["archetype"]["separation_interpretation"]
    assert client.get("/api/players/999/intelligence").status_code == 404


@pytest.mark.artifacts
def test_comparison_includes_position_relative_intelligence(client: TestClient) -> None:
    body = client.get("/api/compare", params={"player_ids": "1,2"}).json()
    assert len(body["intelligence"]) == 2
    assert body["intelligence"][0]["position_group"] == "MID"
    assert body["intelligence"][1]["position_group"] == "DEF"
    assert body["intelligence"][0]["archetype"]["name"] == "Direct Progressor"
    assert body["intelligence"][1]["archetype"]["id"] is None
    assert body["intelligence"][1]["archetype"]["eligible"] is False


@pytest.mark.artifacts
def test_archetype_catalogue_uses_persisted_assignments(client: TestClient) -> None:
    response = client.get("/api/archetypes")
    assert response.status_code == 200
    body = response.json()
    assert body["methodology_version"] == "V3.2C"
    assert "not quality" in body["purpose"]
    assert [item["id"] for item in body["definitions"]] == [
        "direct_progressor",
        "safe_circulator",
    ]
    direct = body["definitions"][0]
    assert direct["player_count"] == 1
    assert direct["position_composition"]["MID"]["count"] == 1
    assert direct["representative_players"][0]["player_name"] == "Alice Playmaker"
    assert body["definitions"][1]["player_count"] == 0
    assert body["methodology"]["k"] == 2
    assert len(body["methodology"]["feature_order"]) == 6


def test_team_intelligence_and_role_endpoints(client: TestClient) -> None:
    intelligence = client.get("/api/teams/904/intelligence")
    assert intelligence.status_code == 200
    body = intelligence.json()
    assert body["team"]["matches_observed"] == 34
    assert body["team"]["metrics"]["expected_completion_rate"] == pytest.approx(0.87)
    assert [role["position_group"] for role in body["roles"]] == ["DEF", "FWD", "MID"]
    assert "observed playing style" in body["fit_definition"]

    fwd = client.get("/api/teams/904/roles/FWD")
    assert fwd.status_code == 200
    assert fwd.json()["contributor_count"] == 3
    assert "Limited contributor diversity" in fwd.json()["support_message"]
    assert len(fwd.json()["dimensions"]) == 6
    assert client.get("/api/teams/904/roles/GK").status_code == 404
    unavailable = client.get("/api/teams/169/intelligence")
    assert unavailable.status_code == 404
    assert "complete 34-match" in unavailable.json()["detail"]


def test_external_and_leave_self_out_role_fit(client: TestClient) -> None:
    external = client.get("/api/players/4/role-fit").json()
    assert external["available"] is True
    assert external["calculation_scope"] == "full_target_role"
    assert external["role_distance"] == pytest.approx(0.39)
    assert external["sample_support"] == "limited"
    assert external["closest_dimensions"][0] == "progressive_pass_rate"

    current = client.get("/api/players/1/role-fit").json()
    assert current["is_target_team_player"] is True
    assert current["calculation_scope"] == "leave_self_out_target_role"
    assert current["role_distance"] == pytest.approx(0.62)

    unavailable = client.get("/api/players/2/role-fit").json()
    assert unavailable["available"] is False
    assert "50 passes and 29 carries" in unavailable["unavailable_reason"]


def test_scouting_recommendations_use_pure_distance_order(client: TestClient) -> None:
    response = client.get("/api/teams/904/roles/MID/recommendations")
    assert response.status_code == 200
    body = response.json()
    assert body["methodology_version"] == "V4.4"
    assert body["total"] == 1
    assert body["items"][0]["player"]["player_id"] == 4
    assert body["items"][0]["role_distance"] == pytest.approx(0.39)
    assert body["items"][0]["sample_support"] == "limited"
    assert "not predictions" in body["disclaimer"]


@pytest.mark.parametrize("player_ids", ["1", "1,2,3", "1,1", "1,nope", "-1,2"])
def test_malformed_comparison(client: TestClient, player_ids: str) -> None:
    assert client.get("/api/compare", params={"player_ids": player_ids}).status_code == 422


def test_missing_and_unknown_comparison_ids(client: TestClient) -> None:
    assert client.get("/api/compare").status_code == 422
    assert client.get("/api/compare", params={"player_ids": "1,999"}).status_code == 404


def test_overall_leaderboard(client: TestClient) -> None:
    response = client.get("/api/leaderboard")
    assert response.status_code == 200
    body = response.json()
    assert body["metric"] == "completion_above_expected_pp"
    assert [item["player_id"] for item in body["items"]] == [2, 1, 4]
    assert [item["rank"] for item in body["items"]] == [1, 2, 3]
    assert 3 not in {item["player_id"] for item in body["items"]}


def test_subset_leaderboard_applies_reliability(client: TestClient) -> None:
    response = client.get(
        "/api/leaderboard",
        params={"metric": "pressure_above_expected_pp"},
    )
    assert response.status_code == 200
    assert [item["player_id"] for item in response.json()["items"]] == [1, 4]
    assert 2 not in {item["player_id"] for item in response.json()["items"]}


def test_leaderboard_filters_and_invalid_metric(client: TestClient) -> None:
    filtered = client.get(
        "/api/leaderboard",
        params={"position_group": "MID", "team": "Alpha FC", "limit": 1},
    )
    assert [item["player_id"] for item in filtered.json()["items"]] == [1]
    invalid = client.get("/api/leaderboard", params={"metric": "player_name"})
    assert invalid.status_code == 422


def test_attacking_leaderboard_uses_metric_specific_reliability(client: TestClient) -> None:
    response = client.get(
        "/api/leaderboard",
        params={"metric": "attacking_value_per_100_actions"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["minimum_pass_attempts"] == 0
    assert [item["player_id"] for item in body["items"]] == [1]
    assert body["items"][0]["relevant_attempts"] == 100


@pytest.mark.artifacts
def test_model_metadata_is_curated_and_methodologically_explicit(client: TestClient) -> None:
    response = client.get("/api/model")
    assert response.status_code == 200
    body = response.json()
    assert body["selected_model"] == "xgboost"
    assert body["dataset"]["pass_count"] == 39_214
    assert body["oof_fold_count"] == 5
    assert body["selection"]["test_metrics_used"] is False
    assert body["models_evaluated"] == ["logistic_regression", "pytorch_mlp", "xgboost"]
    assert body["xgboost"]["version"] == "3.4.1"
    assert body["xgboost"]["best_iteration"] == 314
    assert body["xgboost"]["feature_importance_type"] == "gain"
    assert body["validation_metrics"]["xgboost_effective"]["log_loss"] < body[
        "validation_metrics"
    ]["pytorch_mlp_effective"]["log_loss"]
    assert "validation data" in body["methodology"]["model_selection"]
    assert "not used for model selection" in body["methodology"]["test_set"]
    assert "out-of-fold" in body["methodology"]["player_profiles"]
    assert "artifacts" not in body
    assert "path" not in body["dataset"]


@pytest.mark.artifacts
def test_xg_model_metadata_is_curated_and_grouped(client: TestClient) -> None:
    response = client.get("/api/models/xg")
    assert response.status_code == 200
    body = response.json()
    assert body["selected_model"] == "xgboost"
    assert body["dataset"]["shots"] == 5_784
    assert body["dataset"]["eligible_non_penalty_shots"] == 5_545
    assert body["selection"]["test_metrics_used"] is False
    assert body["oof"]["prediction_count"] == 5_545
    assert body["oof"]["group_integrity"] is True
    assert "train_match_ids" not in body["split_methodology"]
    assert "path" not in body


@pytest.mark.artifacts
def test_action_value_metadata_is_curated_grouped_and_non_causal(client: TestClient) -> None:
    response = client.get("/api/models/action-value")
    assert response.status_code == 200
    body = response.json()
    assert body["selection"]["test_metrics_used"] is False
    assert body["oof"]["prediction_count"] == 667_062
    assert body["oof"]["group_integrity"] is True
    assert body["target_distribution"]["total_possessions"] == 38_419
    assert body["training_corpus"]["matches"] == 233
    assert body["training_corpus"]["events"] == 823_553
    assert body["nested_cross_fitting"]["heldout_outcomes_used_for_training"] is False
    assert body["nested_cross_fitting"]["upstream_model_selection_changed"] is False
    assert "train_match_ids" not in body["split_methodology"]
    assert any("not causal" in limitation for limitation in body["limitations"])


def test_model_metadata_errors_are_safe(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(model_info, "MODEL_METADATA_PATH", missing)
    unavailable = client.get("/api/model")
    assert unavailable.status_code == 503
    assert unavailable.json() == {"detail": "Model metadata is currently unavailable."}

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    monkeypatch.setattr(model_info, "MODEL_METADATA_PATH", malformed)
    invalid = client.get("/api/model")
    assert invalid.status_code == 500
    assert invalid.json() == {"detail": "Model metadata is malformed."}


def test_meta_endpoint(client: TestClient) -> None:
    response = client.get("/api/meta")
    assert response.status_code == 200
    assert response.json() == {
        "teams": ["Alpha FC", "Beta United", "Gamma City"],
        "position_groups": ["DEF", "MID", "FWD"],
        "player_count": 4,
        "reliable_player_count": 3,
    }


def test_openapi_generation_and_similarity_score_description(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/api/players/{player_id}/passes" in schema["paths"]
    assert "/api/players/{player_id}/shots" in schema["paths"]
    assert "/api/players/{player_id}/shooting" in schema["paths"]
    assert "/api/models/xg" in schema["paths"]
    assert "/api/models/action-value" in schema["paths"]
    assert "/api/players/{player_id}/attacking" in schema["paths"]
    assert "/api/players/{player_id}/actions" in schema["paths"]
    assert "/api/players/{player_id}/intelligence" in schema["paths"]
    assert "/api/archetypes" in schema["paths"]
    assert "/api/teams/{team_id}/intelligence" in schema["paths"]
    assert "/api/players/{player_id}/role-fit" in schema["paths"]
    assert "/api/teams/{team_id}/roles/{position_group}/recommendations" in schema["paths"]
    similarity_score = schema["components"]["schemas"]["SimilarPlayerResponse"][
        "properties"
    ]["similarity_score"]
    assert "not a probability" in similarity_score["description"]
    assert client.get("/docs").status_code == 200
