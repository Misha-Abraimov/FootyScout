import pandas as pd

from analytics.player_shooting_profiles import (
    MIN_RELIABLE_SHOTS,
    build_player_shooting_profiles,
)


def test_player_shooting_aggregates_and_reliability() -> None:
    shots = pd.DataFrame(
        [
            {
                "shot_id": f"shot-{index}",
                "player_id": 1,
                "match_id": index % 3,
                "goal": int(index < 3),
                "expected_goal": 0.1,
                "model_eligible": True,
            }
            for index in range(MIN_RELIABLE_SHOTS)
        ]
        + [
            {
                "shot_id": "penalty",
                "player_id": 1,
                "match_id": 4,
                "goal": 1,
                "expected_goal": None,
                "model_eligible": False,
            }
        ]
    )
    profile = build_player_shooting_profiles(shots).iloc[0]
    assert profile["shots"] == MIN_RELIABLE_SHOTS
    assert profile["goals"] == 3
    assert profile["total_xg"] == 2.0
    assert profile["goals_minus_xg"] == 1.0
    assert profile["shooting_reliable"]


def test_low_sample_players_are_not_discarded() -> None:
    shots = pd.DataFrame(
        [{"shot_id": "one", "player_id": 7, "match_id": 1, "goal": 0, "expected_goal": 0.2, "model_eligible": True}]
    )
    profile = build_player_shooting_profiles(shots)
    assert len(profile) == 1
    assert not profile.iloc[0]["shooting_reliable"]
