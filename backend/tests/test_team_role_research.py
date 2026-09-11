from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.player_similarity import SIMILARITY_FEATURE_COLUMNS
from analytics.player_style_stability import build_match_chronology
from analytics.team_role_research import (
    FIT_FEATURES,
    _aggregate_profiles,
    _profile_for_subset,
    build_team_match_profiles,
    deterministic_match_resamples,
    normalize_profile,
    rank_external_candidates,
    validate_fit_contract,
    vector_distance,
)


def _passes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": 1, "team_id": 10, "team_name": "Target", "player_id": 1,
                "position_group": "MID", "expected_completion": 0.8,
                "under_pressure": True, "progressive": True, "pass_length": 35.0,
                "forward_distance": 20.0, "start_x": 70.0, "end_x": 90.0,
            },
            {
                "match_id": 1, "team_id": 10, "team_name": "Target", "player_id": 2,
                "position_group": "DEF", "expected_completion": 0.9,
                "under_pressure": False, "progressive": False, "pass_length": 10.0,
                "forward_distance": -5.0, "start_x": 20.0, "end_x": 15.0,
            },
            {
                "match_id": 1, "team_id": 20, "team_name": "Opponent", "player_id": 3,
                "position_group": "MID", "expected_completion": 0.7,
                "under_pressure": False, "progressive": True, "pass_length": 20.0,
                "forward_distance": 10.0, "start_x": 60.0, "end_x": 70.0,
            },
        ]
    )


def _actions() -> pd.DataFrame:
    rows = [
        ("a", 1, 10, 1, "MID", "Pass", True, True, 0.1),
        ("b", 1, 10, 1, "MID", "Carry", True, False, 0.2),
        ("c", 1, 10, 2, "DEF", "Pass", False, False, -0.1),
        ("d", 1, 20, 3, "MID", "Pass", True, False, 0.0),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "action_id", "match_id", "team_id", "player_id", "position_group",
            "action_type", "progressive", "under_pressure", "attacking_value",
        ],
    )


def _shots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"shot_id": "s1", "match_id": 1, "team_id": 10, "team_name": "Target",
             "player_id": 1, "position_group": "MID", "expected_goal": 0.2, "goal": True},
            {"shot_id": "s2", "match_id": 1, "team_id": 20, "team_name": "Opponent",
             "player_id": 3, "position_group": "MID", "expected_goal": 0.1, "goal": False},
        ]
    )


def _statistics() -> dict[str, dict[str, dict[str, float]]]:
    return {
        group: {feature: {"mean": 0.0, "std": 2.0} for feature in FIT_FEATURES}
        for group in ("DEF", "MID", "FWD")
    }


def test_exact_v33_style_contract_has_no_leakage() -> None:
    validate_fit_contract()
    assert list(FIT_FEATURES) == SIMILARITY_FEATURE_COLUMNS
    forbidden = {
        "player_id", "team_id", "completed", "goal", "attacking_value",
        "archetype_name", "completion_above_expected_pp",
    }
    assert forbidden.isdisjoint(FIT_FEATURES)
    with pytest.raises(ValueError):
        validate_fit_contract([*FIT_FEATURES[:-1], "attacking_value"])


def test_exact_feature_definitions_are_recomputed_from_events() -> None:
    profile = _aggregate_profiles(
        _passes().loc[lambda x: x.team_id.eq(10)].assign(group="all"),
        _actions().loc[lambda x: x.team_id.eq(10)].assign(group="all"),
        ["group"],
        _shots().loc[lambda x: x.team_id.eq(10)].assign(group="all"),
    ).iloc[0]
    assert profile["expected_completion_rate"] == pytest.approx(0.85)
    assert profile["pressure_pass_rate"] == pytest.approx(0.5)
    assert profile["progressive_pass_rate"] == pytest.approx(0.5)
    assert profile["long_pass_rate"] == pytest.approx(0.5)
    assert profile["positive_forward_distance_per_100_passes"] == pytest.approx(1000)
    assert profile["carry_share_of_actions"] == pytest.approx(1 / 3)
    assert profile["final_third_entries_per_100_passes"] == pytest.approx(50)


def test_team_match_profiles_are_unique_and_chronological() -> None:
    chronology = pd.DataFrame(
        [{"match_id": 1, "match_date": "2024-01-01", "kick_off": "15:30:00",
          "home_team": "Target", "away_team": "Opponent"}]
    )
    result = build_team_match_profiles(_passes(), _actions(), _shots(), chronology)
    assert len(result) == 2
    assert not result.duplicated(["match_id", "team_id"]).any()
    assert result.loc[result.team_id.eq(10), "opponent_id"].iloc[0] == 20
    assert result.loc[result.team_id.eq(10), "home_away"].iloc[0] == "home"


def test_target_match_chronology_is_deterministic() -> None:
    source = pd.DataFrame(
        [
            {"match_id": 2, "match_date": "2024-02-01", "kick_off": "15:30:00",
             "home_team": "A", "away_team": "B"},
            {"match_id": 1, "match_date": "2024-01-01", "kick_off": "20:30:00",
             "home_team": "B", "away_team": "A"},
            {"match_id": 3, "match_date": "2024-02-01", "kick_off": "12:30:00",
             "home_team": "A", "away_team": "C"},
        ]
    )
    first = build_match_chronology(source, {1, 2, 3})
    second = build_match_chronology(source.sample(frac=1, random_state=8), {1, 2, 3})
    assert first["match_id"].tolist() == [1, 3, 2]
    pd.testing.assert_frame_equal(first, second)


def test_role_subset_uses_only_requested_team_position_and_removes_player() -> None:
    full = _profile_for_subset(
        _passes(), _actions(), _shots(), 10, position_group="MID"
    )
    assert full["pass_attempts"] == 1
    assert full["attacking_actions"] == 2
    assert full["action_players"] == 1
    with pytest.raises(ValueError):
        _profile_for_subset(
            _passes(), _actions(), _shots(), 10, position_group="MID",
            excluded_player_id=1,
        )


def test_match_resampling_is_grouped_and_deterministic() -> None:
    first = deterministic_match_resamples(range(1, 11), count=20, fraction=0.8, seed=7)
    second = deterministic_match_resamples(range(1, 11), count=20, fraction=0.8, seed=7)
    assert first == second
    assert all(len(sample) == 8 for sample in first)
    assert all(len(sample) == len(set(sample)) for sample in first)
    assert all(set(sample).issubset(set(range(1, 11))) for sample in first)


def test_role_and_player_normalization_use_same_statistics() -> None:
    raw = {feature: float(index * 2) for index, feature in enumerate(FIT_FEATURES)}
    role = normalize_profile(raw, "MID", _statistics())
    player = normalize_profile(pd.Series(raw), "MID", _statistics())
    assert np.array_equal(role, player)
    assert vector_distance(role, player) == 0
    with pytest.raises(ValueError):
        normalize_profile(raw, "GK", _statistics())


def test_external_fit_is_same_position_excludes_target_and_has_no_gk() -> None:
    rows = []
    for player_id, name, team, group in (
        (1, "Current", "Target", "MID"),
        (2, "External MID", "Other", "MID"),
        (3, "External DEF", "Other", "DEF"),
        (4, "External GK", "Other", "GK"),
    ):
        row = {
            "player_id": player_id, "player_name": name, "team_name": team,
            "position": "Center Midfield", "position_group": group,
            "pass_attempts": 100, "carries": 50, "matches_observed": 2,
            "actions": 150,
        }
        row.update({feature: 0.1 * player_id for feature in FIT_FEATURES})
        rows.append(row)
    raw = pd.DataFrame(rows)
    normalized = raw.copy()
    role = pd.Series({feature: 0.0 for feature in FIT_FEATURES})
    result = rank_external_candidates(
        normalized, raw, role, "MID", "Target", _statistics(), "rms"
    )
    assert result["player_id"].tolist() == [2]
    assert result["position_group"].eq("MID").all()
    assert result["team_name"].ne("Target").all()


def test_distance_uses_only_supplied_style_vectors() -> None:
    left = np.zeros(6)
    right = np.ones(6)
    assert vector_distance(left, right, "rms") == pytest.approx(1)
    assert vector_distance(left, right, "manhattan") == pytest.approx(1)
    assert np.isnan(vector_distance(left, right, "cosine"))
    with pytest.raises(ValueError):
        vector_distance(np.zeros(7), np.ones(7))
