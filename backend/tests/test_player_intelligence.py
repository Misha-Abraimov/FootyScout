from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.player_feature_registry import (
    FEATURE_REGISTRY,
    PERFORMANCE_FEATURES,
    STYLE_CLUSTERING_FEATURES,
)
from analytics.player_intelligence import (
    build_percentiles,
    build_style_matrix,
    build_unified_profiles,
)


def percentile_profiles(count: int = 10, group: str = "DEF") -> pd.DataFrame:
    rows = []
    for index in range(count):
        row: dict[str, object] = {
            "player_id": index + 1,
            "position_group": group,
            "pass_attempts": 150,
            "pressure_attempts": 25,
            "progressive_attempts": 25,
            "long_pass_attempts": 25,
            "shots": 25,
            "actions": 100,
            "passes": 70,
            "carries": 30,
        }
        row.update({feature.feature_name: float(index) for feature in FEATURE_REGISTRY})
        rows.append(row)
    return pd.DataFrame(rows)


def source_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    passing = pd.DataFrame([{
        "player_id": 1, "player_name": "Player", "team_id": 10, "team_name": "Team",
        "position": "Center Back", "matches_observed": 2, "pass_attempts": 120,
        "pressure_attempts": 20, "progressive_attempts": 22, "long_pass_attempts": 21,
        "overall_reliable": True, "pressure_reliable": True,
        "progressive_reliable": True, "long_pass_reliable": True,
        "expected_completion_rate": 0.8, "pressure_pass_rate": 0.2,
        "progressive_pass_rate": 0.18, "average_forward_distance": 5.0,
        "positive_forward_distance_per_100_passes": 700.0,
        "final_third_entries_per_100_passes": 8.0,
        "completion_above_expected_pp": 2.0, "pressure_above_expected_pp": 1.0,
        "progressive_above_expected_pp": 3.0, "long_pass_above_expected_pp": -1.0,
    }])
    shooting = pd.DataFrame([{
        "player_id": 1, "shots": 20, "goals": 2, "total_xg": 1.5,
        "xg_per_shot": 0.075, "goals_minus_xg": 0.5, "shooting_reliable": True,
    }])
    attacking = pd.DataFrame([{
        "player_id": 1, "actions": 1, "passes": 0, "carries": 1,
        "attacking_value_per_100_actions": 0.2, "pass_value_per_100_passes": np.nan,
        "carry_value_per_100_carries": 0.2, "progressive_value_per_100_actions": 0.2,
        "pressure_value_per_100_actions": 0.0, "attacking_value_reliable": False,
        "pass_value_reliable": False, "carry_value_reliable": False,
    }])
    actions = pd.DataFrame([{
        "player_id": 1, "action_type": "Carry", "progressive": True,
        "under_pressure": False,
    }])
    return passing, shooting, attacking, actions


def test_profile_join_preserves_one_player_and_position_mapping() -> None:
    profile = build_unified_profiles(*source_frames())
    assert len(profile) == 1
    assert profile.loc[0, "position_group"] == "DEF"
    assert profile.loc[0, "long_pass_rate"] == pytest.approx(21 / 120)
    assert profile.loc[0, "progressive_carry_rate"] == 1.0


def test_profile_join_rejects_duplicate_players() -> None:
    passing, shooting, attacking, actions = source_frames()
    with pytest.raises(ValueError, match="one row per player"):
        build_unified_profiles(pd.concat([passing, passing]), shooting, attacking, actions)


def test_missing_optional_profiles_do_not_delete_player() -> None:
    passing, shooting, attacking, actions = source_frames()
    profile = build_unified_profiles(
        passing, shooting.iloc[0:0], attacking.iloc[0:0], actions.iloc[0:0]
    )
    assert len(profile) == 1
    assert profile.loc[0, "shots"] == 0
    assert pd.isna(profile.loc[0, "xg_per_shot"])
    assert pd.isna(profile.loc[0, "carry_share_of_actions"])


def test_percentile_formula_uses_average_ties_and_full_zero_to_100_range() -> None:
    profiles = percentile_profiles()
    profiles.loc[1, "completion_above_expected_pp"] = 0.0
    result = build_percentiles(profiles)
    metric = result.loc[result["metric_name"].eq("completion_above_expected_pp")]
    assert metric.loc[metric["player_id"].eq(1), "percentile"].iloc[0] == pytest.approx(100 * 0.5 / 9)
    assert metric.loc[metric["player_id"].eq(2), "percentile"].iloc[0] == pytest.approx(100 * 0.5 / 9)
    assert metric["percentile"].max() == 100
    assert metric["peer_count"].eq(10).all()


def test_metric_specific_eligibility_does_not_use_one_universal_threshold() -> None:
    profiles = percentile_profiles()
    profiles.loc[0, "pass_attempts"] = 99
    result = build_percentiles(profiles).set_index(["player_id", "metric_name"])
    assert not result.at[(1, "completion_above_expected_pp"), "eligible"]
    assert result.at[(1, "completion_above_expected_pp"), "eligibility_reason"] == "requires_100_passes"
    assert result.at[(1, "carry_value_per_100_carries"), "eligible"]


def test_small_position_peer_group_keeps_null_percentiles() -> None:
    result = build_percentiles(percentile_profiles(9))
    assert result["percentile"].isna().all()
    assert result["eligibility_reason"].eq("requires_10_eligible_position_peers").all()


def test_style_matrix_excludes_identity_and_performance_features() -> None:
    profiles = percentile_profiles()
    percentiles = build_percentiles(profiles)
    matrix = build_style_matrix(profiles, percentiles)
    assert set(STYLE_CLUSTERING_FEATURES).issubset(matrix.columns)
    assert not set(PERFORMANCE_FEATURES).intersection(matrix.columns)
    assert "player_name" not in matrix and "team_id" not in matrix
    assert not any(column.startswith("percentile") for column in matrix)


def test_missing_metric_is_ineligible_without_imputation() -> None:
    profiles = percentile_profiles()
    profiles.loc[0, "xg_per_shot"] = np.nan
    result = build_percentiles(profiles).set_index(["player_id", "metric_name"])
    assert pd.isna(result.at[(1, "xg_per_shot"), "raw_value"])
    assert pd.isna(result.at[(1, "xg_per_shot"), "percentile"])
    assert result.at[(1, "xg_per_shot"), "eligibility_reason"] == "metric_unavailable"
