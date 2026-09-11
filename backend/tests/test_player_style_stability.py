from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.player_feature_registry import PERFORMANCE_FEATURES, STYLE_CLUSTERING_FEATURES
from analytics.player_style_stability import (
    assign_player_match_halves,
    build_expanded_matrix,
    build_match_chronology,
    eligibility_mask,
    feature_stability,
    position_normalize,
    recompute_style_features,
)


def sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    passes = pd.DataFrame([
        {"player_id": 1, "match_id": 10, "expected_completion": 0.8,
         "under_pressure": False, "progressive": True, "pass_length": 35.0,
         "forward_distance": 20.0, "start_x": 70.0, "end_x": 90.0},
        {"player_id": 1, "match_id": 20, "expected_completion": 0.6,
         "under_pressure": True, "progressive": False, "pass_length": 10.0,
         "forward_distance": -5.0, "start_x": 85.0, "end_x": 75.0},
        {"player_id": 1, "match_id": 30, "expected_completion": 1.0,
         "under_pressure": False, "progressive": True, "pass_length": 40.0,
         "forward_distance": 10.0, "start_x": 75.0, "end_x": 85.0},
    ])
    actions = pd.DataFrame([
        {"player_id": 1, "match_id": 10, "action_type": "Pass"},
        {"player_id": 1, "match_id": 10, "action_type": "Carry"},
        {"player_id": 1, "match_id": 20, "action_type": "Carry"},
        {"player_id": 1, "match_id": 30, "action_type": "Pass"},
    ])
    profiles = pd.DataFrame([{"player_id": 1, "position_group": "MID"}])
    matches = pd.DataFrame([
        {"match_id": 30, "match_date": "2024-01-03", "kick_off": "15:00",
         "home_team": "C", "away_team": "D"},
        {"match_id": 10, "match_date": "2024-01-01", "kick_off": "15:00",
         "home_team": "A", "away_team": "B"},
        {"match_id": 20, "match_date": "2024-01-02", "kick_off": "15:00",
         "home_team": "B", "away_team": "C"},
    ])
    return passes, actions, profiles, matches


def test_match_halves_use_dates_and_are_disjoint() -> None:
    passes, actions, _profiles, matches = sources()
    chronology = build_match_chronology(matches, {10, 20, 30})
    halves = assign_player_match_halves(passes, actions, chronology, {1})
    assert halves.loc[halves["half"].eq("first"), "match_id"].tolist() == [10]
    assert halves.loc[halves["half"].eq("second"), "match_id"].tolist() == [20, 30]
    assert not halves.duplicated(["player_id", "match_id"]).any()


def test_match_chronology_rejects_missing_match() -> None:
    *_, matches = sources()
    with pytest.raises(ValueError, match="missing IDs"):
        build_match_chronology(matches, {10, 20, 30, 40})


def test_style_recomputation_matches_definitions_and_excludes_performance() -> None:
    passes, actions, profiles, _ = sources()
    result = recompute_style_features(passes, actions, profiles)
    assert result.loc[0, "expected_completion_rate"] == pytest.approx(0.8)
    assert result.loc[0, "pressure_pass_rate"] == pytest.approx(1 / 3)
    assert result.loc[0, "progressive_pass_rate"] == pytest.approx(2 / 3)
    assert result.loc[0, "long_pass_rate"] == pytest.approx(2 / 3)
    assert result.loc[0, "positive_forward_distance_per_100_passes"] == 1_000
    assert result.loc[0, "final_third_entries_per_100_passes"] == pytest.approx(200 / 3)
    assert result.loc[0, "carry_share_of_actions"] == 0.5
    assert not set(PERFORMANCE_FEATURES).intersection(result.columns)


def test_eligibility_is_complete_case_and_uses_separate_thresholds() -> None:
    passes, actions, profiles, _ = sources()
    frame = recompute_style_features(passes, actions, profiles)
    assert eligibility_mask(frame, 3, 2).tolist() == [True]
    assert eligibility_mask(frame, 4, 2).tolist() == [False]
    assert eligibility_mask(frame, 3, 3).tolist() == [False]
    frame.loc[0, "long_pass_rate"] = np.nan
    assert eligibility_mask(frame, 3, 2).tolist() == [False]


def test_feature_stability_reports_perfect_halves_without_imputation() -> None:
    rows = []
    halves = []
    for player_id, value in enumerate((0.1, 0.2, 0.4), start=1):
        row = {"player_id": player_id, "position_group": "MID", "pass_attempts": 50,
               "actions": 40, "carries": 20, "matches_observed": 2}
        row.update({feature: value for feature in STYLE_CLUSTERING_FEATURES})
        rows.append(row)
        for half in ("first", "second"):
            half_row = row.copy()
            half_row["half"] = half
            halves.append(half_row)
    result = feature_stability(
        pd.DataFrame(rows), pd.DataFrame(halves), 25, 15, require_complete_cohort=True
    )
    assert all(metric["eligible_players"] == 3 for metric in result.values())
    assert all(metric["pearson"] == pytest.approx(1.0) for metric in result.values())
    assert all(metric["median_absolute_difference"] == 0 for metric in result.values())


def test_expanded_matrix_records_exact_ineligibility_causes() -> None:
    passes, actions, profiles, _ = sources()
    frame = recompute_style_features(passes, actions, profiles)
    result = build_expanded_matrix(frame)
    reason = result.loc[0, "ineligibility__pass_25__carry_15"]
    assert "six_pass_features:passes<25" in reason
    assert "carry_share_of_actions:carries<15" in reason
    assert not bool(result.loc[0, "complete__pass_25__carry_15"])


def test_position_normalization_is_within_group_and_preserves_metadata() -> None:
    rows = []
    for group, offset in (("DEF", 0.0), ("MID", 10.0)):
        for index in range(3):
            row = {"player_id": len(rows) + 1, "position_group": group}
            row.update({feature: offset + index for feature in STYLE_CLUSTERING_FEATURES})
            rows.append(row)
    normalized = position_normalize(pd.DataFrame(rows))
    assert list(normalized.columns[:2]) == ["player_id", "position_group"]
    for feature in STYLE_CLUSTERING_FEATURES:
        column = f"{feature}_position_z"
        assert normalized.groupby("position_group")[column].mean().abs().max() < 1e-12
        assert np.allclose(
            normalized.groupby("position_group")[column].std(ddof=0), 1.0
        )
