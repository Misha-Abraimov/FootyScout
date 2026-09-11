import math

import numpy as np
import pandas as pd

from analytics.player_profiles import (
    LONG_PASS_LENGTH_THRESHOLD,
    MIN_OVERALL_ATTEMPTS,
    MIN_SUBSET_ATTEMPTS,
    build_player_profiles,
    is_final_third_entry,
    validate_profiles,
)


def sample_oof_data() -> pd.DataFrame:
    rows = [
        # Player A: completion=3/4, expected=2/4.
        (1, 1, 10, "Player A", 100, "Team A", "Midfield", 1, 0.8, True, True, 10.0, 10.0, 79.0, 80.0, 1),
        (2, 1, 10, "Player A", 100, "Team A", "Midfield", 1, 0.6, True, False, 30.0, -5.0, 81.0, 70.0, 1),
        (3, 2, 10, "Player A", 100, "Team A", "Forward", 0, 0.4, False, True, 20.0, 20.0, 70.0, 85.0, 2),
        (4, 2, 10, "Player A", 100, "Team A", "Midfield", 1, 0.2, False, False, 35.0, 0.0, 90.0, 95.0, 2),
        # Player B has no pressure, progressive, or long passes.
        (5, 1, 20, "Player B", 200, "Team B", "Defender", 1, 0.9, False, False, 5.0, 2.0, 20.0, 22.0, 1),
        (6, 2, 20, "Player B", 200, "Team B", "Defender", 0, 0.3, False, False, 8.0, -2.0, 30.0, 28.0, 2),
    ]
    columns = [
        "pass_index",
        "match_id",
        "player_id",
        "player_name",
        "team_id",
        "team_name",
        "position",
        "completed",
        "expected_completion",
        "under_pressure",
        "progressive",
        "pass_length",
        "forward_distance",
        "start_x",
        "end_x",
        "fold",
    ]
    return pd.DataFrame(rows, columns=columns)


def player_a_profile() -> pd.Series:
    profiles, _ = build_player_profiles(sample_oof_data())
    return profiles.set_index("player_id").loc[10]


def test_overall_actual_expected_and_percentage_points() -> None:
    profile = player_a_profile()

    assert profile["pass_attempts"] == 4
    assert profile["passes_completed"] == 3
    assert profile["actual_completion_rate"] == 0.75
    assert math.isclose(profile["expected_completions"], 2.0)
    assert math.isclose(profile["expected_completion_rate"], 0.5)
    assert math.isclose(profile["completions_above_expected"], 1.0)
    assert math.isclose(profile["completion_above_expected_pp"], 25.0)


def test_under_pressure_subset() -> None:
    profile = player_a_profile()

    assert profile["pressure_attempts"] == 2
    assert profile["pressure_completed"] == 2
    assert profile["pressure_actual_completion_rate"] == 1.0
    assert math.isclose(profile["pressure_expected_completion_rate"], 0.7)
    assert math.isclose(profile["pressure_completions_above_expected"], 0.6)
    assert math.isclose(profile["pressure_above_expected_pp"], 30.0)
    assert profile["pressure_pass_rate"] == 0.5


def test_progressive_subset() -> None:
    profile = player_a_profile()

    assert profile["progressive_attempts"] == 2
    assert profile["progressive_completed"] == 1
    assert profile["progressive_actual_completion_rate"] == 0.5
    assert math.isclose(profile["progressive_expected_completion_rate"], 0.6)
    assert math.isclose(profile["progressive_completions_above_expected"], -0.2)
    assert math.isclose(profile["progressive_above_expected_pp"], -10.0)
    assert profile["progressive_pass_rate"] == 0.5


def test_long_pass_subset_includes_threshold() -> None:
    profile = player_a_profile()

    assert LONG_PASS_LENGTH_THRESHOLD == 30.0
    assert profile["long_pass_attempts"] == 2
    assert profile["long_pass_completed"] == 2
    assert profile["long_pass_actual_completion_rate"] == 1.0
    assert math.isclose(profile["long_pass_expected_completion_rate"], 0.4)
    assert math.isclose(profile["long_pass_completions_above_expected"], 1.2)
    assert math.isclose(profile["long_pass_above_expected_pp"], 60.0)


def test_final_third_entry_and_per_100_calculations() -> None:
    profile = player_a_profile()

    assert is_final_third_entry(79.9, 80.0)
    assert not is_final_third_entry(80.0, 90.0)
    assert not is_final_third_entry(70.0, 79.9)
    assert profile["final_third_entries"] == 2
    assert profile["final_third_entries_per_100_passes"] == 50.0
    assert profile["average_forward_distance"] == 6.25
    assert profile["net_forward_distance_per_100_passes"] == 625.0
    assert profile["positive_forward_distance_per_100_passes"] == 750.0


def test_zero_attempt_subsets_have_null_metrics() -> None:
    profiles, _ = build_player_profiles(sample_oof_data())
    profile = profiles.set_index("player_id").loc[20]

    for prefix in ("pressure", "progressive", "long_pass"):
        assert profile[f"{prefix}_attempts"] == 0
        assert profile[f"{prefix}_completed"] == 0
        assert pd.isna(profile[f"{prefix}_actual_completion_rate"])
        assert pd.isna(profile[f"{prefix}_expected_completion_rate"])
        assert pd.isna(profile[f"{prefix}_completions_above_expected"])
        assert pd.isna(profile[f"{prefix}_above_expected_pp"])


def test_reliability_thresholds_are_exact() -> None:
    base = sample_oof_data().iloc[[0]].copy()
    rows = pd.concat([base] * MIN_OVERALL_ATTEMPTS, ignore_index=True)
    rows["pass_index"] = np.arange(len(rows))
    rows["under_pressure"] = rows.index < MIN_SUBSET_ATTEMPTS
    rows["progressive"] = rows.index < MIN_SUBSET_ATTEMPTS
    rows["pass_length"] = np.where(
        rows.index < MIN_SUBSET_ATTEMPTS,
        LONG_PASS_LENGTH_THRESHOLD,
        5.0,
    )
    profiles, _ = build_player_profiles(rows)
    profile = profiles.iloc[0]

    assert profile["overall_reliable"]
    assert profile["pressure_reliable"]
    assert profile["progressive_reliable"]
    assert profile["long_pass_reliable"]


def test_aggregation_totals_and_weighted_expected_mean() -> None:
    source = sample_oof_data()
    profiles, _ = build_player_profiles(source)

    validate_profiles(profiles, source)
    assert profiles["pass_attempts"].sum() == len(source)
    assert profiles["passes_completed"].sum() == source["completed"].sum()
    weighted_expected = np.average(
        profiles["expected_completion_rate"],
        weights=profiles["pass_attempts"],
    )
    assert math.isclose(weighted_expected, source["expected_completion"].mean())
