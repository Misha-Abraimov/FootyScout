import math

import pandas as pd
import pytest

from analytics.pass_features import (
    KNOWN_LEAKAGE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    REGULAR_PASS_TYPE,
    UNKNOWN_CATEGORY,
    calculate_distance_to_goal,
    calculate_forward_distance,
    calculate_lateral_distance,
    classify_pitch_zone,
    engineer_pass_features,
    is_progressive_pass,
    validate_model_feature_columns,
)


def sample_passes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": 1,
                "player_id": 10,
                "player_name": "Player One",
                "team_id": 20,
                "team_name": "Team One",
                "position": "Center Midfield",
                "start_x": 20.0,
                "start_y": 10.0,
                "end_x": 60.0,
                "end_y": 30.0,
                "pass_length": 44.72,
                "pass_angle": 0.46,
                "pass_height": "Ground Pass",
                "body_part": None,
                "pass_type": pd.NA,
                "under_pressure": False,
                "pass_outcome": None,
                "completed": 1,
            }
        ]
    )


def test_forward_distance() -> None:
    assert calculate_forward_distance(20.0, 60.0) == 40.0
    assert calculate_forward_distance(60.0, 20.0) == -40.0
    assert calculate_forward_distance(None, 20.0) is None


def test_lateral_distance() -> None:
    assert calculate_lateral_distance(10.0, 30.0) == 20.0
    assert calculate_lateral_distance(60.0, 20.0) == 40.0
    assert calculate_lateral_distance(math.nan, 20.0) is None


def test_distance_to_goal_uses_statsbomb_goal_center() -> None:
    assert calculate_distance_to_goal(120.0, 40.0) == 0.0
    assert calculate_distance_to_goal(100.0, 40.0) == 20.0
    assert calculate_distance_to_goal(120.0, 10.0) == 30.0
    assert calculate_distance_to_goal(None, 40.0) is None


def test_progressive_pass_requires_25_percent_distance_reduction() -> None:
    assert is_progressive_pass(100.0, 75.0)
    assert not is_progressive_pass(100.0, 75.01)
    assert not is_progressive_pass(100.0, 110.0)
    assert not is_progressive_pass(0.0, 0.0)
    assert not is_progressive_pass(None, 10.0)


def test_pitch_zone_classification() -> None:
    assert classify_pitch_zone(0.0, 0.0) == "defensive_left"
    assert classify_pitch_zone(40.0, 40.0) == "middle_centre"
    assert classify_pitch_zone(80.0, 80.0) == "attacking_right"
    assert classify_pitch_zone(120.0, 40.0) == "attacking_centre"
    assert classify_pitch_zone(None, 40.0) == UNKNOWN_CATEGORY
    assert classify_pitch_zone(121.0, 40.0) == UNKNOWN_CATEGORY


def test_missing_categorical_values_are_retained_as_unknown() -> None:
    features, report = engineer_pass_features(sample_passes())

    assert features.loc[0, "body_part"] == UNKNOWN_CATEGORY
    assert features.loc[0, "pass_type"] == REGULAR_PASS_TYPE
    assert report.categorical_values_filled["body_part"] == 1
    assert report.categorical_values_filled["pass_type"] == 1
    assert report.rows_removed == 0


def test_model_feature_columns_prevent_target_leakage() -> None:
    assert not KNOWN_LEAKAGE_COLUMNS.intersection(MODEL_FEATURE_COLUMNS)
    validate_model_feature_columns()

    with pytest.raises(ValueError, match="Target leakage"):
        validate_model_feature_columns([*MODEL_FEATURE_COLUMNS, "pass_outcome"])

    with pytest.raises(ValueError, match="Target leakage"):
        validate_model_feature_columns([*MODEL_FEATURE_COLUMNS, "completed"])


def test_output_schema_contains_only_metadata_features_and_target() -> None:
    features, _ = engineer_pass_features(sample_passes())

    assert list(features.columns) == OUTPUT_COLUMNS
    assert "pass_outcome" not in features.columns
    assert "minute" not in features.columns
    assert "second" not in features.columns
    assert features.loc[0, "forward_distance"] == 40.0
    assert features.loc[0, "lateral_distance"] == 20.0
