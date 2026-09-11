import math

import pandas as pd
import pytest

from analytics.shot_features import (
    KNOWN_XG_LEAKAGE_COLUMNS,
    XG_MODEL_FEATURE_COLUMNS,
    angle_to_goal,
    distance_to_goal,
    goal_from_outcome,
    normalize_shot_event,
    validate_xg_model_contract,
)


def shot_event(**shot_overrides: object) -> dict[str, object]:
    shot = {
        "outcome": {"id": 97, "name": "Goal"},
        "type": {"id": 87, "name": "Open Play"},
        "body_part": {"name": "Right Foot"},
        "technique": {"name": "Normal"},
        **shot_overrides,
    }
    return {
        "id": "shot-1",
        "index": 12,
        "type": {"name": "Shot"},
        "location": [108.0, 40.0],
        "match_id": 1,
        "period": 1,
        "minute": 10,
        "second": 4,
        "player": {"id": 10, "name": "Player"},
        "team": {"id": 20, "name": "Team"},
        "position": {"name": "Forward"},
        "play_pattern": {"name": "Regular Play"},
        "shot": shot,
    }


def test_geometry_uses_statsbomb_goal() -> None:
    assert distance_to_goal(108.0, 40.0) == pytest.approx(12.0)
    expected = 2 * math.atan(4 / 12)
    assert angle_to_goal(108.0, 40.0) == pytest.approx(expected)
    assert angle_to_goal(114.0, 40.0) > angle_to_goal(108.0, 40.0)


def test_goal_label_uses_explicit_outcome_name() -> None:
    assert goal_from_outcome({"name": "Goal"}) == 1
    assert goal_from_outcome({"name": "Saved"}) == 0
    assert goal_from_outcome(None) is None


def test_penalties_and_shootouts_are_not_model_eligible() -> None:
    event = shot_event(type={"name": "Penalty"})
    event["period"] = 5
    record = normalize_shot_event(event, competition_id=9, season_id=281, product_cohort=True)
    assert record is not None
    assert record["penalty"] is True
    assert record["penalty_shootout"] is True
    assert record["model_eligible"] is False


def test_missing_categories_are_unknown_and_flags_are_false() -> None:
    event = shot_event(body_part=None, technique=None)
    event["play_pattern"] = None
    record = normalize_shot_event(event, competition_id=9, season_id=281, product_cohort=True)
    assert record is not None
    assert record["body_part"] == "Unknown"
    assert record["technique"] == "Unknown"
    assert record["play_pattern"] == "Unknown"
    assert record["under_pressure"] is False
    assert record["first_time"] is False


def test_missing_geometry_is_preserved_but_not_model_eligible() -> None:
    event = shot_event()
    event["location"] = pd.NA
    record = normalize_shot_event(event, competition_id=9, season_id=281, product_cohort=True)
    assert record is not None
    assert record["shot_x"] is None
    assert record["model_eligible"] is False


def test_xg_feature_contract_excludes_known_leakage() -> None:
    validate_xg_model_contract()
    assert not KNOWN_XG_LEAKAGE_COLUMNS.intersection(XG_MODEL_FEATURE_COLUMNS)

