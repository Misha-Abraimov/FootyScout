import math

import pandas as pd

from scripts.download_statsbomb import (
    PASS_COLUMNS,
    available_match_ids,
    completion_from_outcome,
    extract_coordinates,
    normalize_pass_event,
    normalize_pass_events,
    save_parquet,
    select_competition,
)


def test_extract_coordinates() -> None:
    assert extract_coordinates([61, 40.1]) == (61.0, 40.1)
    assert extract_coordinates((58.5, 38.9, 2.0)) == (58.5, 38.9)


def test_extract_coordinates_handles_missing_and_invalid_values() -> None:
    assert extract_coordinates(None) == (None, None)
    assert extract_coordinates([]) == (None, None)
    assert extract_coordinates([math.nan, "invalid"]) == (None, None)


def test_completion_uses_statsbomb_null_outcome_convention() -> None:
    assert completion_from_outcome(None) == 1
    assert completion_from_outcome(pd.NA) == 1
    assert completion_from_outcome(float("nan")) == 1
    assert completion_from_outcome({}) == 1
    assert completion_from_outcome({"id": 9, "name": "Incomplete"}) == 0
    assert completion_from_outcome({"id": 74, "name": "Out"}) == 0


def test_normalize_pass_event_extracts_requested_fields() -> None:
    event = {
        "match_id": 3895292,
        "type": {"id": 30, "name": "Pass"},
        "player": {"id": 24243, "name": "Brenden Aaronson"},
        "team": {"id": 190, "name": "Union Berlin"},
        "position": {"id": 19, "name": "Center Attacking Midfield"},
        "minute": 0,
        "second": 1,
        "location": [61.0, 40.1],
        "under_pressure": True,
        "pass": {
            "length": 2.7730849,
            "angle": -2.6940727,
            "height": {"id": 1, "name": "Ground Pass"},
            "end_location": [58.5, 38.9],
            "body_part": {"id": 40, "name": "Right Foot"},
            "type": {"id": 65, "name": "Kick Off"},
        },
    }

    record = normalize_pass_event(event)

    assert record is not None
    assert record["start_x"] == 61.0
    assert record["start_y"] == 40.1
    assert record["end_x"] == 58.5
    assert record["end_y"] == 38.9
    assert record["pass_outcome"] is None
    assert record["completed"] == 1
    assert record["under_pressure"] is True


def test_normalize_pass_events_handles_missing_values_safely() -> None:
    events = [
        {
            "match_id": 1,
            "type": {"name": "Pass"},
            "location": None,
            "pass": {
                "end_location": [90.0],
                "outcome": {"id": 9, "name": "Incomplete"},
            },
        },
        {"match_id": 1, "type": {"name": "Carry"}},
    ]

    passes = normalize_pass_events(events)

    assert list(passes.columns) == PASS_COLUMNS
    assert len(passes) == 1
    assert pd.isna(passes.loc[0, "player_id"])
    assert pd.isna(passes.loc[0, "start_x"])
    assert pd.isna(passes.loc[0, "end_x"])
    assert not passes.loc[0, "under_pressure"]
    assert passes.loc[0, "completed"] == 0
    assert passes.loc[0, "pass_outcome"] == "Incomplete"


def test_select_competition_prefers_2023_24_bundesliga() -> None:
    competitions = pd.DataFrame(
        [
            {
                "competition_id": 9,
                "season_id": 27,
                "country_name": "Germany",
                "competition_name": "1. Bundesliga",
                "competition_gender": "male",
                "season_name": "2015/2016",
            },
            {
                "competition_id": 9,
                "season_id": 281,
                "country_name": "Germany",
                "competition_name": "1. Bundesliga",
                "competition_gender": "male",
                "season_name": "2023/2024",
            },
        ]
    )

    selected = select_competition(competitions)

    assert selected.competition_id == 9
    assert selected.season_id == 281
    assert selected.season_name == "2023/2024"


def test_available_match_ids_filters_non_available_rows() -> None:
    matches = pd.DataFrame(
        {
            "match_id": [10, 11, 10, None],
            "match_status": ["available", "processing", "available", "available"],
        }
    )

    assert available_match_ids(matches) == [10]


def test_save_parquet_round_trip(tmp_path) -> None:
    passes = normalize_pass_events(
        [
            {
                "match_id": 1,
                "type": {"name": "Pass"},
                "location": [10.0, 20.0],
                "pass": {"end_location": [30.0, 40.0]},
            }
        ]
    )
    output_path = tmp_path / "passes.parquet"

    save_parquet(passes, output_path)
    restored = pd.read_parquet(output_path)

    assert output_path.exists()
    assert list(restored.columns) == PASS_COLUMNS
    assert restored.loc[0, "completed"] == 1
