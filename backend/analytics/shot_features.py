"""Normalization and leakage-safe pre-shot features for StatsBomb shot events."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_CENTER = (120.0, 40.0)
GOAL_POST_Y = (36.0, 44.0)
UNKNOWN_CATEGORY = "Unknown"

SHOT_METADATA_COLUMNS = [
    "shot_id",
    "event_index",
    "competition_id",
    "season_id",
    "match_id",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "position",
    "minute",
    "second",
    "period",
    "is_product_cohort",
]
XG_NUMERIC_FEATURE_COLUMNS = [
    "shot_x",
    "shot_y",
    "distance_to_goal",
    "angle_to_goal",
]
XG_BOOLEAN_FEATURE_COLUMNS = [
    "under_pressure",
    "first_time",
    "one_on_one",
    "open_goal",
]
XG_CATEGORICAL_FEATURE_COLUMNS = [
    "body_part",
    "shot_type",
    "technique",
    "play_pattern",
]
XG_MODEL_FEATURE_COLUMNS = (
    XG_NUMERIC_FEATURE_COLUMNS
    + XG_BOOLEAN_FEATURE_COLUMNS
    + XG_CATEGORICAL_FEATURE_COLUMNS
)
SHOT_AUDIT_COLUMNS = ["shot_outcome", "statsbomb_xg", "penalty", "penalty_shootout"]
TARGET_COLUMN = "goal"
SHOT_COLUMNS = (
    SHOT_METADATA_COLUMNS
    + XG_MODEL_FEATURE_COLUMNS
    + SHOT_AUDIT_COLUMNS
    + [TARGET_COLUMN, "model_eligible"]
)
KNOWN_XG_LEAKAGE_COLUMNS = {
    TARGET_COLUMN,
    "shot_outcome",
    "statsbomb_xg",
    "shot_end_location",
    "goalkeeper_outcome",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "shot_id",
    "event_index",
    "next_event",
}


def _missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _entity_name(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    name = value.get("name")
    return None if _missing(name) else str(name)


def _entity_id(value: object) -> int | None:
    if not isinstance(value, Mapping):
        return None
    value_id = value.get("id")
    if _missing(value_id):
        return None
    try:
        return int(value_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def extract_xy(value: object) -> tuple[float | None, float | None]:
    """Extract the first two finite coordinates from a StatsBomb location."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None, None
    try:
        x, y = float(value[0]), float(value[1])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(x) or not math.isfinite(y):
        return None, None
    return x, y


def distance_to_goal(x: float, y: float) -> float:
    """Straight-line distance to the StatsBomb attacking goal centre (120, 40)."""
    return math.hypot(GOAL_CENTER[0] - x, GOAL_CENTER[1] - y)


def angle_to_goal(x: float, y: float) -> float:
    """Angle in radians subtended by the goalposts at (120, 36) and (120, 44)."""
    left = (PITCH_LENGTH - x, GOAL_POST_Y[0] - y)
    right = (PITCH_LENGTH - x, GOAL_POST_Y[1] - y)
    cross = abs(left[0] * right[1] - left[1] * right[0])
    dot = left[0] * right[0] + left[1] * right[1]
    return float(math.atan2(cross, dot))


def goal_from_outcome(outcome: object) -> int | None:
    """Map the explicit StatsBomb shot outcome name to a goal label."""
    name = _entity_name(outcome)
    if name is None:
        return None
    return int(name.casefold() == "goal")


def validate_xg_model_contract() -> None:
    leaked = KNOWN_XG_LEAKAGE_COLUMNS.intersection(XG_MODEL_FEATURE_COLUMNS)
    if leaked:
        raise ValueError(f"Leakage columns configured as xG inputs: {sorted(leaked)}")
    if len(XG_MODEL_FEATURE_COLUMNS) != len(set(XG_MODEL_FEATURE_COLUMNS)):
        raise ValueError("xG model features must be unique")


def normalize_shot_event(
    event: Mapping[str, Any],
    *,
    competition_id: int,
    season_id: int,
    product_cohort: bool,
) -> dict[str, object] | None:
    """Normalize one StatsBomb event; omitted boolean flags mean false."""
    if _entity_name(event.get("type")) != "Shot":
        return None
    shot = event.get("shot")
    if not isinstance(shot, Mapping):
        shot = {}
    x, y = extract_xy(event.get("location"))
    shot_type = _entity_name(shot.get("type"))
    outcome = shot.get("outcome")
    goal = goal_from_outcome(outcome)
    period = event.get("period")
    penalty = shot_type == "Penalty"
    penalty_shootout = penalty and period == 5
    geometry_available = x is not None and y is not None
    return {
        "shot_id": event.get("id"),
        "event_index": event.get("index"),
        "competition_id": competition_id,
        "season_id": season_id,
        "match_id": event.get("match_id"),
        "player_id": _entity_id(event.get("player")),
        "player_name": _entity_name(event.get("player")),
        "team_id": _entity_id(event.get("team")),
        "team_name": _entity_name(event.get("team")),
        "position": _entity_name(event.get("position")),
        "minute": event.get("minute"),
        "second": event.get("second"),
        "period": period,
        "is_product_cohort": product_cohort,
        "shot_x": x,
        "shot_y": y,
        "distance_to_goal": distance_to_goal(x, y) if geometry_available else None,
        "angle_to_goal": angle_to_goal(x, y) if geometry_available else None,
        "under_pressure": bool(event.get("under_pressure", False)),
        "first_time": bool(shot.get("first_time", False)),
        "one_on_one": bool(shot.get("one_on_one", False)),
        "open_goal": bool(shot.get("open_goal", False)),
        "body_part": _entity_name(shot.get("body_part")) or UNKNOWN_CATEGORY,
        "shot_type": shot_type or UNKNOWN_CATEGORY,
        "technique": _entity_name(shot.get("technique")) or UNKNOWN_CATEGORY,
        "play_pattern": _entity_name(event.get("play_pattern")) or UNKNOWN_CATEGORY,
        "shot_outcome": _entity_name(outcome),
        "statsbomb_xg": shot.get("statsbomb_xg"),
        "penalty": penalty,
        "penalty_shootout": penalty_shootout,
        "goal": goal,
        "model_eligible": bool(not penalty and period != 5 and geometry_available and goal is not None),
    }


def normalize_shot_events(
    events: Sequence[Mapping[str, Any]],
    *,
    competition_id: int,
    season_id: int,
    product_cohort: bool,
) -> pd.DataFrame:
    records = [
        record
        for event in events
        if (
            record := normalize_shot_event(
                event,
                competition_id=competition_id,
                season_id=season_id,
                product_cohort=product_cohort,
            )
        )
        is not None
    ]
    frame = pd.DataFrame.from_records(records, columns=SHOT_COLUMNS)
    for column in [
        "event_index",
        "competition_id",
        "season_id",
        "match_id",
        "player_id",
        "team_id",
        "minute",
        "second",
        "period",
        "goal",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in ["shot_x", "shot_y", "distance_to_goal", "angle_to_goal", "statsbomb_xg"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    for column in [
        "shot_id",
        "player_name",
        "team_name",
        "position",
        "body_part",
        "shot_type",
        "technique",
        "play_pattern",
        "shot_outcome",
    ]:
        frame[column] = frame[column].astype("string")
    for column in XG_BOOLEAN_FEATURE_COLUMNS + [
        "is_product_cohort",
        "penalty",
        "penalty_shootout",
        "model_eligible",
    ]:
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame
