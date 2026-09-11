"""Normalize StatsBomb possession states and build leakage-safe action-value targets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analytics.shot_features import angle_to_goal, distance_to_goal

UNKNOWN_CATEGORY = "Unknown"
TARGET_COLUMN = "future_oof_xg_same_possession"
HORIZON_COLUMNS = {
    "remaining_possession": TARGET_COLUMN,
    "next_5_events": "future_oof_xg_next_5",
    "next_10_events": "future_oof_xg_next_10",
}

ACTION_VALUE_NUMERIC_FEATURE_COLUMNS = [
    "ball_x",
    "ball_y",
    "distance_to_goal",
    "angle_to_goal",
    "period",
    "minute",
    "score_difference",
    "possession_action_number",
    "possession_elapsed_time",
    "possession_start_x",
    "possession_start_y",
    "distance_progressed_from_possession_start",
]
ACTION_VALUE_BOOLEAN_FEATURE_COLUMNS = ["previous_action_success", "under_pressure"]
ACTION_VALUE_CATEGORICAL_FEATURE_COLUMNS = ["previous_action_type", "current_play_pattern"]
ACTION_VALUE_FEATURE_COLUMNS = (
    ACTION_VALUE_NUMERIC_FEATURE_COLUMNS
    + ACTION_VALUE_BOOLEAN_FEATURE_COLUMNS
    + ACTION_VALUE_CATEGORICAL_FEATURE_COLUMNS
)

KNOWN_ACTION_VALUE_LEAKAGE_COLUMNS = frozenset(
    {
        TARGET_COLUMN,
        "future_oof_xg_next_5",
        "future_oof_xg_next_10",
        "oof_shot_xg",
        "statsbomb_xg",
        "goal",
        "shot_outcome",
        "next_event_type",
        "next_action_type",
        "possession_final_outcome",
        "possession_scores",
        "player_id",
        "player_name",
        "team_id",
        "team_name",
        "possession_team_id",
        "possession_team_name",
        "event_id",
        "match_id",
    }
)

ON_BALL_EVENT_TYPES = frozenset(
    {
        "50/50",
        "Ball Receipt*",
        "Ball Recovery",
        "Carry",
        "Clearance",
        "Dispossessed",
        "Dribble",
        "Duel",
        "Foul Won",
        "Goal Keeper",
        "Interception",
        "Miscontrol",
        "Pass",
        "Shot",
    }
)
TERMINAL_EVENT_TYPES = frozenset(
    {"Dispossessed", "Half End", "Miscontrol", "Out", "Shot", "Match End"}
)

STATE_METADATA_COLUMNS = [
    "event_id",
    "event_index",
    "match_id",
    "competition_id",
    "season_id",
    "is_product_cohort",
    "possession_id",
    "possession_team_id",
    "possession_team_name",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "event_type",
    "period",
    "minute",
    "second",
    "timestamp",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "event_success",
    "pass_ordinal",
]


def _entity(value: object, key: str) -> object | None:
    return value.get(key) if isinstance(value, Mapping) else None


def extract_xy(value: object) -> tuple[float | None, float | None]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None, None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None, None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else (None, None)


def timestamp_seconds(value: object, period: object) -> float | None:
    """Convert StatsBomb HH:MM:SS.s timestamps to seconds within a period."""
    if not isinstance(value, str):
        return None
    try:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        return None


def pass_completed(pass_data: Mapping[str, Any]) -> bool:
    """StatsBomb successful passes omit pass.outcome; a populated outcome is failure."""
    outcome = pass_data.get("outcome")
    return not (isinstance(outcome, Mapping) and (outcome.get("id") or outcome.get("name")))


def event_success(event_type: str, event: Mapping[str, Any]) -> bool:
    if event_type == "Pass":
        payload = event.get("pass")
        return pass_completed(payload if isinstance(payload, Mapping) else {})
    if event_type == "Dribble":
        payload = event.get("dribble")
        outcome = _entity(payload, "outcome")
        return _entity(outcome, "name") == "Complete"
    return event_type not in {"Dispossessed", "Miscontrol"}


def normalize_event(
    event: Mapping[str, Any],
    *,
    competition_id: int,
    season_id: int,
    product_cohort: bool,
) -> dict[str, object]:
    """Flatten the verified event fields needed for deterministic possession analysis."""
    event_type = str(_entity(event.get("type"), "name") or UNKNOWN_CATEGORY)
    start_x, start_y = extract_xy(event.get("location"))
    payload = event.get(event_type.casefold().replace(" ", "_"))
    if event_type == "Ball Receipt*":
        payload = event.get("ball_receipt")
    end_x = end_y = None
    if isinstance(payload, Mapping):
        end_x, end_y = extract_xy(payload.get("end_location"))
    shot = event.get("shot") if isinstance(event.get("shot"), Mapping) else {}
    outcome = _entity(shot.get("outcome"), "name")
    return {
        "event_id": event.get("id"),
        "event_index": event.get("index"),
        "match_id": event.get("match_id"),
        "competition_id": competition_id,
        "season_id": season_id,
        "is_product_cohort": product_cohort,
        "possession_id": event.get("possession"),
        "possession_team_id": _entity(event.get("possession_team"), "id"),
        "possession_team_name": _entity(event.get("possession_team"), "name"),
        "player_id": _entity(event.get("player"), "id"),
        "player_name": _entity(event.get("player"), "name"),
        "team_id": _entity(event.get("team"), "id"),
        "team_name": _entity(event.get("team"), "name"),
        "event_type": event_type,
        "period": event.get("period"),
        "minute": event.get("minute"),
        "second": event.get("second"),
        "timestamp": event.get("timestamp"),
        "event_time_seconds": timestamp_seconds(event.get("timestamp"), event.get("period")),
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "under_pressure": bool(event.get("under_pressure", False)),
        "play_pattern": str(_entity(event.get("play_pattern"), "name") or UNKNOWN_CATEGORY),
        "event_success": event_success(event_type, event),
        "shot_goal": event_type == "Shot" and outcome == "Goal",
        "shot_outcome": outcome,
    }


def validate_feature_contract() -> None:
    leaked = KNOWN_ACTION_VALUE_LEAKAGE_COLUMNS.intersection(ACTION_VALUE_FEATURE_COLUMNS)
    if leaked:
        raise ValueError(f"Action-value leakage columns configured: {sorted(leaked)}")
    if len(ACTION_VALUE_FEATURE_COLUMNS) != len(set(ACTION_VALUE_FEATURE_COLUMNS)):
        raise ValueError("Action-value feature columns must be unique")


def validate_event_order(events: pd.DataFrame) -> None:
    """Prove unique, strictly increasing authoritative indices within each match."""
    required = {"event_id", "event_index", "match_id", "possession_id"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Events missing ordering columns: {sorted(missing)}")
    if events["event_id"].isna().any() or events["event_id"].duplicated().any():
        raise ValueError("Event IDs must be complete and globally unique")
    if events.duplicated(["match_id", "event_index"]).any():
        raise ValueError("StatsBomb event index must be unique within a match")
    ordered = events.sort_values(["match_id", "event_index"], kind="stable")
    if not ordered.groupby("match_id")["event_index"].apply(lambda s: s.is_monotonic_increasing).all():
        raise ValueError("StatsBomb event indices are not deterministic within matches")


def _future_window(values: np.ndarray, width: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    ends = np.minimum(np.arange(len(values)) + width, len(values))
    return cumulative[ends] - cumulative[np.arange(len(values))]


def build_possession_states(
    events: pd.DataFrame, shot_oof: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build pre-event states with OOF-xG-only future possession targets."""
    validate_feature_contract()
    validate_event_order(events)
    if shot_oof["shot_id"].duplicated().any() or shot_oof["expected_goal"].isna().any():
        raise ValueError("Shot OOF predictions must map one-to-one and be complete")
    ordered = events.sort_values(["match_id", "event_index"], kind="stable").reset_index(drop=True)
    shot_values = shot_oof.set_index("shot_id")["expected_goal"]
    eligible_shots = ordered.loc[
        ordered["event_type"].eq("Shot") & ordered["event_id"].isin(shot_values.index), "event_id"
    ]
    if len(eligible_shots) != len(shot_oof) or eligible_shots.nunique() != len(shot_oof):
        raise ValueError("Every OOF shot must map to exactly one normalized event")
    ordered["oof_shot_xg"] = ordered["event_id"].map(shot_values).fillna(0.0).astype(float)
    ordered["pass_ordinal"] = pd.NA
    ordered_pass_mask = ordered["event_type"].eq("Pass")
    ordered.loc[ordered_pass_mask, "pass_ordinal"] = (
        ordered.loc[ordered_pass_mask].groupby("match_id").cumcount().astype("Int64")
    )
    state_mask = (
        ordered["event_type"].isin(ON_BALL_EVENT_TYPES)
        & ordered["start_x"].notna()
        & ordered["start_y"].notna()
        & ordered["team_id"].eq(ordered["possession_team_id"])
        & ordered["possession_id"].notna()
    )
    states = ordered.loc[state_mask].copy()
    states = states.sort_values(["match_id", "possession_id", "event_index"], kind="stable")
    group_columns = ["match_id", "possession_id"]
    states["possession_action_number"] = states.groupby(group_columns).cumcount() + 1
    states["possession_start_x"] = states.groupby(group_columns)["start_x"].transform("first")
    states["possession_start_y"] = states.groupby(group_columns)["start_y"].transform("first")
    states["possession_start_time"] = states.groupby(group_columns)["event_time_seconds"].transform("first")
    states["possession_elapsed_time"] = (
        states["event_time_seconds"] - states["possession_start_time"]
    ).clip(lower=0)
    states["distance_progressed_from_possession_start"] = (
        states["start_x"] - states["possession_start_x"]
    )
    states["previous_action_type"] = (
        states.groupby(group_columns)["event_type"].shift().fillna("Possession Start")
    )
    states["previous_action_success"] = (
        states.groupby(group_columns)["event_success"]
        .shift()
        .astype("boolean")
        .fillna(True)
        .astype(bool)
    )
    states["ball_x"] = states["start_x"].astype(float)
    states["ball_y"] = states["start_y"].astype(float)
    states["distance_to_goal"] = [
        distance_to_goal(float(x), float(y)) for x, y in zip(states.ball_x, states.ball_y, strict=True)
    ]
    states["angle_to_goal"] = [
        angle_to_goal(float(x), float(y)) for x, y in zip(states.ball_x, states.ball_y, strict=True)
    ]
    states["current_play_pattern"] = states["play_pattern"].fillna(UNKNOWN_CATEGORY)

    # Score for the possession team, using only goals whose event index is earlier.
    score_difference = pd.Series(0, index=ordered.index, dtype=int)
    for _, match in ordered.groupby("match_id", sort=False):
        scoring = match["shot_goal"] | match["event_type"].eq("Own Goal For")
        total_before = scoring.astype(int).cumsum().shift(fill_value=0).to_numpy()
        for team_id in match["team_id"].dropna().astype(int).unique():
            team_scoring = scoring & match["team_id"].eq(team_id)
            team_before = team_scoring.astype(int).cumsum().shift(fill_value=0).to_numpy()
            perspective = match["possession_team_id"].eq(team_id).to_numpy()
            score_difference.loc[match.index[perspective]] = (
                2 * team_before[perspective] - total_before[perspective]
            )
    states["score_difference"] = states["event_id"].map(
        pd.Series(score_difference.to_numpy(), index=ordered["event_id"])
    ).astype(int)

    target_parts: list[pd.DataFrame] = []
    for _, group in states.groupby(group_columns, sort=False):
        values = group["oof_shot_xg"].to_numpy(dtype=float)
        part = pd.DataFrame(index=group.index)
        part[TARGET_COLUMN] = np.cumsum(values[::-1])[::-1]
        part["future_oof_xg_next_5"] = _future_window(values, 5)
        part["future_oof_xg_next_10"] = _future_window(values, 10)
        target_parts.append(part)
    targets = pd.concat(target_parts).sort_index()
    for column in HORIZON_COLUMNS.values():
        states[column] = targets[column]
    output_columns = STATE_METADATA_COLUMNS + [
        "under_pressure",
        "play_pattern",
        "event_time_seconds",
        "shot_goal",
        "shot_outcome",
        "oof_shot_xg",
    ] + ACTION_VALUE_FEATURE_COLUMNS + list(HORIZON_COLUMNS.values())
    output_columns = list(dict.fromkeys(output_columns))
    states = states[output_columns].reset_index(drop=True)
    if states["event_id"].duplicated().any():
        raise AssertionError("Every state must have one unique source event")
    return states, {
        "events": len(ordered),
        "states": len(states),
        "possessions": states[["match_id", "possession_id"]].drop_duplicates().shape[0],
        "mapped_oof_shots": len(eligible_shots),
    }


def horizon_audit(states: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    audit: dict[str, dict[str, float | int]] = {}
    for name, column in HORIZON_COLUMNS.items():
        values = states[column].astype(float)
        audit[name] = {
            "states": len(values),
            "zero_percentage": float(values.eq(0).mean()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
            "maximum": float(values.max()),
            "positive_states": int(values.gt(0).sum()),
        }
    return audit


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
