"""Create pre-outcome features for the expected-pass-completion model.

StatsBomb locations use a 120 x 80 coordinate system. Events are normalized so
the team in possession attacks toward increasing x, and the opponent's goal is
treated as the point (120, 40).
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_Y = 40.0
PROGRESSIVE_DISTANCE_REDUCTION = 0.25
UNKNOWN_CATEGORY = "Unknown"
REGULAR_PASS_TYPE = "Regular"
CATEGORICAL_DEFAULTS = {
    "position": UNKNOWN_CATEGORY,
    "pass_height": UNKNOWN_CATEGORY,
    "body_part": UNKNOWN_CATEGORY,
    # StatsBomb pass.type is an optional special-pass qualifier. If it is
    # absent, the pass is an ordinary/default pass rather than unknown.
    "pass_type": REGULAR_PASS_TYPE,
}

DEFAULT_INPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "passes.parquet"
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "pass_features.parquet"
)

METADATA_COLUMNS = [
    "match_id",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "position",
]

NUMERIC_FEATURE_COLUMNS = [
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "pass_length",
    "pass_angle",
    "forward_distance",
    "lateral_distance",
    "distance_to_goal_before",
    "distance_to_goal_after",
    "distance_toward_goal",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "under_pressure",
    "pass_height",
    "body_part",
    "pass_type",
    "start_zone",
    "end_zone",
    "progressive",
]

MODEL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
TARGET_COLUMN = "completed"
OUTPUT_COLUMNS = METADATA_COLUMNS + MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]

KNOWN_LEAKAGE_COLUMNS = frozenset(
    {
        "pass_outcome",
        "completed",
        "match_id",
        "player_id",
        "player_name",
        "team_id",
        "team_name",
        "position",
        "pass_recipient",
        "pass_recipient_id",
        "pass_recipient_name",
        "recipient_id",
        "recipient_name",
        "next_event",
        "next_event_type",
    }
)

REQUIRED_INPUT_COLUMNS = set(METADATA_COLUMNS) | {
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "pass_length",
    "pass_angle",
    "under_pressure",
    "pass_height",
    "body_part",
    "pass_type",
    TARGET_COLUMN,
}


@dataclass(frozen=True)
class FeatureEngineeringReport:
    rows_before: int
    rows_after: int
    rows_removed: int
    removal_reason: str
    categorical_values_filled: dict[str, int]
    pressure_values_filled: int


def _finite_number(value: object) -> float | None:
    if value is None or value is pd.NA or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def calculate_forward_distance(start_x: object, end_x: object) -> float | None:
    """Return signed movement toward the opponent's goal along the x-axis."""
    start = _finite_number(start_x)
    end = _finite_number(end_x)
    return None if start is None or end is None else end - start


def calculate_lateral_distance(start_y: object, end_y: object) -> float | None:
    """Return absolute movement across the width of the pitch."""
    start = _finite_number(start_y)
    end = _finite_number(end_y)
    return None if start is None or end is None else abs(end - start)


def calculate_distance_to_goal(x: object, y: object) -> float | None:
    """Return straight-line distance from a location to goal centre (120, 40)."""
    x_value = _finite_number(x)
    y_value = _finite_number(y)
    if x_value is None or y_value is None:
        return None
    return math.hypot(GOAL_X - x_value, GOAL_Y - y_value)


def is_progressive_pass(
    distance_before: object,
    distance_after: object,
    threshold: float = PROGRESSIVE_DISTANCE_REDUCTION,
) -> bool:
    """Return whether a pass reduces distance to goal by at least ``threshold``.

    FootyScout V1 uses a 25% reduction threshold. A pass is progressive when
    ``distance_after <= 0.75 * distance_before``. Passes beginning at the goal
    centre or with missing/non-finite distances are not classified as progressive.
    """
    if not 0 < threshold < 1:
        raise ValueError("Progressive-pass threshold must be between 0 and 1")

    before = _finite_number(distance_before)
    after = _finite_number(distance_after)
    if before is None or after is None or before <= 0:
        return False
    return (before - after) / before >= threshold


def classify_pitch_zone(x: object, y: object) -> str:
    """Classify a location in an equal 3-by-3 pitch grid.

    Longitudinal bands are defensive [0, 40), middle [40, 80), and attacking
    [80, 120]. Lateral bands relative to the attacking direction are left
    [0, 80/3), centre [80/3, 160/3), and right [160/3, 80]. Missing or
    out-of-bounds locations are classified as ``Unknown``.
    """
    x_value = _finite_number(x)
    y_value = _finite_number(y)
    if (
        x_value is None
        or y_value is None
        or not 0 <= x_value <= PITCH_LENGTH
        or not 0 <= y_value <= PITCH_WIDTH
    ):
        return UNKNOWN_CATEGORY

    if x_value < PITCH_LENGTH / 3:
        longitudinal = "defensive"
    elif x_value < 2 * PITCH_LENGTH / 3:
        longitudinal = "middle"
    else:
        longitudinal = "attacking"

    if y_value < PITCH_WIDTH / 3:
        lateral = "left"
    elif y_value < 2 * PITCH_WIDTH / 3:
        lateral = "centre"
    else:
        lateral = "right"

    return f"{longitudinal}_{lateral}"


def normalize_category(value: object, default: str = UNKNOWN_CATEGORY) -> str:
    """Normalize a categorical scalar, replacing null/blank values with a default."""
    if value is None or value is pd.NA:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    normalized = str(value).strip()
    return normalized or default


def validate_model_feature_columns(columns: Sequence[str] = MODEL_FEATURE_COLUMNS) -> None:
    """Fail fast if metadata, target, or known post-outcome fields enter the model."""
    duplicates = {column for column in columns if columns.count(column) > 1}
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate model feature columns: {names}")

    leaked = KNOWN_LEAKAGE_COLUMNS.intersection(columns)
    if leaked:
        names = ", ".join(sorted(leaked))
        raise ValueError(f"Target leakage detected in model feature columns: {names}")


def _validate_input(passes: pd.DataFrame) -> None:
    missing_columns = REQUIRED_INPUT_COLUMNS.difference(passes.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Pass dataset is missing required columns: {names}")

    target = pd.to_numeric(passes[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("The completed target must contain only non-null binary values 0 and 1")


def engineer_pass_features(
    passes: pd.DataFrame,
) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
    """Transform normalized passes into a leakage-safe feature dataset."""
    validate_model_feature_columns()
    _validate_input(passes)

    rows_before = len(passes)
    features = passes.copy()

    for column in [
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "pass_length",
        "pass_angle",
    ]:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("Float64")

    features["forward_distance"] = [
        calculate_forward_distance(start, end)
        for start, end in zip(features["start_x"], features["end_x"], strict=True)
    ]
    features["lateral_distance"] = [
        calculate_lateral_distance(start, end)
        for start, end in zip(features["start_y"], features["end_y"], strict=True)
    ]
    features["distance_to_goal_before"] = [
        calculate_distance_to_goal(x, y)
        for x, y in zip(features["start_x"], features["start_y"], strict=True)
    ]
    features["distance_to_goal_after"] = [
        calculate_distance_to_goal(x, y)
        for x, y in zip(features["end_x"], features["end_y"], strict=True)
    ]
    features["distance_toward_goal"] = (
        features["distance_to_goal_before"] - features["distance_to_goal_after"]
    )
    features["progressive"] = [
        is_progressive_pass(before, after)
        for before, after in zip(
            features["distance_to_goal_before"],
            features["distance_to_goal_after"],
            strict=True,
        )
    ]
    features["start_zone"] = [
        classify_pitch_zone(x, y)
        for x, y in zip(features["start_x"], features["start_y"], strict=True)
    ]
    features["end_zone"] = [
        classify_pitch_zone(x, y)
        for x, y in zip(features["end_x"], features["end_y"], strict=True)
    ]

    categorical_values_filled: dict[str, int] = {}
    for column, default in CATEGORICAL_DEFAULTS.items():
        missing = features[column].isna() | features[column].astype("string").str.strip().eq("")
        categorical_values_filled[column] = int(missing.fillna(True).sum())
        features[column] = features[column].map(
            lambda value, fallback=default: normalize_category(value, fallback)
        )
        features[column] = features[column].astype("string")

    pressure_missing = int(features["under_pressure"].isna().sum())
    features["under_pressure"] = features["under_pressure"].fillna(False).astype(bool)
    features["progressive"] = features["progressive"].astype(bool)
    features[TARGET_COLUMN] = pd.to_numeric(features[TARGET_COLUMN]).astype("Int8")

    for column in [
        "forward_distance",
        "lateral_distance",
        "distance_to_goal_before",
        "distance_to_goal_after",
        "distance_toward_goal",
    ]:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("Float64")

    result = features.loc[:, OUTPUT_COLUMNS].copy()
    rows_after = len(result)
    report = FeatureEngineeringReport(
        rows_before=rows_before,
        rows_after=rows_after,
        rows_removed=rows_before - rows_after,
        removal_reason="No rows removed; missing categories are retained with documented defaults.",
        categorical_values_filled=categorical_values_filled,
        pressure_values_filled=pressure_missing,
    )
    return result, report


def save_feature_dataset(features: pd.DataFrame, output_path: Path) -> None:
    """Write the feature dataset atomically as Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        features.to_parquet(temporary_path, engine="pyarrow", index=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Read normalized passes, build features, save them, and print a report."""
    passes = pd.read_parquet(input_path, engine="pyarrow")
    features, report = engineer_pass_features(passes)
    save_feature_dataset(features, output_path)

    print("Pass feature engineering complete")
    print(f"Input rows: {report.rows_before:,}")
    print(f"Output rows: {report.rows_after:,}")
    print(f"Rows removed: {report.rows_removed:,}")
    print(f"Removal policy: {report.removal_reason}")
    for column, count in report.categorical_values_filled.items():
        print(f"Missing {column} values filled with {CATEGORICAL_DEFAULTS[column]}: {count:,}")
    print(f"Missing under_pressure values filled with False: {report.pressure_values_filled:,}")
    print(f"Progressive passes: {int(features['progressive'].sum()):,}")
    print(f"Saved dataset: {output_path}")
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
