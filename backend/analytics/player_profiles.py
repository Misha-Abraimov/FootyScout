"""Aggregate out-of-fold pass predictions into player scouting profiles.

These metrics cover the available StatsBomb matches in the input, not
necessarily a player's complete domestic season.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "pass_oof_predictions.parquet"
)
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "data" / "processed" / "player_profiles.parquet"

LONG_PASS_LENGTH_THRESHOLD = 30.0
FINAL_THIRD_X = 80.0
MIN_OVERALL_ATTEMPTS = 100
MIN_SUBSET_ATTEMPTS = 20

REQUIRED_INPUT_COLUMNS = {
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
}

OUTPUT_COLUMNS = [
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "position",
    "matches_observed",
    "pass_attempts",
    "passes_completed",
    "actual_completion_rate",
    "expected_completions",
    "expected_completion_rate",
    "completions_above_expected",
    "completion_above_expected_pp",
    "pressure_attempts",
    "pressure_completed",
    "pressure_actual_completion_rate",
    "pressure_expected_completion_rate",
    "pressure_completions_above_expected",
    "pressure_above_expected_pp",
    "pressure_pass_rate",
    "progressive_attempts",
    "progressive_completed",
    "progressive_actual_completion_rate",
    "progressive_expected_completion_rate",
    "progressive_completions_above_expected",
    "progressive_above_expected_pp",
    "progressive_pass_rate",
    "long_pass_attempts",
    "long_pass_completed",
    "long_pass_actual_completion_rate",
    "long_pass_expected_completion_rate",
    "long_pass_completions_above_expected",
    "long_pass_above_expected_pp",
    "average_forward_distance",
    "net_forward_distance_per_100_passes",
    "positive_forward_distance_per_100_passes",
    "final_third_entries",
    "final_third_entries_per_100_passes",
    "overall_reliable",
    "pressure_reliable",
    "progressive_reliable",
    "long_pass_reliable",
]

RATE_COLUMNS = [
    "actual_completion_rate",
    "expected_completion_rate",
    "pressure_actual_completion_rate",
    "pressure_expected_completion_rate",
    "pressure_pass_rate",
    "progressive_actual_completion_rate",
    "progressive_expected_completion_rate",
    "progressive_pass_rate",
    "long_pass_actual_completion_rate",
    "long_pass_expected_completion_rate",
]


@dataclass(frozen=True)
class ProfileReport:
    input_passes: int
    player_profiles: int
    overall_reliable_players: int
    pressure_reliable_players: int
    progressive_reliable_players: int
    long_pass_reliable_players: int
    median_passes: float
    maximum_passes: int
    multi_team_players: int
    multi_position_players: int


def validate_oof_input(data: pd.DataFrame) -> None:
    """Reject incomplete, duplicated, or non-OOF pass predictions."""
    missing_columns = REQUIRED_INPUT_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"OOF input is missing columns: {sorted(missing_columns)}")
    if data["pass_index"].isna().any() or data["pass_index"].duplicated().any():
        raise ValueError("OOF input must contain one unique prediction per pass_index")
    if data["fold"].isna().any() or (data["fold"] < 1).any():
        raise ValueError("Every OOF input row must have a positive fold value")
    probabilities = data["expected_completion"]
    if probabilities.isna().any() or not probabilities.between(0.0, 1.0).all():
        raise ValueError("Every expected_completion must be a probability from 0 to 1")
    if data["player_id"].isna().any():
        raise ValueError("player_id cannot be missing for player aggregation")
    if not data["completed"].isin([0, 1]).all():
        raise ValueError("completed must contain only binary 0/1 values")


def _dominant_text(group: pd.DataFrame, column: str) -> object:
    """Return the event-count mode, breaking ties lexicographically."""
    counts = group.groupby(column, dropna=False).size().reset_index(name="count")
    counts["sort_value"] = counts[column].astype("string").fillna("")
    return counts.sort_values(
        ["count", "sort_value"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0][column]


def _dominant_team(group: pd.DataFrame) -> tuple[object, object]:
    """Return the most-observed team pair, then lowest ID/name on ties."""
    counts = (
        group.groupby(["team_id", "team_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(
            ["count", "team_id", "team_name"],
            ascending=[False, True, True],
            kind="stable",
        )
    )
    row = counts.iloc[0]
    return row["team_id"], row["team_name"]


def is_final_third_entry(start_x: object, end_x: object) -> bool:
    """Return whether a pass crosses x=80 into the attacking final third."""
    try:
        return float(start_x) < FINAL_THIRD_X <= float(end_x)
    except (TypeError, ValueError):
        return False


def _safe_rate(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else np.nan


def _subset_metrics(group: pd.DataFrame, mask: pd.Series, prefix: str) -> dict[str, float | int]:
    subset = group.loc[mask]
    attempts = len(subset)
    completed = int(subset["completed"].sum())
    expected = float(subset["expected_completion"].sum())
    actual_rate = _safe_rate(completed, attempts)
    expected_rate = _safe_rate(expected, attempts)
    above_expected = completed - expected if attempts else np.nan
    above_expected_pp = (
        100.0 * (actual_rate - expected_rate) if attempts else np.nan
    )
    return {
        f"{prefix}_attempts": attempts,
        f"{prefix}_completed": completed,
        f"{prefix}_actual_completion_rate": actual_rate,
        f"{prefix}_expected_completion_rate": expected_rate,
        f"{prefix}_completions_above_expected": above_expected,
        f"{prefix}_above_expected_pp": above_expected_pp,
    }


def _aggregate_player(player_id: object, group: pd.DataFrame) -> dict[str, object]:
    attempts = len(group)
    completed = int(group["completed"].sum())
    expected = float(group["expected_completion"].sum())
    actual_rate = completed / attempts
    expected_rate = expected / attempts
    team_id, team_name = _dominant_team(group)

    result: dict[str, object] = {
        "player_id": player_id,
        "player_name": _dominant_text(group, "player_name"),
        "team_id": team_id,
        "team_name": team_name,
        "position": _dominant_text(group, "position"),
        "matches_observed": int(group["match_id"].nunique()),
        "pass_attempts": attempts,
        "passes_completed": completed,
        "actual_completion_rate": actual_rate,
        "expected_completions": expected,
        "expected_completion_rate": expected_rate,
        "completions_above_expected": completed - expected,
        "completion_above_expected_pp": 100.0 * (actual_rate - expected_rate),
    }
    result.update(_subset_metrics(group, group["under_pressure"].eq(True), "pressure"))
    result["pressure_pass_rate"] = int(result["pressure_attempts"]) / attempts
    result.update(_subset_metrics(group, group["progressive"].eq(True), "progressive"))
    result["progressive_pass_rate"] = int(result["progressive_attempts"]) / attempts
    result.update(
        _subset_metrics(
            group,
            group["pass_length"].ge(LONG_PASS_LENGTH_THRESHOLD),
            "long_pass",
        )
    )

    forward_distance = pd.to_numeric(group["forward_distance"], errors="coerce")
    final_third_entries = int(
        (group["start_x"].lt(FINAL_THIRD_X) & group["end_x"].ge(FINAL_THIRD_X)).sum()
    )
    result.update(
        {
            "average_forward_distance": float(forward_distance.mean()),
            "net_forward_distance_per_100_passes": (
                float(forward_distance.sum()) / attempts * 100.0
            ),
            "positive_forward_distance_per_100_passes": (
                float(forward_distance.clip(lower=0).sum()) / attempts * 100.0
            ),
            "final_third_entries": final_third_entries,
            "final_third_entries_per_100_passes": final_third_entries / attempts * 100.0,
            "overall_reliable": attempts >= MIN_OVERALL_ATTEMPTS,
            "pressure_reliable": int(result["pressure_attempts"]) >= MIN_SUBSET_ATTEMPTS,
            "progressive_reliable": (
                int(result["progressive_attempts"]) >= MIN_SUBSET_ATTEMPTS
            ),
            "long_pass_reliable": int(result["long_pass_attempts"]) >= MIN_SUBSET_ATTEMPTS,
        }
    )
    return result


def validate_profiles(profiles: pd.DataFrame, source: pd.DataFrame) -> None:
    """Verify conservation, bounds, null handling, and reliability rules."""
    if list(profiles.columns) != OUTPUT_COLUMNS:
        raise AssertionError("Player-profile output columns do not match the schema")
    if profiles["player_id"].duplicated().any():
        raise AssertionError("Player profiles must be unique by player_id")
    if int(profiles["pass_attempts"].sum()) != len(source):
        raise AssertionError("Player pass attempts do not sum to the OOF input row count")
    if int(profiles["passes_completed"].sum()) != int(source["completed"].sum()):
        raise AssertionError("Player completed passes do not match the OOF input")
    weighted_expected = np.average(
        profiles["expected_completion_rate"],
        weights=profiles["pass_attempts"],
    )
    if not np.isclose(weighted_expected, source["expected_completion"].mean()):
        raise AssertionError("Weighted player expected completion does not match pass-level mean")
    for column in RATE_COLUMNS:
        values = profiles[column].dropna()
        if not values.between(0.0, 1.0).all():
            raise AssertionError(f"Rate outside [0, 1] in {column}")

    expected_flags = {
        "overall_reliable": profiles["pass_attempts"].ge(MIN_OVERALL_ATTEMPTS),
        "pressure_reliable": profiles["pressure_attempts"].ge(MIN_SUBSET_ATTEMPTS),
        "progressive_reliable": profiles["progressive_attempts"].ge(MIN_SUBSET_ATTEMPTS),
        "long_pass_reliable": profiles["long_pass_attempts"].ge(MIN_SUBSET_ATTEMPTS),
    }
    for column, expected in expected_flags.items():
        if not profiles[column].equals(expected):
            raise AssertionError(f"Reliability flag does not match its threshold: {column}")

    for prefix in ("pressure", "progressive", "long_pass"):
        zero_attempts = profiles[f"{prefix}_attempts"].eq(0)
        nullable_columns = [
            f"{prefix}_actual_completion_rate",
            f"{prefix}_expected_completion_rate",
            f"{prefix}_completions_above_expected",
            f"{prefix}_above_expected_pp",
        ]
        if profiles.loc[zero_attempts, nullable_columns].notna().any().any():
            raise AssertionError(f"Zero-attempt {prefix} metrics must be null")


def build_player_profiles(data: pd.DataFrame) -> tuple[pd.DataFrame, ProfileReport]:
    """Build one profile per player from valid out-of-fold predictions."""
    validate_oof_input(data)
    multi_team_players = int((data.groupby("player_id")["team_id"].nunique() > 1).sum())
    multi_position_players = int(
        (data.groupby("player_id")["position"].nunique() > 1).sum()
    )
    records = [
        _aggregate_player(player_id, group)
        for player_id, group in data.groupby("player_id", sort=True, observed=True)
    ]
    profiles = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    validate_profiles(profiles, data)
    report = ProfileReport(
        input_passes=len(data),
        player_profiles=len(profiles),
        overall_reliable_players=int(profiles["overall_reliable"].sum()),
        pressure_reliable_players=int(profiles["pressure_reliable"].sum()),
        progressive_reliable_players=int(profiles["progressive_reliable"].sum()),
        long_pass_reliable_players=int(profiles["long_pass_reliable"].sum()),
        median_passes=float(profiles["pass_attempts"].median()),
        maximum_passes=int(profiles["pass_attempts"].max()),
        multi_team_players=multi_team_players,
        multi_position_players=multi_position_players,
    )
    return profiles, report


def print_report(profiles: pd.DataFrame, report: ProfileReport) -> None:
    print("Player-profile generation complete")
    print(f"Available-match passes: {report.input_passes:,}")
    print(f"Total players: {report.player_profiles:,}")
    print(f"Players with at least {MIN_OVERALL_ATTEMPTS} passes: {report.overall_reliable_players:,}")
    print(f"Median passes per player: {report.median_passes:,.1f}")
    print(f"Maximum passes for one player: {report.maximum_passes:,}")
    print(f"Players reliable for pressure metrics: {report.pressure_reliable_players:,}")
    print(f"Players reliable for progressive metrics: {report.progressive_reliable_players:,}")
    print(f"Players reliable for long-pass metrics: {report.long_pass_reliable_players:,}")
    print(f"Players observed for multiple teams: {report.multi_team_players:,}")
    print(f"Players observed in multiple positions: {report.multi_position_players:,}")
    print("Sanity-check preview only (not a definitive ranking):")
    preview = (
        profiles.loc[profiles["overall_reliable"]]
        .nlargest(5, "completion_above_expected_pp")
        .loc[
            :,
            [
                "player_name",
                "team_name",
                "matches_observed",
                "pass_attempts",
                "actual_completion_rate",
                "expected_completion_rate",
                "completion_above_expected_pp",
            ],
        ]
    )
    print(preview.to_string(index=False))


def run_pipeline(input_path: Path, output_path: Path) -> tuple[pd.DataFrame, ProfileReport]:
    data = pd.read_parquet(input_path)
    profiles, report = build_player_profiles(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(output_path, index=False)
    print_report(profiles, report)
    print(f"Saved profiles: {output_path}")
    return profiles, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
