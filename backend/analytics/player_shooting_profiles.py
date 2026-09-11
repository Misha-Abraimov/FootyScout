"""Player-level non-penalty shooting aggregates for the product cohort."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MIN_RELIABLE_SHOTS = 20
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHOTS_PATH = ROOT / "data" / "processed" / "product_shot_predictions.parquet"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "processed" / "player_shooting_profiles.parquet"

PROFILE_COLUMNS = [
    "player_id",
    "shots",
    "goals",
    "total_xg",
    "xg_per_shot",
    "goals_minus_xg",
    "goals_per_shot",
    "matches_observed",
    "shooting_reliable",
]


def build_player_shooting_profiles(shots: pd.DataFrame) -> pd.DataFrame:
    """Aggregate modeled, non-penalty shots without discarding small samples."""
    required = {"player_id", "match_id", "goal", "expected_goal", "model_eligible"}
    missing = required.difference(shots.columns)
    if missing:
        raise ValueError(f"Product shots are missing columns: {sorted(missing)}")
    eligible = shots.loc[shots["model_eligible"] & shots["player_id"].notna()].copy()
    if eligible["expected_goal"].isna().any():
        raise ValueError("Every eligible product shot must have expected_goal")
    grouped = eligible.groupby("player_id", sort=True, observed=True)
    profile = grouped.agg(
        shots=("shot_id", "size"),
        goals=("goal", "sum"),
        total_xg=("expected_goal", "sum"),
        matches_observed=("match_id", "nunique"),
    ).reset_index()
    profile["xg_per_shot"] = profile["total_xg"] / profile["shots"]
    profile["goals_minus_xg"] = profile["goals"] - profile["total_xg"]
    profile["goals_per_shot"] = profile["goals"] / profile["shots"]
    profile["shooting_reliable"] = profile["shots"] >= MIN_RELIABLE_SHOTS
    return profile[PROFILE_COLUMNS]


def run(shots_path: Path = DEFAULT_SHOTS_PATH, output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    shots = pd.read_parquet(shots_path)
    profiles = build_player_shooting_profiles(shots)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(output_path, index=False)
    print(f"Player shooting profiles: {len(profiles):,}")
    print(f"Reliable ({MIN_RELIABLE_SHOTS}+ shots): {int(profiles['shooting_reliable'].sum()):,}")
    print(f"Saved: {output_path}")
    return profiles

