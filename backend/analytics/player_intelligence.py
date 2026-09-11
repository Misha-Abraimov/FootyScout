"""Build V3.1 unified player metrics, position percentiles, and style matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analytics.player_feature_registry import (
    FEATURE_REGISTRY,
    MIN_POSITION_PEERS,
    PERFORMANCE_FEATURES,
    STYLE_CLUSTERING_FEATURES,
    STYLE_FEATURES,
    registry_records,
    validate_registry,
)
from analytics.player_similarity import POSITION_GROUP_MAPPING, broad_position_group

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_PASSING_PATH = PROCESSED / "player_profiles.parquet"
DEFAULT_SHOOTING_PATH = PROCESSED / "player_shooting_profiles.parquet"
DEFAULT_ATTACKING_PATH = PROCESSED / "player_attacking_profiles.parquet"
DEFAULT_ACTIONS_PATH = PROCESSED / "attacking_actions.parquet"
DEFAULT_PROFILE_OUTPUT = PROCESSED / "player_intelligence_profiles.parquet"
DEFAULT_PERCENTILE_OUTPUT = PROCESSED / "player_percentiles.parquet"
DEFAULT_STYLE_OUTPUT = PROCESSED / "player_style_features.parquet"
DEFAULT_AUDIT_OUTPUT = ROOT / "models" / "player_intelligence_metadata.json"

PROFILE_METADATA = [
    "player_id", "player_name", "team_id", "team_name", "position",
    "position_group", "matches_observed",
]
SAMPLE_COLUMNS = [
    "pass_attempts", "pressure_attempts", "progressive_attempts",
    "long_pass_attempts", "shots", "actions", "passes", "carries",
]
RELIABILITY_COLUMNS = [
    "overall_reliable", "pressure_reliable", "progressive_reliable",
    "long_pass_reliable", "shooting_reliable", "attacking_value_reliable",
    "pass_value_reliable", "carry_value_reliable",
]
PERCENTILE_COLUMNS = [
    "player_id", "metric_name", "family", "raw_value", "percentile",
    "peer_position_group", "peer_count", "sample_count", "eligible",
    "eligibility_reason",
]


def _require_unique(frame: pd.DataFrame, name: str) -> None:
    if "player_id" not in frame or frame["player_id"].isna().any():
        raise ValueError(f"{name} requires non-null player_id")
    if frame["player_id"].duplicated().any():
        raise ValueError(f"{name} must contain one row per player_id")


def _action_tendencies(actions: pd.DataFrame, player_ids: set[int]) -> pd.DataFrame:
    required = {"player_id", "action_type", "progressive", "under_pressure"}
    missing = required.difference(actions.columns)
    if missing:
        raise ValueError(f"Attacking actions are missing columns: {sorted(missing)}")
    source = actions.loc[actions["player_id"].isin(player_ids)].copy()
    grouped = source.groupby("player_id", sort=True, observed=True)
    totals = grouped.agg(
        derived_actions=("action_type", "size"),
        progressive_actions=("progressive", "sum"),
        pressure_actions=("under_pressure", "sum"),
    )
    carries = source.loc[source["action_type"].eq("Carry")].groupby("player_id").agg(
        derived_carries=("action_type", "size"),
        progressive_carries=("progressive", "sum"),
    )
    return totals.join(carries, how="left").reset_index()


def build_unified_profiles(
    passing: pd.DataFrame,
    shooting: pd.DataFrame,
    attacking: pd.DataFrame,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join frozen V1/V2 aggregates onto the 369-player product cohort."""
    validate_registry()
    for frame, name in ((passing, "passing"), (shooting, "shooting"), (attacking, "attacking")):
        _require_unique(frame, name)
    required_passing = set(PROFILE_METADATA).difference({"position_group"}) | {
        "pass_attempts", "pressure_attempts", "progressive_attempts", "long_pass_attempts",
        "overall_reliable", "pressure_reliable", "progressive_reliable", "long_pass_reliable",
        *[f.feature_name for f in FEATURE_REGISTRY if f.source == "player_profiles"],
    }
    missing = required_passing.difference(passing.columns)
    if missing:
        raise ValueError(f"Passing profiles are missing columns: {sorted(missing)}")

    profile = passing.copy()
    profile.insert(5, "position_group", profile["position"].map(broad_position_group))
    player_ids = set(profile["player_id"].astype(int))

    shooting_columns = [
        "player_id", "shots", "goals", "total_xg", "xg_per_shot", "goals_minus_xg",
        "shooting_reliable",
    ]
    profile = profile.merge(
        shooting[shooting_columns], on="player_id", how="left", validate="one_to_one"
    )
    for column in ("shots", "goals", "total_xg", "goals_minus_xg"):
        profile[column] = profile[column].fillna(0)
    profile["shooting_reliable"] = profile["shooting_reliable"].eq(True)

    attacking_columns = [
        "player_id", "actions", "passes", "carries", "attacking_value_per_100_actions",
        "pass_value_per_100_passes", "carry_value_per_100_carries",
        "progressive_value_per_100_actions", "pressure_value_per_100_actions",
        "attacking_value_reliable", "pass_value_reliable", "carry_value_reliable",
    ]
    profile = profile.merge(
        attacking[attacking_columns], on="player_id", how="left", validate="one_to_one"
    )
    tendencies = _action_tendencies(actions, player_ids)
    profile = profile.merge(tendencies, on="player_id", how="left", validate="one_to_one")
    for column in ("actions", "passes", "carries", "derived_actions", "derived_carries", "progressive_actions", "pressure_actions", "progressive_carries"):
        profile[column] = profile[column].fillna(0).astype(int)
    for column in ("attacking_value_reliable", "pass_value_reliable", "carry_value_reliable"):
        profile[column] = profile[column].eq(True)
    has_actions = profile["actions"].gt(0)
    if not profile.loc[has_actions, "actions"].equals(profile.loc[has_actions, "derived_actions"]):
        raise ValueError("Action counts do not align with frozen attacking profiles")
    if not profile.loc[has_actions, "carries"].equals(profile.loc[has_actions, "derived_carries"]):
        raise ValueError("Carry counts do not align with frozen attacking profiles")

    profile["long_pass_rate"] = profile["long_pass_attempts"] / profile["pass_attempts"]
    profile["carry_share_of_actions"] = profile["carries"] / profile["actions"].replace(0, np.nan)
    profile["progressive_carry_rate"] = (
        profile["progressive_carries"] / profile["carries"].replace(0, np.nan)
    )
    profile["progressive_action_rate"] = (
        profile["progressive_actions"] / profile["actions"].replace(0, np.nan)
    )
    profile["pressure_action_rate"] = (
        profile["pressure_actions"] / profile["actions"].replace(0, np.nan)
    )
    profile["shots_per_match_observed"] = profile["shots"] / profile["matches_observed"]

    columns = PROFILE_METADATA + SAMPLE_COLUMNS + RELIABILITY_COLUMNS + STYLE_FEATURES + PERFORMANCE_FEATURES
    result = profile[columns].copy().sort_values("player_id", kind="stable").reset_index(drop=True)
    if result["player_id"].duplicated().any() or len(result) != len(passing):
        raise AssertionError("Unified profiles must preserve exactly one row per product player")
    return result


def build_percentiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Calculate tie-aware empirical ranks within eligible position peers.

    Formula: ``100 * (average_rank - 1) / (n - 1)``. Ties receive their average
    ascending rank. Cohorts smaller than ``MIN_POSITION_PEERS`` remain unavailable.
    """
    records: list[dict[str, object]] = []
    for feature in FEATURE_REGISTRY:
        for position_group, group in profiles.groupby("position_group", sort=True):
            raw = pd.to_numeric(group[feature.feature_name], errors="coerce")
            samples = pd.to_numeric(group[feature.sample_column], errors="coerce").fillna(0)
            sample_eligible = raw.notna() & samples.ge(feature.minimum_sample)
            peer_count = int(sample_eligible.sum())
            ranked = pd.Series(np.nan, index=group.index, dtype=float)
            if peer_count >= MIN_POSITION_PEERS:
                ranks = raw.loc[sample_eligible].rank(method="average", ascending=True)
                ranked.loc[sample_eligible] = 100.0 * (ranks - 1.0) / (peer_count - 1.0)
            for index in group.index:
                value = raw.loc[index]
                sample_count = int(samples.loc[index])
                if pd.isna(value):
                    reason = "metric_unavailable"
                    eligible = False
                elif sample_count < feature.minimum_sample:
                    reason = f"requires_{feature.minimum_sample}_{feature.required_sample_type}"
                    eligible = False
                elif peer_count < MIN_POSITION_PEERS:
                    reason = f"requires_{MIN_POSITION_PEERS}_eligible_position_peers"
                    eligible = False
                else:
                    reason = "eligible"
                    eligible = True
                records.append({
                    "player_id": int(group.at[index, "player_id"]),
                    "metric_name": feature.feature_name,
                    "family": feature.family,
                    "raw_value": None if pd.isna(value) else float(value),
                    "percentile": None if pd.isna(ranked.loc[index]) else float(ranked.loc[index]),
                    "peer_position_group": str(position_group),
                    "peer_count": peer_count,
                    "sample_count": sample_count,
                    "eligible": eligible,
                    "eligibility_reason": reason,
                })
    result = pd.DataFrame.from_records(records, columns=PERCENTILE_COLUMNS)
    if result.duplicated(["player_id", "metric_name"]).any():
        raise AssertionError("Percentiles must be unique per player and metric")
    values = result["percentile"].dropna()
    if not values.between(0, 100).all():
        raise AssertionError("Percentiles must stay within [0, 100]")
    return result.sort_values(["player_id", "metric_name"], kind="stable").reset_index(drop=True)


def build_style_matrix(profiles: pd.DataFrame, percentiles: pd.DataFrame) -> pd.DataFrame:
    """Create an unscaled, style-only matrix with explicit sample eligibility."""
    columns = ["player_id", "position_group", *STYLE_CLUSTERING_FEATURES]
    matrix = profiles[columns].copy()
    lookup = percentiles.set_index(["player_id", "metric_name"])
    for feature_name in STYLE_CLUSTERING_FEATURES:
        matrix[f"{feature_name}__sample_count"] = [
            int(lookup.at[(player_id, feature_name), "sample_count"])
            for player_id in matrix["player_id"]
        ]
        matrix[f"{feature_name}__eligible"] = [
            bool(lookup.at[(player_id, feature_name), "eligible"])
            for player_id in matrix["player_id"]
        ]
    forbidden = set(PERFORMANCE_FEATURES) & set(matrix.columns)
    if forbidden:
        raise AssertionError(f"Performance features entered style matrix: {sorted(forbidden)}")
    return matrix


def _distribution(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    return {
        "count": len(values), "missing_percent": float(100 * series.isna().mean()),
        "mean": float(values.mean()), "std": float(values.std()), "median": float(values.median()),
        "p5": float(values.quantile(0.05)), "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)), "p95": float(values.quantile(0.95)),
        "minimum": float(values.min()), "maximum": float(values.max()),
    }


def build_audit(
    profiles: pd.DataFrame,
    percentiles: pd.DataFrame,
    source_frames: dict[str, pd.DataFrame],
    style_matrix: pd.DataFrame,
) -> dict[str, Any]:
    metrics = STYLE_FEATURES + PERFORMANCE_FEATURES
    correlations = profiles[metrics].corr(method="pearson", min_periods=2)
    high = []
    for index, left in enumerate(metrics):
        for right in metrics[index + 1:]:
            value = correlations.at[left, right]
            if pd.notna(value) and abs(float(value)) >= 0.9:
                high.append({"left": left, "right": right, "pearson": float(value)})
    complete = {}
    for group, rows in style_matrix.groupby("position_group"):
        flags = [f"{name}__eligible" for name in STYLE_CLUSTERING_FEATURES]
        complete[str(group)] = int(rows[flags].all(axis=1).sum())
    availability: dict[str, dict[str, int]] = {}
    for feature in FEATURE_REGISTRY:
        metric_rows = percentiles.loc[percentiles["metric_name"].eq(feature.feature_name)]
        availability[feature.feature_name] = {
            group: int(metric_rows.loc[metric_rows["peer_position_group"].eq(group), "percentile"].notna().sum())
            for group in ("GK", "DEF", "MID", "FWD")
        }
    return {
        "version": "V3.1",
        "definitions": {
            "raw_metric": "Observed or model-derived player statistic.",
            "percentile": "Relative position among eligible peers in the same broad position group; not a rating.",
            "style": "Observed tendency or behavior; a high percentile is not automatically better.",
            "performance": "Observed execution or model-derived output relative to opportunities.",
        },
        "sources": {name: {"rows": len(frame), "columns": list(frame.columns)} for name, frame in source_frames.items()},
        "players": len(profiles),
        "position_groups": profiles["position_group"].value_counts().reindex(["GK", "DEF", "MID", "FWD"], fill_value=0).astype(int).to_dict(),
        "position_mapping": POSITION_GROUP_MAPPING,
        "sample_distributions": {column: _distribution(profiles[column]) for column in SAMPLE_COLUMNS},
        "reliability_counts": {column: int(profiles[column].sum()) for column in RELIABILITY_COLUMNS},
        "feature_registry": registry_records(),
        "feature_distributions": {metric: _distribution(profiles[metric]) for metric in metrics},
        "highly_correlated_pairs": high,
        "percentile_algorithm": "Within position_group, ascending average tie rank: 100 * (average_rank - 1) / (eligible_peer_count - 1).",
        "minimum_position_peers": MIN_POSITION_PEERS,
        "percentile_availability": availability,
        "style_matrix_features": STYLE_CLUSTERING_FEATURES,
        "complete_eligible_style_cohorts": {group: complete.get(group, 0) for group in ("GK", "DEF", "MID", "FWD")},
    }


def run_pipeline(
    passing_path: Path = DEFAULT_PASSING_PATH,
    shooting_path: Path = DEFAULT_SHOOTING_PATH,
    attacking_path: Path = DEFAULT_ATTACKING_PATH,
    actions_path: Path = DEFAULT_ACTIONS_PATH,
    profile_output: Path = DEFAULT_PROFILE_OUTPUT,
    percentile_output: Path = DEFAULT_PERCENTILE_OUTPUT,
    style_output: Path = DEFAULT_STYLE_OUTPUT,
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sources = {
        "player_profiles": pd.read_parquet(passing_path),
        "player_shooting_profiles": pd.read_parquet(shooting_path),
        "player_attacking_profiles": pd.read_parquet(attacking_path),
        "attacking_actions": pd.read_parquet(actions_path),
    }
    profiles = build_unified_profiles(*sources.values())
    percentiles = build_percentiles(profiles)
    style = build_style_matrix(profiles, percentiles)
    audit = build_audit(profiles, percentiles, sources, style)
    for path in (profile_output, percentile_output, style_output, audit_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(profile_output, index=False)
    percentiles.to_parquet(percentile_output, index=False)
    style.to_parquet(style_output, index=False)
    audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Unified player profiles: {len(profiles):,}")
    print(f"Percentile rows: {len(percentiles):,}")
    print(f"Position groups: {audit['position_groups']}")
    print(f"Complete eligible style cohorts: {audit['complete_eligible_style_cohorts']}")
    return profiles, percentiles, style, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passing", type=Path, default=DEFAULT_PASSING_PATH)
    parser.add_argument("--shooting", type=Path, default=DEFAULT_SHOOTING_PATH)
    parser.add_argument("--attacking", type=Path, default=DEFAULT_ATTACKING_PATH)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.passing, args.shooting, args.attacking, args.actions)


if __name__ == "__main__":
    main()
