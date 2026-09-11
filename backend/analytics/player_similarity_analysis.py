"""Research and validate V3.3 player-style similarity candidates.

This historical V3.3A module is deliberately analysis-only. It reconstructs
the former V1 methodology for comparison, writes only research outputs, and
never overwrites production similarity artifacts or changes the API contract.
Similarity is computed from pre-outcome playing-style tendencies, never player
quality, outcomes, archetype labels, or identity features.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analytics.artifact_paths import repository_relative_path
from analytics.player_feature_registry import PERFORMANCE_FEATURES
from analytics.player_style_stability import (
    assign_player_match_halves,
    build_match_chronology,
    recompute_style_features,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_PASSES = PROCESSED / "pass_oof_predictions.parquet"
DEFAULT_ACTIONS = PROCESSED / "attacking_actions.parquet"
DEFAULT_PROFILES = PROCESSED / "player_intelligence_profiles.parquet"
DEFAULT_MATCHES = PROCESSED / "product_match_chronology.parquet"
DEFAULT_ARCHETYPES = PROCESSED / "player_archetypes.parquet"
DEFAULT_FEATURE_OUTPUT = PROCESSED / "player_similarity_features_v3.parquet"
DEFAULT_ANALYSIS_OUTPUT = PROCESSED / "player_similarity_analysis_v3.parquet"
DEFAULT_NEIGHBOR_OUTPUT = PROCESSED / "player_similarity_neighbors_v3_research.parquet"
DEFAULT_METADATA_OUTPUT = ROOT / "models" / "player_similarity_analysis_v3.json"

VERSION = "V3.3A"
PASS_THRESHOLD = 50
CARRY_THRESHOLD = 29
OUTFIELD_GROUPS = ("DEF", "MID", "FWD")
POSITION_GROUPS = ("GK", *OUTFIELD_GROUPS)
DISTANCE_METHODS = ("rms_euclidean", "cosine", "mean_manhattan")
RESAMPLE_COUNT = 100
RESAMPLE_FRACTION = 0.8
RANDOM_STATE = 42
TOP_K_VALUES = (5, 6, 10)
SCORE_LN2 = float(np.log(2.0))

# Frozen adjacent-position fallback used only by the historical V3.3A comparison arm.
# Production V3.3B similarity is strictly same-position and does not import this policy.
ADJACENT_OUTFIELD_GROUPS = {
    "DEF": ("MID",),
    "MID": ("DEF", "FWD"),
    "FWD": ("MID",),
    "GK": (),
}

# Frozen legacy contract used only for the V3.3A V1 comparison arm.
V1_FEATURES = (
    "expected_completion_rate",
    "pressure_pass_rate",
    "progressive_pass_rate",
    "long_pass_rate",
    "average_forward_distance",
    "positive_forward_distance_per_100_passes",
    "final_third_entries_per_100_passes",
)

CORE_FEATURES = (
    "expected_completion_rate",
    "pressure_pass_rate",
    "progressive_pass_rate",
    "long_pass_rate",
    "positive_forward_distance_per_100_passes",
    "carry_share_of_actions",
)
OPTIONAL_FEATURES = (
    "progressive_carry_rate",
    "progressive_action_rate",
    "pressure_action_rate",
)
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "core_6": CORE_FEATURES,
    "core_6_plus_progressive_carry": (*CORE_FEATURES, "progressive_carry_rate"),
    "core_6_plus_progressive_action": (*CORE_FEATURES, "progressive_action_rate"),
    "core_6_plus_pressure_action": (*CORE_FEATURES, "pressure_action_rate"),
    "core_6_plus_all_optional": (*CORE_FEATURES, *OPTIONAL_FEATURES),
}

PROHIBITED_FEATURES = {
    *PERFORMANCE_FEATURES,
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "position",
    "position_group",
    "archetype_id",
    "archetype_name",
    "raw_cluster_id",
    "completed",
    "goal",
    "goals",
    "pass_outcome",
    "success",
}

FEATURE_LABELS = {
    "expected_completion_rate": "Expected Completion Rate",
    "pressure_pass_rate": "Pressure Pass Rate",
    "progressive_pass_rate": "Progressive Pass Rate",
    "long_pass_rate": "Long Pass Rate",
    "positive_forward_distance_per_100_passes": ("Positive Forward Distance Per 100 Passes"),
    "carry_share_of_actions": "Carry Share Of Actions",
    "progressive_carry_rate": "Progressive Carry Rate",
    "progressive_action_rate": "Progressive Action Rate",
    "pressure_action_rate": "Pressure Action Rate",
    "average_forward_distance": "Average Forward Distance",
    "final_third_entries_per_100_passes": "Final Third Entries Per 100 Passes",
}


def validate_feature_sets(feature_sets: dict[str, Sequence[str]] = FEATURE_SETS) -> None:
    """Reject quality, identity, outcome, and archetype leakage."""
    for name, features in feature_sets.items():
        if not features or len(features) != len(set(features)):
            raise ValueError(f"Feature set {name!r} must contain unique features")
        prohibited = PROHIBITED_FEATURES.intersection(features)
        if prohibited:
            raise ValueError(f"Prohibited features in {name}: {sorted(prohibited)}")
        if not set(features).issubset({*CORE_FEATURES, *OPTIONAL_FEATURES}):
            raise ValueError(f"Unregistered style feature in {name}")


def _action_rates(
    actions: pd.DataFrame,
    player_ids: set[int],
    half_assignments: pd.DataFrame | None,
) -> pd.DataFrame:
    required = {"player_id", "match_id", "action_type", "progressive", "under_pressure"}
    missing = required.difference(actions.columns)
    if missing:
        raise ValueError(f"Actions are missing columns: {sorted(missing)}")
    source = actions.loc[actions["player_id"].isin(player_ids)].copy()
    keys = ["player_id"]
    if half_assignments is not None:
        source = source.merge(
            half_assignments[["player_id", "match_id", "half"]],
            on=["player_id", "match_id"],
            how="left",
            validate="many_to_one",
        )
        if source["half"].isna().any():
            raise ValueError("Every action must map to a player-specific half")
        keys.append("half")
    source["is_carry"] = source["action_type"].eq("Carry")
    source["progressive_action"] = source["progressive"].fillna(False).astype(bool)
    source["pressure_action"] = source["under_pressure"].fillna(False).astype(bool)
    source["progressive_carry"] = source["is_carry"] & source["progressive_action"]
    grouped = (
        source.groupby(keys, sort=True, observed=True)
        .agg(
            derived_actions=("action_type", "size"),
            derived_carries=("is_carry", "sum"),
            progressive_actions=("progressive_action", "sum"),
            pressure_actions=("pressure_action", "sum"),
            progressive_carries=("progressive_carry", "sum"),
        )
        .reset_index()
    )
    grouped["progressive_action_rate"] = grouped["progressive_actions"] / grouped[
        "derived_actions"
    ].replace(0, np.nan)
    grouped["pressure_action_rate"] = grouped["pressure_actions"] / grouped[
        "derived_actions"
    ].replace(0, np.nan)
    grouped["progressive_carry_rate"] = grouped["progressive_carries"] / grouped[
        "derived_carries"
    ].replace(0, np.nan)
    return grouped


def _average_forward_distance(
    passes: pd.DataFrame,
    player_ids: set[int],
    half_assignments: pd.DataFrame | None,
) -> pd.DataFrame:
    source = passes.loc[passes["player_id"].isin(player_ids)].copy()
    keys = ["player_id"]
    if half_assignments is not None:
        source = source.merge(
            half_assignments[["player_id", "match_id", "half"]],
            on=["player_id", "match_id"],
            how="left",
            validate="many_to_one",
        )
        if source["half"].isna().any():
            raise ValueError("Every pass must map to a player-specific half")
        keys.append("half")
    source["forward_distance"] = pd.to_numeric(source["forward_distance"], errors="coerce")
    return (
        source.groupby(keys, sort=True, observed=True)
        .agg(average_forward_distance=("forward_distance", "mean"))
        .reset_index()
    )


def recompute_similarity_features(
    passes: pd.DataFrame,
    actions: pd.DataFrame,
    profiles: pd.DataFrame,
    half_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rebuild all V3.3 candidates independently from event-level rows."""
    validate_feature_sets()
    required_profiles = {
        "player_id",
        "player_name",
        "team_name",
        "position",
        "position_group",
    }
    missing = required_profiles.difference(profiles.columns)
    if missing:
        raise ValueError(f"Profiles are missing columns: {sorted(missing)}")
    player_ids = set(profiles["player_id"].astype(int))
    base = recompute_style_features(passes, actions, profiles, half_assignments)
    keys = ["player_id"] + (["half"] if half_assignments is not None else [])
    rates = _action_rates(actions, player_ids, half_assignments)
    forward = _average_forward_distance(passes, player_ids, half_assignments)
    result = base.merge(rates, on=keys, how="left", validate="one_to_one").merge(
        forward, on=keys, how="left", validate="one_to_one"
    )
    for column in (
        "derived_actions",
        "derived_carries",
        "progressive_actions",
        "pressure_actions",
        "progressive_carries",
    ):
        result[column] = result[column].fillna(0).astype(int)
    if not result["actions"].equals(result["derived_actions"].astype(int)):
        raise AssertionError("Action totals differ while rebuilding similarity features")
    if not result["carries"].equals(result["derived_carries"].astype(int)):
        raise AssertionError("Carry totals differ while rebuilding similarity features")
    metadata = profiles[["player_id", "player_name", "team_name", "position", "position_group"]]
    result = result.drop(columns=["position_group"]).merge(
        metadata, on="player_id", how="left", validate="many_to_one"
    )
    columns = [
        "player_id",
        "player_name",
        "team_name",
        "position",
        "position_group",
        *(["half"] if half_assignments is not None else []),
        "pass_attempts",
        "actions",
        "carries",
        "matches_observed",
        *CORE_FEATURES,
        *OPTIONAL_FEATURES,
        "average_forward_distance",
        "final_third_entries_per_100_passes",
    ]
    result = result[columns].sort_values(
        ["player_id", *(["half"] if half_assignments is not None else [])],
        kind="stable",
    )
    if PROHIBITED_FEATURES.intersection(CORE_FEATURES + OPTIONAL_FEATURES):
        raise AssertionError("Style feature contract contains prohibited columns")
    return result.reset_index(drop=True)


def eligible_cohort(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Apply the fixed V3.2C outfield sample thresholds."""
    required = {"player_id", "position_group", "pass_attempts", "carries", *features}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Similarity cohort is missing columns: {sorted(missing)}")
    cohort = frame.loc[
        frame["position_group"].isin(OUTFIELD_GROUPS)
        & frame["pass_attempts"].ge(PASS_THRESHOLD)
        & frame["carries"].ge(CARRY_THRESHOLD)
        & frame[list(features)].notna().all(axis=1)
    ].copy()
    if cohort["player_id"].duplicated().any():
        raise ValueError("Similarity cohort must contain unique player IDs")
    return cohort.sort_values("player_id", kind="stable").reset_index(drop=True)


def normalize_features(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    by_position: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit population z-scores on exactly the supplied sample."""
    if frame[list(features)].isna().any().any():
        raise ValueError("Normalization cannot silently impute missing style values")
    result = frame[["player_id", "position_group"]].copy()
    statistics: dict[str, Any] = {}
    groups: Iterable[tuple[str, pd.DataFrame]]
    if by_position:
        groups = frame.groupby("position_group", sort=True)
    else:
        groups = (("ALL", frame),)
    for group_name, group in groups:
        statistics[str(group_name)] = {}
        for feature in features:
            values = pd.to_numeric(group[feature], errors="raise").astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if not np.isfinite(std) or std <= 0:
                raise ValueError(f"Cannot normalize constant feature {feature} in {group_name}")
            statistics[str(group_name)][feature] = {"mean": mean, "std": std}
            result.loc[group.index, feature] = (values - mean) / std
    if result[list(features)].isna().any().any():
        raise AssertionError("Normalization produced missing values")
    return result, statistics


def distance_matrix(values: np.ndarray, method: str) -> np.ndarray:
    """Compute one of the three predeclared continuous style distances."""
    if method not in DISTANCE_METHODS:
        raise ValueError(f"Unsupported distance method: {method}")
    differences = values[:, None, :] - values[None, :, :]
    if method == "rms_euclidean":
        matrix = np.sqrt(np.mean(np.square(differences), axis=2))
    elif method == "mean_manhattan":
        matrix = np.mean(np.abs(differences), axis=2)
    else:
        norms = np.linalg.norm(values, axis=1)
        denominator = np.outer(norms, norms)
        similarities = np.divide(
            values @ values.T,
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator > 0,
        )
        similarities = np.clip(similarities, -1.0, 1.0)
        matrix = 1.0 - similarities
    np.fill_diagonal(matrix, 0.0)
    return np.asarray(matrix, dtype=float)


def _candidate_indices(
    frame: pd.DataFrame,
    player_index: int,
    *,
    candidate_mode: str,
) -> list[int]:
    group = str(frame.iloc[player_index]["position_group"])
    same = [
        index
        for index in range(len(frame))
        if index != player_index and frame.iloc[index]["position_group"] == group
    ]
    if candidate_mode == "same_position":
        return same
    if candidate_mode != "v1_fallback":
        raise ValueError(f"Unknown candidate mode: {candidate_mode}")
    if len(same) >= 5 or group == "GK":
        return same
    adjacent = [
        index
        for index in range(len(frame))
        if index != player_index
        and frame.iloc[index]["position_group"] in ADJACENT_OUTFIELD_GROUPS[group]
    ]
    return [*same, *adjacent]


def rank_neighbors(
    normalized: pd.DataFrame,
    features: Sequence[str],
    method: str,
    *,
    top_n: int | None = None,
    candidate_mode: str = "same_position",
    include_explanations: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Rank deterministic neighbors and derive cohort-calibrated scores."""
    if normalized["player_id"].duplicated().any():
        raise ValueError("Normalized profiles must be unique by player")
    values = normalized[list(features)].to_numpy(dtype=float)
    distances = distance_matrix(values, method)
    player_ids = normalized["player_id"].astype(int).to_numpy()
    positions = normalized["position_group"].astype(str).to_numpy()
    indices = np.arange(len(normalized))

    def candidates_for(player_index: int) -> list[int]:
        group = positions[player_index]
        same = indices[(indices != player_index) & (positions == group)].tolist()
        if candidate_mode == "same_position" or len(same) >= 5 or group == "GK":
            return same
        if candidate_mode != "v1_fallback":
            raise ValueError(f"Unknown candidate mode: {candidate_mode}")
        adjacent = indices[
            (indices != player_index) & np.isin(positions, ADJACENT_OUTFIELD_GROUPS[group])
        ].tolist()
        return [*same, *adjacent]

    pair_distances: list[float] = []
    for left in range(len(normalized)):
        for right in candidates_for(left):
            if left < right:
                pair_distances.append(float(distances[left, right]))
    positive = [value for value in pair_distances if value > 0 and np.isfinite(value)]
    if not positive:
        raise ValueError("A positive same-position distance is required for score mapping")
    d50 = float(np.median(positive))
    rows: list[dict[str, Any]] = []
    for player_index in range(len(normalized)):
        candidates = candidates_for(player_index)
        ranked = sorted(
            candidates,
            key=lambda index: (
                float(distances[player_index, index]),
                int(player_ids[index]),
            ),
        )
        if top_n is not None:
            ranked = ranked[:top_n]
        for rank, candidate_index in enumerate(ranked, start=1):
            distance = float(distances[player_index, candidate_index])
            row: dict[str, Any] = {
                "player_id": int(player_ids[player_index]),
                "position_group": str(positions[player_index]),
                "similar_player_id": int(player_ids[candidate_index]),
                "similar_position_group": str(positions[candidate_index]),
                "rank": rank,
                "distance": distance,
                "similarity_score": 100.0 * np.exp(-SCORE_LN2 * distance / d50),
            }
            if include_explanations:
                gaps = np.abs(values[player_index] - values[candidate_index])
                closest = sorted(
                    range(len(features)), key=lambda index: (float(gaps[index]), index)
                )[:3]
                closest_names: list[str | None] = [str(features[index]) for index in closest]
                closest_names.extend([None] * (3 - len(closest_names)))
                if method == "rms_euclidean":
                    raw_contributions = np.square(gaps)
                else:
                    raw_contributions = gaps
                total = float(raw_contributions.sum())
                contributions = (
                    raw_contributions / total if total > 0 else np.zeros_like(raw_contributions)
                )
                row.update(
                    {
                        "closest_feature_1": closest_names[0],
                        "closest_feature_2": closest_names[1],
                        "closest_feature_3": closest_names[2],
                    }
                )
                row["feature_gaps"] = json.dumps(
                    {feature: float(gap) for feature, gap in zip(features, gaps, strict=True)},
                    sort_keys=True,
                )
                row["distance_contributions"] = json.dumps(
                    {
                        feature: float(contribution)
                        for feature, contribution in zip(features, contributions, strict=True)
                    },
                    sort_keys=True,
                )
            rows.append(row)
    result = pd.DataFrame.from_records(rows)
    validate_rankings(result, normalized, candidate_mode)
    return result, d50


def validate_rankings(
    rankings: pd.DataFrame,
    cohort: pd.DataFrame,
    candidate_mode: str = "same_position",
) -> None:
    """Enforce deterministic, identity-free ranking invariants."""
    if rankings.empty:
        raise ValueError("Similarity rankings cannot be empty")
    if rankings["player_id"].eq(rankings["similar_player_id"]).any():
        raise AssertionError("A player cannot be their own neighbor")
    if rankings.duplicated(["player_id", "similar_player_id"]).any():
        raise AssertionError("A candidate may appear only once per player")
    if rankings.duplicated(["player_id", "rank"]).any():
        raise AssertionError("Ranks must be unique per player")
    if not rankings["similarity_score"].between(0.0, 100.0).all():
        raise AssertionError("Similarity indices must remain in [0, 100]")
    if (
        candidate_mode == "same_position"
        and not rankings["position_group"].eq(rankings["similar_position_group"]).all()
    ):
        raise AssertionError("V3 recommendations must remain within broad position")
    for _, group in rankings.groupby("player_id", sort=False):
        if group["rank"].tolist() != list(range(1, len(group) + 1)):
            raise AssertionError("Ranks must be consecutive and deterministic")
    if set(rankings["player_id"]) != set(cohort["player_id"]):
        raise AssertionError("Every eligible player must receive candidates")


def compare_rankings(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    top_ks: Sequence[int] = TOP_K_VALUES,
) -> pd.DataFrame:
    """Calculate player-level neighbor overlap and rank consistency."""
    records: list[dict[str, Any]] = []
    reference_groups = {
        int(player_id): group.sort_values("rank")
        for player_id, group in reference.groupby("player_id", sort=False)
    }
    comparison_groups = {
        int(player_id): group.sort_values("rank")
        for player_id, group in comparison.groupby("player_id", sort=False)
    }
    common_players = sorted(set(reference_groups).intersection(comparison_groups))
    for player_id in common_players:
        left = reference_groups[player_id]
        right = comparison_groups[player_id]
        left_ids = left["similar_player_id"].astype(int).tolist()
        right_ids = right["similar_player_id"].astype(int).tolist()
        candidate_count = min(len(left_ids), len(right_ids))
        record: dict[str, Any] = {
            "player_id": int(player_id),
            "position_group": str(left.iloc[0]["position_group"]),
            "candidate_count": candidate_count,
        }
        for k in top_ks:
            # If k exhausts a candidate pool, overlap is mechanically perfect and
            # is intentionally marked unavailable as a stability diagnostic.
            if candidate_count <= k:
                record[f"top_{k}_overlap"] = np.nan
                record[f"top_{k}_jaccard"] = np.nan
                continue
            left_set = set(left_ids[:k])
            right_set = set(right_ids[:k])
            intersection = len(left_set.intersection(right_set))
            union = len(left_set.union(right_set))
            record[f"top_{k}_overlap"] = intersection / k
            record[f"top_{k}_jaccard"] = intersection / union if union else np.nan
        left_ranks = left.set_index("similar_player_id")["rank"]
        right_ranks = right.set_index("similar_player_id")["rank"]
        common_candidates = left_ranks.index.intersection(right_ranks.index)
        record["rank_spearman"] = (
            float(
                left_ranks.loc[common_candidates].corr(
                    right_ranks.loc[common_candidates], method="spearman"
                )
            )
            if len(common_candidates) >= 3
            else np.nan
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _distribution(values: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p25": None}
    return {
        "count": len(clean),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p10": float(clean.quantile(0.10)),
        "p25": float(clean.quantile(0.25)),
    }


def summarize_stability(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = [
        *(f"top_{k}_{kind}" for k in TOP_K_VALUES for kind in ("overlap", "jaccard")),
        "rank_spearman",
    ]
    return {
        "players": int(frame["player_id"].nunique()),
        "overall": {metric: _distribution(frame[metric]) for metric in metrics},
        "by_position": {
            str(position): {metric: _distribution(group[metric]) for metric in metrics}
            for position, group in frame.groupby("position_group", sort=True)
        },
    }


def pairwise_distance_consistency(
    first: pd.DataFrame, second: pd.DataFrame
) -> dict[str, dict[str, float | int | None]]:
    left = first.loc[first["player_id"].lt(first["similar_player_id"])][
        ["player_id", "similar_player_id", "position_group", "distance"]
    ]
    right = second.loc[second["player_id"].lt(second["similar_player_id"])][
        ["player_id", "similar_player_id", "distance"]
    ]
    merged = left.merge(
        right,
        on=["player_id", "similar_player_id"],
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    output: dict[str, dict[str, float | int | None]] = {}
    for position, group in [("ALL", merged), *list(merged.groupby("position_group"))]:
        correlation = group["distance_first"].corr(group["distance_second"], method="spearman")
        output[str(position)] = {
            "pairs": len(group),
            "spearman": None if pd.isna(correlation) else float(correlation),
        }
    return output


def evaluate_split_half(
    halves: pd.DataFrame,
    full_ids: set[int],
    features: Sequence[str],
    method: str,
    *,
    by_position: bool,
    candidate_mode: str = "same_position",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pivot_counts = halves.groupby("player_id")["half"].nunique()
    ids = full_ids.intersection(pivot_counts.loc[pivot_counts.eq(2)].index.astype(int))
    source = halves.loc[halves["player_id"].isin(ids)].copy()
    complete_ids = set(
        source.groupby("player_id")[list(features)]
        .apply(lambda group: bool(group.notna().all().all()))
        .loc[lambda values: values]
        .index.astype(int)
    )
    source = source.loc[source["player_id"].isin(complete_ids)]
    rankings: dict[str, pd.DataFrame] = {}
    d50: dict[str, float] = {}
    for half in ("first", "second"):
        half_frame = source.loc[source["half"].eq(half)].reset_index(drop=True)
        normalized, _ = normalize_features(half_frame, features, by_position=by_position)
        rankings[half], d50[half] = rank_neighbors(
            normalized,
            features,
            method,
            candidate_mode=candidate_mode,
            include_explanations=False,
        )
    player_metrics = compare_rankings(rankings["first"], rankings["second"])
    report = summarize_stability(player_metrics)
    report["eligible_full_players"] = len(full_ids)
    report["players_with_two_complete_halves"] = len(complete_ids)
    report["d50_by_half"] = d50
    report["pairwise_distance_consistency"] = pairwise_distance_consistency(
        rankings["first"], rankings["second"]
    )
    return player_metrics, report


def evaluate_resampling(
    resampled_feature_frames: Sequence[pd.DataFrame],
    full_cohort: pd.DataFrame,
    full_rankings: pd.DataFrame,
    features: Sequence[str],
    method: str,
    *,
    by_position: bool,
    candidate_mode: str = "same_position",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[pd.DataFrame] = []
    full_ids = set(full_cohort["player_id"].astype(int))
    for resample, feature_frame in enumerate(resampled_feature_frames):
        sampled = feature_frame.copy()
        sampled = sampled.loc[
            sampled["player_id"].isin(full_ids) & sampled[list(features)].notna().all(axis=1)
        ].reset_index(drop=True)
        valid_positions = sampled["position_group"].value_counts().loc[lambda x: x.ge(2)].index
        sampled = sampled.loc[sampled["position_group"].isin(valid_positions)].reset_index(
            drop=True
        )
        if len(sampled) < 2:
            continue
        normalized, _ = normalize_features(sampled, features, by_position=by_position)
        sampled_rankings, _ = rank_neighbors(
            normalized,
            features,
            method,
            # Keep one row beyond top-10 so top-10 is not mistaken for the
            # entire candidate universe. FWD genuinely has only 10 peers.
            top_n=11,
            candidate_mode=candidate_mode,
            include_explanations=False,
        )
        comparison = compare_rankings(
            full_rankings.loc[full_rankings["rank"].le(11)], sampled_rankings
        )
        comparison["resample"] = resample
        comparison["sampled_players"] = len(sampled)
        records.append(comparison)
    if not records:
        raise ValueError("No match-level resample produced comparable players")
    results = pd.concat(records, ignore_index=True)
    player_means = results.groupby(["player_id", "position_group"], as_index=False).agg(
        evaluated_resamples=("resample", "nunique"),
        top_5_overlap=("top_5_overlap", "mean"),
        top_5_jaccard=("top_5_jaccard", "mean"),
        top_6_overlap=("top_6_overlap", "mean"),
        top_6_jaccard=("top_6_jaccard", "mean"),
        top_10_overlap=("top_10_overlap", "mean"),
        top_10_jaccard=("top_10_jaccard", "mean"),
        rank_spearman=("rank_spearman", "mean"),
    )
    report = summarize_stability(player_means)
    report.update(
        {
            "resamples": len(resampled_feature_frames),
            "sample_fraction": RESAMPLE_FRACTION,
            "sample_unit": "whole match_id without replacement",
            "players_evaluated_at_least_once": int(player_means["player_id"].nunique()),
            "mean_player_resample_coverage": float(
                player_means["evaluated_resamples"].mean() / len(resampled_feature_frames)
            ),
            "minimum_player_resample_coverage": float(
                player_means["evaluated_resamples"].min() / len(resampled_feature_frames)
            ),
        }
    )
    return player_means, report


def feature_stability(
    full: pd.DataFrame, halves: pd.DataFrame, eligible_ids: set[int]
) -> dict[str, Any]:
    source = halves.loc[halves["player_id"].isin(eligible_ids)]
    output: dict[str, Any] = {}
    for feature in (*CORE_FEATURES, *OPTIONAL_FEATURES):
        pivot = source.pivot(index="player_id", columns="half", values=feature)
        if not {"first", "second"}.issubset(pivot.columns):
            valid = pivot.iloc[0:0]
        else:
            valid = pivot[["first", "second"]].dropna()
        overall = valid["first"].corr(valid["second"], method="spearman")
        positions: dict[str, Any] = {}
        lookup = full.set_index("player_id")["position_group"]
        for position in OUTFIELD_GROUPS:
            position_ids = valid.index.intersection(lookup.loc[lookup.eq(position)].index)
            correlation = valid.loc[position_ids, "first"].corr(
                valid.loc[position_ids, "second"], method="spearman"
            )
            positions[position] = {
                "players": len(position_ids),
                "spearman": None if pd.isna(correlation) else float(correlation),
            }
        output[feature] = {
            "players": len(valid),
            "spearman": None if pd.isna(overall) else float(overall),
            "median_absolute_difference": (
                float((valid["first"] - valid["second"]).abs().median()) if len(valid) else None
            ),
            "by_position": positions,
        }
    return output


def feature_availability(full: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for feature in (*CORE_FEATURES, *OPTIONAL_FEATURES):
        output[feature] = {
            "available_product_players": int(full[feature].notna().sum()),
            "missing_product_players": int(full[feature].isna().sum()),
            "available_by_position": {
                str(position): int(group[feature].notna().sum())
                for position, group in full.groupby("position_group", sort=True)
            },
        }
    return output


def correlation_audit(cohort: pd.DataFrame) -> list[dict[str, Any]]:
    normalized, _ = normalize_features(
        cohort, (*CORE_FEATURES, *OPTIONAL_FEATURES), by_position=True
    )
    correlations = normalized[[*CORE_FEATURES, *OPTIONAL_FEATURES]].corr()
    rows = []
    features = [*CORE_FEATURES, *OPTIONAL_FEATURES]
    for index, left in enumerate(features):
        for right in features[index + 1 :]:
            value = float(correlations.at[left, right])
            if abs(value) >= 0.80:
                rows.append({"left": left, "right": right, "pearson": value})
    return sorted(rows, key=lambda row: (-abs(row["pearson"]), row["left"], row["right"]))


def build_match_resamples(
    match_ids: Sequence[int],
    *,
    count: int = RESAMPLE_COUNT,
    fraction: float = RESAMPLE_FRACTION,
    seed: int = RANDOM_STATE,
) -> list[list[int]]:
    """Draw deterministic whole-match samples without replacement."""
    unique_ids = np.asarray(sorted({int(match_id) for match_id in match_ids}), dtype=int)
    if len(unique_ids) < 2:
        raise ValueError("Match-aware resampling requires at least two matches")
    if count < 1 or not 0 < fraction < 1:
        raise ValueError("Resample count and fraction must be valid")
    sample_size = int(np.ceil(fraction * len(unique_ids)))
    rng = np.random.default_rng(seed)
    return [
        sorted(rng.choice(unique_ids, size=sample_size, replace=False).astype(int).tolist())
        for _ in range(count)
    ]


def _methodology_key(feature_set: str, method: str, normalization: str) -> str:
    return f"{feature_set}__{method}__{normalization}"


def select_methodology(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select on stability, then parsimony and interpretability within 0.01."""
    eligible = [
        row
        for row in results
        if row["split_top_6_overlap_mean"] is not None
        and row["resample_top_6_overlap_mean"] is not None
    ]
    if not eligible:
        raise ValueError("No similarity methodology has complete validation evidence")
    for row in eligible:
        row["selection_stability_score"] = float(
            0.5 * row["split_top_6_overlap_mean"] + 0.5 * row["resample_top_6_overlap_mean"]
        )
    best_score = max(row["selection_stability_score"] for row in eligible)
    near_best = [row for row in eligible if best_score - row["selection_stability_score"] <= 0.01]
    method_priority = {"rms_euclidean": 0, "mean_manhattan": 1, "cosine": 2}
    selected = min(
        near_best,
        key=lambda row: (
            row["feature_count"],
            method_priority[row["distance_method"]],
            -row["selection_stability_score"],
            row["feature_set"],
        ),
    )
    defensible = bool(
        selected["split_top_6_overlap_mean"] >= 0.35
        and selected["split_top_6_overlap_median"] >= 1 / 3
        and selected["split_rank_spearman_mean"] >= 0.25
        and selected["resample_top_6_overlap_mean"] >= 0.60
    )
    return {
        **selected,
        "best_observed_stability_score": best_score,
        "parsimony_tolerance": 0.01,
        "defensible": defensible,
        "defensibility_floor": {
            "split_top_6_overlap_mean": 0.35,
            "split_top_6_overlap_median": 1 / 3,
            "split_rank_spearman_mean": 0.25,
            "resample_top_6_overlap_mean": 0.60,
            "rationale": (
                "Require at least one-third typical top-six recurrence, a positive "
                "moderate mean rank relationship, and robust match-resample retention."
            ),
        },
    }


def v1_comparison(
    old: pd.DataFrame,
    proposed: pd.DataFrame,
    old_split_stability: dict[str, Any],
    proposed_split_stability: dict[str, Any],
    old_resample_stability: dict[str, Any],
    proposed_resample_stability: dict[str, Any],
) -> dict[str, Any]:
    old_top = old.loc[old["rank"].le(6)]
    proposed_top = proposed.loc[proposed["rank"].le(6)]
    common = sorted(set(old_top["player_id"]).intersection(proposed_top["player_id"]))
    overlaps = []
    top_one = []
    rank_correlations = []
    for player_id in common:
        left = old_top.loc[old_top["player_id"].eq(player_id)].sort_values("rank")
        right = proposed_top.loc[proposed_top["player_id"].eq(player_id)].sort_values("rank")
        left_ids = left["similar_player_id"].astype(int).tolist()
        right_ids = right["similar_player_id"].astype(int).tolist()
        denominator = min(6, len(left_ids), len(right_ids))
        if denominator:
            overlaps.append(len(set(left_ids[:6]).intersection(right_ids[:6])) / denominator)
        if left_ids and right_ids:
            top_one.append(left_ids[0] == right_ids[0])
        left_ranks = left.set_index("similar_player_id")["rank"]
        right_ranks = right.set_index("similar_player_id")["rank"]
        candidates = left_ranks.index.intersection(right_ranks.index)
        if len(candidates) >= 3:
            rank_correlations.append(
                float(
                    left_ranks.loc[candidates].corr(right_ranks.loc[candidates], method="spearman")
                )
            )
    return {
        "v1_recipients": int(old["player_id"].nunique()),
        "v1_rows": len(old),
        "v3_recipients": int(proposed["player_id"].nunique()),
        "v3_research_rows": len(proposed),
        "common_recipients": len(common),
        "top_6_overlap": _distribution(pd.Series(overlaps, dtype=float)),
        "top_1_retention_rate": float(np.mean(top_one)) if top_one else None,
        "top_1_changed_players": int(len(top_one) - sum(top_one)),
        "common_candidate_rank_spearman": _distribution(pd.Series(rank_correlations, dtype=float)),
        "split_half_top_6_overlap_mean": {
            "v1": old_split_stability["overall"]["top_6_overlap"]["mean"],
            "v3": proposed_split_stability["overall"]["top_6_overlap"]["mean"],
        },
        "resample_top_6_overlap_mean": {
            "v1": old_resample_stability["overall"]["top_6_overlap"]["mean"],
            "v3": proposed_resample_stability["overall"]["top_6_overlap"]["mean"],
        },
    }


def ranking_change_examples(
    old: pd.DataFrame,
    proposed_all: pd.DataFrame,
    profiles: pd.DataFrame,
    split_players: pd.DataFrame,
    resample_players: pd.DataFrame,
) -> dict[str, Any]:
    """Expose representative changes and weak cases without subjective tuning."""
    name_lookup = profiles.set_index("player_id")["player_name"].astype(str).to_dict()
    old_groups = {
        int(player_id): group.sort_values("rank")
        for player_id, group in old.groupby("player_id", sort=False)
    }
    new_groups = {
        int(player_id): group.sort_values("rank")
        for player_id, group in proposed_all.groupby("player_id", sort=False)
    }
    changes: list[dict[str, Any]] = []
    for player_id in sorted(set(old_groups).intersection(new_groups)):
        old_top = int(old_groups[player_id].iloc[0]["similar_player_id"])
        new_top = int(new_groups[player_id].iloc[0]["similar_player_id"])
        if old_top == new_top:
            continue
        old_candidate = new_groups[player_id].loc[
            new_groups[player_id]["similar_player_id"].eq(old_top)
        ]
        changes.append(
            {
                "player_id": player_id,
                "player_name": name_lookup.get(player_id),
                "v1_top_neighbor_id": old_top,
                "v1_top_neighbor_name": name_lookup.get(old_top),
                "v3_top_neighbor_id": new_top,
                "v3_top_neighbor_name": name_lookup.get(new_top),
                "v3_top_neighbor_distance": float(new_groups[player_id].iloc[0]["distance"]),
                "v1_top_neighbor_distance_under_v3_method": (
                    float(old_candidate.iloc[0]["distance"]) if not old_candidate.empty else None
                ),
            }
        )
    comparable = [
        row for row in changes if row["v1_top_neighbor_distance_under_v3_method"] is not None
    ]
    comparable.sort(
        key=lambda row: (
            -(row["v1_top_neighbor_distance_under_v3_method"] - row["v3_top_neighbor_distance"]),
            row["player_id"],
        )
    )

    weak = split_players.merge(
        resample_players[["player_id", "top_6_overlap", "evaluated_resamples"]].rename(
            columns={"top_6_overlap": "resample_top_6_overlap"}
        ),
        on="player_id",
        how="left",
        validate="one_to_one",
    ).merge(
        profiles[["player_id", "player_name"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    weak = weak.rename(columns={"top_6_overlap": "split_top_6_overlap"}).sort_values(
        ["split_top_6_overlap", "resample_top_6_overlap", "player_id"],
        kind="stable",
    )
    return {
        "largest_method_relative_top_neighbor_changes": comparable[:5],
        "interpretation": (
            "These are method-relative examples: the V3 neighbor is closer under the "
            "proposed style distance. They are not claims of footballing truth or quality."
        ),
        "weak_stability_examples": weak[
            [
                "player_id",
                "player_name",
                "position_group",
                "split_top_6_overlap",
                "resample_top_6_overlap",
                "evaluated_resamples",
            ]
        ]
        .head(5)
        .to_dict("records"),
    }


def archetype_audit(
    neighbors: pd.DataFrame,
    archetypes: pd.DataFrame,
) -> dict[str, Any]:
    required = {
        "player_id",
        "archetype_id",
        "archetype_name",
        "separation_margin",
    }
    missing = required.difference(archetypes.columns)
    if missing:
        raise ValueError(f"Archetype artifact is missing columns: {sorted(missing)}")
    source = neighbors.merge(
        archetypes[list(required)].rename(
            columns={
                "archetype_id": "player_archetype_id",
                "archetype_name": "player_archetype_name",
                "separation_margin": "player_separation_margin",
            }
        ),
        on="player_id",
        how="left",
        validate="many_to_one",
    ).merge(
        archetypes[list(required)].rename(
            columns={
                "player_id": "similar_player_id",
                "archetype_id": "similar_archetype_id",
                "archetype_name": "similar_archetype_name",
                "separation_margin": "similar_separation_margin",
            }
        ),
        on="similar_player_id",
        how="left",
        validate="many_to_one",
    )
    if source[["player_archetype_id", "similar_archetype_id"]].isna().any().any():
        raise ValueError("Every V3.3 cohort player must have a diagnostic archetype")
    source["same_archetype"] = source["player_archetype_id"].eq(source["similar_archetype_id"])
    top6 = source.loc[source["rank"].le(6)]
    same_far = source.loc[source["same_archetype"] & source["rank"].gt(6)].nlargest(5, "distance")
    cross_close = top6.loc[~top6["same_archetype"]].nsmallest(5, "distance")
    columns = [
        "player_id",
        "similar_player_id",
        "position_group",
        "rank",
        "distance",
        "similarity_score",
        "player_archetype_name",
        "similar_archetype_name",
        "player_separation_margin",
        "similar_separation_margin",
    ]
    return {
        "top_6_pairs": len(top6),
        "top_6_same_archetype_proportion": float(top6["same_archetype"].mean()),
        "same_archetype_not_close_examples": same_far[columns].to_dict("records"),
        "cross_archetype_close_examples": cross_close[columns].to_dict("records"),
        "diagnostic_only": True,
        "archetype_used_as_input_or_filter": False,
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _analysis_records(
    frame: pd.DataFrame,
    analysis_type: str,
    feature_set: str,
    method: str,
    normalization: str,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "analysis_type", analysis_type)
    result.insert(1, "feature_set", feature_set)
    result.insert(2, "distance_method", method)
    result.insert(3, "normalization", normalization)
    return result


def run_analysis(
    passes_path: Path = DEFAULT_PASSES,
    actions_path: Path = DEFAULT_ACTIONS,
    profiles_path: Path = DEFAULT_PROFILES,
    matches_path: Path = DEFAULT_MATCHES,
    archetypes_path: Path = DEFAULT_ARCHETYPES,
    feature_output: Path = DEFAULT_FEATURE_OUTPUT,
    analysis_output: Path = DEFAULT_ANALYSIS_OUTPUT,
    neighbor_output: Path = DEFAULT_NEIGHBOR_OUTPUT,
    metadata_output: Path = DEFAULT_METADATA_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the complete V3.3A audit without touching production artifacts."""
    validate_feature_sets()
    passes = pd.read_parquet(passes_path)
    actions = pd.read_parquet(actions_path)
    profiles = pd.read_parquet(profiles_path)
    matches = build_match_chronology(
        pd.read_parquet(matches_path), set(passes["match_id"].astype(int))
    )
    archetypes = pd.read_parquet(archetypes_path)
    assignments = assign_player_match_halves(
        passes, actions, matches, set(profiles["player_id"].astype(int))
    )
    full = recompute_similarity_features(passes, actions, profiles)
    halves = recompute_similarity_features(passes, actions, profiles, assignments)

    feature_artifact = full.copy()
    cohorts: dict[str, pd.DataFrame] = {}
    for name, features in FEATURE_SETS.items():
        cohort = eligible_cohort(full, features)
        cohorts[name] = cohort
        feature_artifact[f"eligible__{name}"] = feature_artifact["player_id"].isin(
            cohort["player_id"]
        )
    core_normalized, core_statistics = normalize_features(
        cohorts["core_6"], CORE_FEATURES, by_position=True
    )
    normalized_lookup = core_normalized.set_index("player_id")
    for feature in CORE_FEATURES:
        feature_artifact[f"{feature}_position_z"] = feature_artifact["player_id"].map(
            normalized_lookup[feature]
        )

    match_ids = matches["match_id"].astype(int).to_numpy()
    sample_size = int(np.ceil(RESAMPLE_FRACTION * len(match_ids)))
    resamples = build_match_resamples(match_ids)
    resampled_feature_frames = [
        recompute_similarity_features(
            passes.loc[passes["match_id"].isin(match_sample)],
            actions.loc[actions["match_id"].isin(match_sample)],
            profiles,
        )
        for match_sample in resamples
    ]

    methodology_results: list[dict[str, Any]] = []
    full_rankings: dict[str, pd.DataFrame] = {}
    analysis_frames: list[pd.DataFrame] = []
    split_reports: dict[str, Any] = {}
    resample_reports: dict[str, Any] = {}
    split_player_results: dict[str, pd.DataFrame] = {}
    resample_player_results: dict[str, pd.DataFrame] = {}
    d50_values: dict[str, float] = {}
    for feature_set, features in FEATURE_SETS.items():
        cohort = cohorts[feature_set]
        normalized, _ = normalize_features(cohort, features, by_position=True)
        for method in DISTANCE_METHODS:
            key = _methodology_key(feature_set, method, "position")
            rankings, d50 = rank_neighbors(normalized, features, method)
            full_rankings[key] = rankings
            d50_values[key] = d50
            split_players, split = evaluate_split_half(
                halves,
                set(cohort["player_id"].astype(int)),
                features,
                method,
                by_position=True,
            )
            resample_players, resample = evaluate_resampling(
                resampled_feature_frames,
                cohort,
                rankings,
                features,
                method,
                by_position=True,
            )
            split_reports[key] = split
            resample_reports[key] = resample
            split_player_results[key] = split_players
            resample_player_results[key] = resample_players
            analysis_frames.extend(
                [
                    _analysis_records(
                        split_players, "split_half_player", feature_set, method, "position"
                    ),
                    _analysis_records(
                        resample_players,
                        "resample_player_mean",
                        feature_set,
                        method,
                        "position",
                    ),
                ]
            )
            methodology_results.append(
                {
                    "key": key,
                    "feature_set": feature_set,
                    "features": list(features),
                    "feature_count": len(features),
                    "distance_method": method,
                    "normalization": "position",
                    "eligible_players": len(cohort),
                    "eligible_by_position": {
                        str(group): int(count)
                        for group, count in cohort["position_group"].value_counts().items()
                    },
                    "split_top_6_overlap_mean": split["overall"]["top_6_overlap"]["mean"],
                    "split_top_6_overlap_median": split["overall"]["top_6_overlap"]["median"],
                    "split_top_6_overlap_p25": split["overall"]["top_6_overlap"]["p25"],
                    "split_top_6_jaccard_mean": split["overall"]["top_6_jaccard"]["mean"],
                    "split_rank_spearman_mean": split["overall"]["rank_spearman"]["mean"],
                    "resample_top_6_overlap_mean": resample["overall"]["top_6_overlap"]["mean"],
                    "resample_top_6_jaccard_mean": resample["overall"]["top_6_jaccard"]["mean"],
                    "resample_rank_spearman_mean": resample["overall"]["rank_spearman"]["mean"],
                    "d50": d50,
                }
            )

    selected = select_methodology(methodology_results)
    selected_key = str(selected["key"])
    selected_features = tuple(selected["features"])
    selected_neighbors = full_rankings[selected_key].loc[lambda frame: frame["rank"].le(10)]
    names = profiles[["player_id", "player_name", "team_name", "position"]]
    selected_neighbors = selected_neighbors.merge(
        names,
        on="player_id",
        how="left",
        validate="many_to_one",
    ).merge(
        names.rename(
            columns={
                "player_id": "similar_player_id",
                "player_name": "similar_player_name",
                "team_name": "similar_team_name",
                "position": "similar_position",
            }
        ),
        on="similar_player_id",
        how="left",
        validate="many_to_one",
    )
    selected_neighbors.insert(0, "methodology_version", VERSION)
    selected_neighbors.insert(1, "feature_set", selected["feature_set"])
    selected_neighbors.insert(2, "distance_method", selected["distance_method"])
    selected_neighbors.insert(3, "normalization", "position-relative population z-score")

    # Sensitivity: hold the selected features/method fixed and change only
    # position-relative versus all-outfield normalization.
    selected_cohort = cohorts[str(selected["feature_set"])]
    global_normalized, _ = normalize_features(selected_cohort, selected_features, by_position=False)
    global_rankings, global_d50 = rank_neighbors(
        global_normalized, selected_features, str(selected["distance_method"])
    )
    normalization_sensitivity = summarize_stability(
        compare_rankings(full_rankings[selected_key], global_rankings)
    )
    distance_sensitivity = {
        method: summarize_stability(
            compare_rankings(
                full_rankings[selected_key],
                full_rankings[_methodology_key(str(selected["feature_set"]), method, "position")],
            )
        )
        for method in DISTANCE_METHODS
        if method != selected["distance_method"]
    }
    threshold_sensitivity: dict[str, Any] = {}
    for pass_threshold in (50, 75, 100, 150):
        threshold_cohort = full.loc[
            full["position_group"].isin(OUTFIELD_GROUPS)
            & full["pass_attempts"].ge(pass_threshold)
            & full["carries"].ge(CARRY_THRESHOLD)
            & full[list(selected_features)].notna().all(axis=1)
        ].copy()
        _, threshold_report = evaluate_split_half(
            halves,
            set(threshold_cohort["player_id"].astype(int)),
            selected_features,
            str(selected["distance_method"]),
            by_position=True,
        )
        threshold_sensitivity[str(pass_threshold)] = {
            "players": len(threshold_cohort),
            "by_position": {
                str(group): int(count)
                for group, count in threshold_cohort["position_group"].value_counts().items()
            },
            "split_half": threshold_report,
        }

    # Reconstruct the historical V1 baseline under its actual global scaling, feature set,
    # 100-pass eligibility, cosine distance, and adjacent-position fallback.
    v1_full = full.loc[full["pass_attempts"].ge(100) & full[V1_FEATURES].notna().all(axis=1)].copy()
    v1_ids = set(v1_full["player_id"].astype(int))
    v1_normalized, _ = normalize_features(v1_full, V1_FEATURES, by_position=False)
    v1_full_rankings, v1_d50 = rank_neighbors(
        v1_normalized, V1_FEATURES, "cosine", candidate_mode="v1_fallback"
    )
    v1_split_players, v1_split_report = evaluate_split_half(
        halves,
        v1_ids,
        V1_FEATURES,
        "cosine",
        by_position=False,
        candidate_mode="v1_fallback",
    )
    analysis_frames.append(
        _analysis_records(v1_split_players, "split_half_player", "v1_7", "cosine", "global")
    )
    v1_resample_players, v1_resample_report = evaluate_resampling(
        resampled_feature_frames,
        v1_full,
        v1_full_rankings,
        V1_FEATURES,
        "cosine",
        by_position=False,
        candidate_mode="v1_fallback",
    )
    analysis_frames.append(
        _analysis_records(
            v1_resample_players,
            "resample_player_mean",
            "v1_7",
            "cosine",
            "global",
        )
    )

    archetype_report = archetype_audit(full_rankings[selected_key], archetypes)
    old_vs_new = v1_comparison(
        v1_full_rankings,
        selected_neighbors,
        v1_split_report,
        split_reports[selected_key],
        v1_resample_report,
        resample_reports[selected_key],
    )
    old_vs_new["examples"] = ranking_change_examples(
        v1_full_rankings,
        full_rankings[selected_key],
        profiles,
        split_player_results[selected_key],
        resample_player_results[selected_key],
    )

    analysis = pd.concat(analysis_frames, ignore_index=True, sort=False)
    if selected["defensible"] is not True:
        recommendation = "No production methodology is recommended from current evidence."
    else:
        recommendation = (
            f"{selected['distance_method']} on {selected['feature_set']} after "
            "within-position population z-scoring, with same-position candidates only."
        )
    metadata: dict[str, Any] = {
        "methodology_version": VERSION,
        "scope": "Historical V3.3A research only; production similarity artifacts are unchanged.",
        "product_cohort": {
            "players": len(full),
            "by_position": {
                str(group): int(count)
                for group, count in full["position_group"].value_counts().items()
            },
        },
        "eligibility": {
            "minimum_pass_attempts": PASS_THRESHOLD,
            "minimum_carries": CARRY_THRESHOLD,
            "positions": list(OUTFIELD_GROUPS),
            "goalkeeper_policy": (
                "Unavailable: current product data do not support a separately validated "
                "goalkeeper style-similarity model."
            ),
            "cohorts": {
                name: {
                    "players": len(cohort),
                    "by_position": {
                        str(group): int(count)
                        for group, count in cohort["position_group"].value_counts().items()
                    },
                }
                for name, cohort in cohorts.items()
            },
        },
        "feature_sets": {name: list(features) for name, features in FEATURE_SETS.items()},
        "prohibited_features": sorted(PROHIBITED_FEATURES),
        "feature_availability": feature_availability(full),
        "split_half_feature_stability": feature_stability(
            full, halves, set(cohorts["core_6_plus_all_optional"]["player_id"].astype(int))
        ),
        "position_relative_redundancy_pairs_abs_ge_0_80": correlation_audit(
            cohorts["core_6_plus_all_optional"]
        ),
        "methodology_results": methodology_results,
        "split_half_reports": split_reports,
        "resampling_reports": resample_reports,
        "selection": selected,
        "recommendation": recommendation,
        "score_mapping": {
            "formula": "100 * exp(-ln(2) * distance / D50)",
            "interpretation": (
                "A bounded cohort-calibrated similarity index, not a probability, quality "
                "score, or success forecast. Identical profiles map to 100 and the median "
                "eligible same-position pair distance maps to 50."
            ),
            "selected_d50": d50_values[selected_key],
            "all_candidate_d50": d50_values,
            "arbitrary_clipping": False,
        },
        "feature_explanations": {
            "closest_dimensions": (
                "The three smallest absolute position-z gaps, ordered deterministically "
                "by gap then declared feature order. A feature is not selected merely "
                "because both raw values are high."
            ),
            "distance_contributions": (
                "For selected RMS Euclidean, squared per-feature z gaps divided by their "
                "sum. These explain dissimilarity; closest dimensions explain similarity."
            ),
        },
        "normalization": {
            "production_candidate": "Population z-score within eligible broad position.",
            "validation": (
                "First half, second half, and every match resample fit their own position "
                "normalization statistics; full-data statistics are never reused."
            ),
            "selected_full_cohort_statistics": core_statistics
            if selected["feature_set"] == "core_6"
            else normalize_features(selected_cohort, selected_features, by_position=True)[1],
            "global_vs_position_neighbor_stability": normalization_sensitivity,
            "global_sensitivity_d50": global_d50,
        },
        "distance_method_neighbor_sensitivity": distance_sensitivity,
        "pass_threshold_sensitivity": threshold_sensitivity,
        "v1": {
            "features": list(V1_FEATURES),
            "minimum_pass_attempts": 100,
            "normalization": "Global StandardScaler-equivalent population z-score",
            "method": "cosine similarity",
            "score_mapping": "100 * (cosine + 1) / 2",
            "candidate_policy": (
                "Same broad position first; if fewer than five outfield peers, append "
                "adjacent outfield groups; never mix GK and outfield."
            ),
            "eligible_players": len(v1_full),
            "eligible_by_position": {
                str(group): int(count)
                for group, count in v1_full["position_group"].value_counts().items()
            },
            "validation_d50_for_distance_diagnostics": v1_d50,
            "split_half_report": v1_split_report,
            "resampling_report": v1_resample_report,
        },
        "v1_vs_v3": old_vs_new,
        "archetype_relationship_audit": archetype_report,
        "resampling": {
            "seed": RANDOM_STATE,
            "resamples": RESAMPLE_COUNT,
            "fraction": RESAMPLE_FRACTION,
            "sample_size_matches": sample_size,
            "total_matches": len(match_ids),
            "unit": "match_id",
        },
        "limitations": [
            (
                "Only one team's 34-match Bundesliga schedule is fully observed; many "
                "opponents have two observed matches, so lower-tail stability remains limited."
            ),
            (
                "FWD has only 11 eligible players; top-10 overlap is mechanically exhaustive "
                "and is intentionally omitted for that position."
            ),
            "No validated goalkeeper model is available from the current cohort.",
            (
                "Similarity describes event-derived style in this sample, not quality, "
                "causal impact, transfer success, or tactical fit."
            ),
        ],
        "artifacts": {
            "features": repository_relative_path(feature_output),
            "analysis": repository_relative_path(analysis_output),
            "proposed_neighbors_research_only": repository_relative_path(neighbor_output),
            "metadata": repository_relative_path(metadata_output),
            "v1_artifacts_overwritten": False,
        },
    }

    _atomic_parquet(feature_artifact, feature_output)
    _atomic_parquet(analysis, analysis_output)
    _atomic_parquet(selected_neighbors, neighbor_output)
    _atomic_json(metadata, metadata_output)
    return feature_artifact, selected_neighbors, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=Path, default=DEFAULT_PASSES)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--archetypes", type=Path, default=DEFAULT_ARCHETYPES)
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_FEATURE_OUTPUT)
    parser.add_argument("--analysis-output", type=Path, default=DEFAULT_ANALYSIS_OUTPUT)
    parser.add_argument("--neighbor-output", type=Path, default=DEFAULT_NEIGHBOR_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, neighbors, metadata = run_analysis(
        args.passes,
        args.actions,
        args.profiles,
        args.matches,
        args.archetypes,
        args.feature_output,
        args.analysis_output,
        args.neighbor_output,
        args.metadata_output,
    )
    selection = metadata["selection"]
    print("FootyScout V3.3A similarity research complete")
    print(f"  Product players: {len(features):,}")
    print(f"  Proposed eligible players: {neighbors['player_id'].nunique():,}")
    print(f"  Proposed method: {metadata['recommendation']}")
    print(f"  Defensible: {selection['defensible']}")
    print(f"  Features: {args.feature_output}")
    print(f"  Analysis: {args.analysis_output}")
    print(f"  Research neighbors: {args.neighbor_output}")
    print(f"  Metadata: {args.metadata_output}")


if __name__ == "__main__":
    main()
