"""Fit the frozen V3.2C production player-style archetype model.

Archetypes describe position-relative playing tendencies, not quality, ability,
or rank. The production fit is deliberately fixed at combined-outfield K-Means
with k=2; this module does not perform model or cluster-count selection.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_SOURCE = PROCESSED / "player_style_features_expanded.parquet"
DEFAULT_PROFILES = PROCESSED / "player_intelligence_profiles.parquet"
DEFAULT_ASSIGNMENTS = PROCESSED / "player_archetypes.parquet"
DEFAULT_MODEL = ROOT / "models" / "player_archetype_model.joblib"
DEFAULT_METADATA = ROOT / "models" / "player_archetype_metadata.json"

MODEL_VERSION = "V3.2C"
PASS_THRESHOLD = 50
CARRY_THRESHOLD = 29
RANDOM_STATE = 42
N_INIT = 100
EPSILON = 1e-12
OUTFIELD_POSITIONS = ("DEF", "MID", "FWD")

PRODUCTION_FEATURES = [
    "expected_completion_rate",
    "pressure_pass_rate",
    "progressive_pass_rate",
    "long_pass_rate",
    "positive_forward_distance_per_100_passes",
    "carry_share_of_actions",
]
NORMALIZED_FEATURES = [f"{feature}_position_z" for feature in PRODUCTION_FEATURES]

FEATURE_LABELS = {
    "expected_completion_rate": "Expected completion",
    "pressure_pass_rate": "Under-pressure pass rate",
    "progressive_pass_rate": "Progressive-pass rate",
    "long_pass_rate": "Long-pass rate",
    "positive_forward_distance_per_100_passes": "Positive forward distance / 100 passes",
    "carry_share_of_actions": "Carry share of actions",
}

ARCHETYPE_NAMES = {
    "direct_progressor": "Direct Progressor",
    "safe_circulator": "Safe Circulator",
}
ARCHETYPE_DESCRIPTIONS = {
    "direct_progressor": (
        "A position-relative passing style oriented toward progressive, longer, and "
        "direct forward distribution, generally through more difficult attempted passes."
    ),
    "safe_circulator": (
        "A position-relative circulation style built around safer attempted passes, less "
        "direct progression, and relatively greater carry involvement in this cohort."
    ),
}


def validate_source(frame: pd.DataFrame) -> None:
    """Validate raw inputs and the frozen six-feature production contract."""
    required = {
        "player_id",
        "position_group",
        "pass_attempts",
        "carries",
        *PRODUCTION_FEATURES,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Archetype source is missing columns: {sorted(missing)}")
    if frame["player_id"].isna().any() or frame["player_id"].duplicated().any():
        raise ValueError("Archetype source requires unique, non-null player_id")
    if "final_third_entries_per_100_passes" in PRODUCTION_FEATURES:
        raise AssertionError("Unstable final-third entries entered the production model")
    if len(PRODUCTION_FEATURES) != 6 or len(set(PRODUCTION_FEATURES)) != 6:
        raise AssertionError("The production archetype model requires exactly six features")


def eligibility_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the frozen clustering-only eligibility mask."""
    return (
        frame["position_group"].isin(OUTFIELD_POSITIONS)
        & frame["pass_attempts"].ge(PASS_THRESHOLD)
        & frame["carries"].ge(CARRY_THRESHOLD)
        & frame[PRODUCTION_FEATURES].notna().all(axis=1)
    )


def normalize_by_position(
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, float]]]]:
    """Apply V3.2A population z-scores within each eligible position group."""
    if not cohort["position_group"].isin(OUTFIELD_POSITIONS).all():
        raise ValueError("Production normalization is restricted to outfield positions")
    result = cohort[["player_id", "position_group"]].copy()
    statistics: dict[str, dict[str, dict[str, float]]] = {}
    for group in OUTFIELD_POSITIONS:
        rows = cohort.loc[cohort["position_group"].eq(group)]
        if rows.empty:
            raise ValueError(f"No eligible {group} players are available")
        statistics[group] = {}
        for feature in PRODUCTION_FEATURES:
            values = pd.to_numeric(rows[feature], errors="coerce")
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if not np.isfinite(std) or std <= 0:
                raise ValueError(f"Cannot normalize constant feature {feature} for {group}")
            statistics[group][feature] = {"mean": mean, "std": std}
            result.loc[rows.index, f"{feature}_position_z"] = (values - mean) / std
    if result[NORMALIZED_FEATURES].isna().any().any():
        raise ValueError("Position normalization produced missing values")
    for group, rows in result.groupby("position_group", sort=True):
        # The frozen expected-completion aggregate is float32, so its centered
        # mean can retain roughly 1e-6 of accumulation error.
        if not np.allclose(rows[NORMALIZED_FEATURES].mean(), 0.0, atol=2e-6):
            raise AssertionError(f"Position-normalized means are not zero for {group}")
        if not np.allclose(rows[NORMALIZED_FEATURES].std(ddof=0), 1.0, atol=1e-12):
            raise AssertionError(f"Position-normalized standard deviations are not one for {group}")
    return result, statistics


def fit_production_model(values: np.ndarray) -> KMeans:
    """Fit the fixed deterministic production K-Means model."""
    model = KMeans(
        n_clusters=2,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        algorithm="lloyd",
    )
    return model.fit(values)


def semantic_cluster_mapping(
    centroids: np.ndarray,
    feature_names: Sequence[str] = PRODUCTION_FEATURES,
) -> dict[int, str]:
    """Verify the fitted centroids before assigning stable product semantics."""
    if centroids.shape != (2, len(feature_names)):
        raise ValueError("Semantic mapping requires two complete six-feature centroids")
    indices = {name: feature_names.index(name) for name in feature_names}
    direct_score = (
        centroids[:, indices["progressive_pass_rate"]]
        + centroids[:, indices["long_pass_rate"]]
        + centroids[:, indices["positive_forward_distance_per_100_passes"]]
        - centroids[:, indices["expected_completion_rate"]]
    )
    direct = int(np.argmax(direct_score))
    safe = 1 - direct
    required_directions = (
        centroids[direct, indices["progressive_pass_rate"]]
        > centroids[safe, indices["progressive_pass_rate"]],
        centroids[direct, indices["long_pass_rate"]]
        > centroids[safe, indices["long_pass_rate"]],
        centroids[direct, indices["positive_forward_distance_per_100_passes"]]
        > centroids[safe, indices["positive_forward_distance_per_100_passes"]],
        centroids[direct, indices["expected_completion_rate"]]
        < centroids[safe, indices["expected_completion_rate"]],
        centroids[direct, indices["carry_share_of_actions"]]
        < centroids[safe, indices["carry_share_of_actions"]],
    )
    if not all(required_directions):
        raise ValueError(
            "Final centroids do not support Direct Progressor / Safe Circulator semantics"
        )
    return {direct: "direct_progressor", safe: "safe_circulator"}


def distance_diagnostics(
    values: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> pd.DataFrame:
    """Calculate assigned/second distances and non-probabilistic separation."""
    distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
    assigned = distances[np.arange(len(values)), labels]
    second = distances[np.arange(len(values)), 1 - labels]
    margin = (second - assigned) / np.maximum(second, EPSILON)
    return pd.DataFrame(
        {
            "centroid_distance": assigned,
            "second_centroid_distance": second,
            "separation_margin": margin,
        }
    )


def _position_composition(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    counts = frame["position_group"].value_counts()
    return {
        group: {
            "count": int(counts.get(group, 0)),
            "percentage": float(100 * counts.get(group, 0) / len(frame)),
        }
        for group in OUTFIELD_POSITIONS
    }


def _margin_summary(values: pd.Series) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "maximum": float(values.max()),
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


def _atomic_joblib(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    temporary.replace(path)


def run_pipeline(
    source_path: Path = DEFAULT_SOURCE,
    profiles_path: Path = DEFAULT_PROFILES,
    assignments_path: Path = DEFAULT_ASSIGNMENTS,
    model_path: Path = DEFAULT_MODEL,
    metadata_path: Path = DEFAULT_METADATA,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit and persist the fixed V3.2C model and eligible assignments."""
    source = pd.read_parquet(source_path)
    validate_source(source)
    eligible = source.loc[eligibility_mask(source)].copy()
    if len(eligible) != 133:
        raise ValueError(f"Expected the audited 133-player cohort, found {len(eligible)}")
    expected_positions = {"DEF": 75, "MID": 47, "FWD": 11}
    actual_positions = eligible["position_group"].value_counts().to_dict()
    if actual_positions != expected_positions:
        raise ValueError(f"Unexpected eligible position composition: {actual_positions}")

    normalized, normalization = normalize_by_position(eligible)
    values = normalized[NORMALIZED_FEATURES].to_numpy(dtype=float)
    model = fit_production_model(values)
    labels = model.labels_.astype(int)
    mapping = semantic_cluster_mapping(model.cluster_centers_)
    distances = distance_diagnostics(values, labels, model.cluster_centers_)

    profiles = pd.read_parquet(profiles_path)[["player_id", "player_name", "team_name"]]
    frame = eligible[["player_id", "position_group"]].merge(
        profiles, on="player_id", how="left", validate="one_to_one"
    )
    if frame[["player_name", "team_name"]].isna().any().any():
        raise ValueError("Every archetype player requires display metadata")
    frame = pd.concat(
        [frame.reset_index(drop=True), normalized[NORMALIZED_FEATURES].reset_index(drop=True), distances],
        axis=1,
    )
    frame["raw_cluster_id"] = labels
    frame["archetype_id"] = [mapping[int(label)] for label in labels]
    frame["archetype_name"] = frame["archetype_id"].map(ARCHETYPE_NAMES)
    frame["eligible"] = True
    frame["model_version"] = MODEL_VERSION
    assignment_columns = [
        "player_id",
        "archetype_id",
        "archetype_name",
        "raw_cluster_id",
        "position_group",
        "centroid_distance",
        "second_centroid_distance",
        "separation_margin",
        "eligible",
        "model_version",
        *NORMALIZED_FEATURES,
    ]
    assignments = frame[assignment_columns].sort_values("player_id").reset_index(drop=True)
    if assignments["player_id"].duplicated().any():
        raise AssertionError("Production assignments must be unique per player")
    if not assignments["separation_margin"].between(0, 1).all():
        raise AssertionError("Archetype separation must remain within [0, 1]")

    centroid_payload: dict[str, dict[str, float]] = {}
    definitions: list[dict[str, Any]] = []
    for raw_cluster_id, archetype_id in sorted(mapping.items(), key=lambda item: item[1]):
        centroid = {
            feature: float(value)
            for feature, value in zip(
                PRODUCTION_FEATURES,
                model.cluster_centers_[raw_cluster_id],
                strict=True,
            )
        }
        centroid_payload[archetype_id] = centroid
        members = frame.loc[frame["archetype_id"].eq(archetype_id)].copy()
        representatives = members.nsmallest(5, "centroid_distance")
        ranked_features = sorted(
            centroid.items(), key=lambda item: (-abs(item[1]), item[0])
        )
        definitions.append(
            {
                "id": archetype_id,
                "name": ARCHETYPE_NAMES[archetype_id],
                "description": ARCHETYPE_DESCRIPTIONS[archetype_id],
                "raw_cluster_id": raw_cluster_id,
                "centroid": centroid,
                "distinguishing_features": [
                    {
                        "feature_name": feature,
                        "label": FEATURE_LABELS[feature],
                        "position_z": value,
                        "direction": "higher" if value > 0 else "lower",
                    }
                    for feature, value in ranked_features[:4]
                ],
                "player_count": len(members),
                "position_composition": _position_composition(members),
                "representative_players": [
                    {
                        "player_id": int(row.player_id),
                        "player_name": str(row.player_name),
                        "team_name": str(row.team_name),
                        "position_group": str(row.position_group),
                        "centroid_distance": float(row.centroid_distance),
                    }
                    for row in representatives.itertuples(index=False)
                ],
                "separation_distribution": _margin_summary(members["separation_margin"]),
            }
        )

    historical_analysis = ROOT / "models" / "player_archetype_analysis.json"
    history = json.loads(historical_analysis.read_text(encoding="utf-8"))
    six_feature_sensitivity = history.get("seven_vs_six_feature_sensitivity", {})
    metadata: dict[str, Any] = {
        "methodology_version": MODEL_VERSION,
        "purpose": "Position-relative playing-style archetypes; not quality, ability, or rank.",
        "method": "K-Means",
        "k": 2,
        "cohort": "combined outfield",
        "feature_order": PRODUCTION_FEATURES,
        "normalized_feature_order": NORMALIZED_FEATURES,
        "thresholds": {
            "minimum_passes": PASS_THRESHOLD,
            "minimum_carries": CARRY_THRESHOLD,
            "positions": list(OUTFIELD_POSITIONS),
            "complete_case_required": True,
        },
        "normalization": {
            "method": "Population z-score within eligible broad position group",
            "ddof": 0,
            "by_position": normalization,
        },
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "algorithm": "lloyd",
        "training_cohort": {
            "players": len(assignments),
            "product_players": len(source),
            "eligible_players": len(assignments),
            "ineligible_players": len(source) - len(assignments),
            "position_composition": actual_positions,
        },
        "raw_cluster_to_archetype": {str(key): value for key, value in mapping.items()},
        "centroids": centroid_payload,
        "definitions": definitions,
        "separation_formula": (
            "(second_centroid_distance - centroid_distance) / "
            "max(second_centroid_distance, 1e-12)"
        ),
        "separation_interpretation": (
            "Distance-based style separation, not a probability. Smaller values indicate "
            "a player lies nearer the boundary between the two archetypes."
        ),
        "overall_separation_distribution": _margin_summary(
            assignments["separation_margin"]
        ),
        "omitted_feature": {
            "feature_name": "final_third_entries_per_100_passes",
            "reason": (
                "V3.2A found weak split-half stability and V3.2B found improved internal "
                "separation after removal while resample/player stability remained similar."
            ),
        },
        "selection_history": {
            "methods_evaluated": ["K-Means", "Gaussian mixture model", "Ward"],
            "k_values_evaluated": [2, 3, 4, 5, 6],
            "selected_before_production_refinement": "K-Means k=2",
            "six_feature_sensitivity": six_feature_sensitivity,
            "interpretation_caution": (
                "Internal clustering diagnostics support a useful descriptive partition; "
                "they do not prove objectively true football roles."
            ),
        },
        "eligibility_reason_for_unassigned": "Limited sample for archetype analysis",
    }
    model_artifact = {
        "model": model,
        "methodology_version": MODEL_VERSION,
        "feature_order": PRODUCTION_FEATURES,
        "normalized_feature_order": NORMALIZED_FEATURES,
        "thresholds": metadata["thresholds"],
        "normalization": normalization,
        "raw_cluster_to_archetype": mapping,
    }
    _atomic_parquet(assignments, assignments_path)
    _atomic_joblib(model_artifact, model_path)
    _atomic_json(metadata, metadata_path)
    return assignments, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignments, metadata = run_pipeline(
        args.source, args.profiles, args.assignments, args.model, args.metadata
    )
    print("FootyScout V3.2C player archetypes complete")
    print(f"  Eligible players: {len(assignments):,}")
    print(f"  Ineligible players: {metadata['training_cohort']['ineligible_players']:,}")
    for definition in metadata["definitions"]:
        print(f"  {definition['name']}: {definition['player_count']:,}")
    print(f"  Assignments: {args.assignments}")
    print(f"  Model: {args.model}")
    print(f"  Metadata: {args.metadata}")


if __name__ == "__main__":
    main()
