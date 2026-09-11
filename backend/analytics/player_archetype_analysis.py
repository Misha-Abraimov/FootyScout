"""Analyze position-relative player style archetypes without production persistence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from analytics.artifact_paths import repository_relative_path
from analytics.player_feature_registry import PERFORMANCE_FEATURES, STYLE_CLUSTERING_FEATURES

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_MATRIX = PROCESSED / "player_style_features_position_normalized.parquet"
DEFAULT_PROFILES = PROCESSED / "player_intelligence_profiles.parquet"
DEFAULT_OUTPUT = PROCESSED / "player_archetype_analysis.parquet"
DEFAULT_METADATA = ROOT / "models" / "player_archetype_analysis.json"

POSITION_GROUPS = ("DEF", "MID", "FWD")
METHODS = ("kmeans", "gmm", "ward")
K_VALUES = (2, 3, 4, 5, 6)
SEED_COUNT = 25
RESAMPLE_COUNT = 100
RESAMPLE_FRACTION = 0.8
MIN_CLUSTER_SIZE = 10
EPSILON = 1e-12
SERIOUS_CANDIDATES = frozenset({("kmeans", 2), ("kmeans", 3), ("gmm", 2), ("ward", 2)})

DEFAULT_PROPOSED_NAMES = {
    "1": ["Direct Progressor", "Vertical Distributor", "Risk-Taking Progressor"],
    "2": ["Safe Circulator", "Retention-Oriented Connector", "Carry-Supporting Link"],
}

NORMALIZED_FEATURES = [f"{feature}_position_z" for feature in STYLE_CLUSTERING_FEATURES]
FINAL_THIRD_FEATURE = "final_third_entries_per_100_passes_position_z"
CORRELATION_SENSITIVITY_EXCLUDED = "expected_completion_rate_position_z"


def validate_clustering_matrix(frame: pd.DataFrame) -> None:
    """Validate the frozen V3.2A matrix and keep identity columns out of features."""
    required = {"player_id", "position_group", *NORMALIZED_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Clustering matrix is missing columns: {sorted(missing)}")
    if len(frame) != 133:
        raise ValueError(f"Expected 133 clustering players, found {len(frame)}")
    if frame["player_id"].isna().any() or frame["player_id"].duplicated().any():
        raise ValueError("Clustering matrix requires unique, non-null player_id")
    counts = frame["position_group"].value_counts().to_dict()
    expected_counts = {"DEF": 75, "MID": 47, "FWD": 11}
    if counts != expected_counts:
        raise ValueError(f"Unexpected position cohort: {counts}")
    if frame[NORMALIZED_FEATURES].isna().any().any():
        raise ValueError("Clustering features cannot contain missing values")
    if any(not column.endswith("_position_z") for column in NORMALIZED_FEATURES):
        raise AssertionError("Every clustering input must be a position-normalized z-score")
    forbidden = set(PERFORMANCE_FEATURES).intersection(frame.columns)
    if forbidden:
        raise ValueError(f"Performance features entered clustering matrix: {sorted(forbidden)}")
    if "player_id" in NORMALIZED_FEATURES or "position_group" in NORMALIZED_FEATURES:
        raise AssertionError("Metadata entered the clustering feature list")
    for group, rows in frame.groupby("position_group"):
        means = rows[NORMALIZED_FEATURES].mean().abs()
        standard_deviations = rows[NORMALIZED_FEATURES].std(ddof=0)
        if (means > 2e-6).any() or not np.allclose(standard_deviations, 1.0, atol=1e-9):
            raise ValueError(f"Position normalization check failed for {group}")


def feature_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {"overall": {}, "by_position": {}}
    for feature in NORMALIZED_FEATURES:
        output["overall"][feature] = {
            "mean": float(frame[feature].mean()),
            "std": float(frame[feature].std(ddof=0)),
        }
    for group in POSITION_GROUPS:
        rows = frame.loc[frame["position_group"].eq(group)]
        output["by_position"][group] = {
            feature: {
                "mean": float(rows[feature].mean()),
                "std": float(rows[feature].std(ddof=0)),
            }
            for feature in NORMALIZED_FEATURES
        }
    return output


def fit_pca(values: np.ndarray, feature_names: Sequence[str]) -> tuple[PCA, np.ndarray]:
    """Fit deterministic full PCA and canonicalize otherwise arbitrary signs."""
    pca = PCA(n_components=len(feature_names), svd_solver="full")
    coordinates = pca.fit_transform(values)
    for index, component in enumerate(pca.components_):
        anchor = int(np.argmax(np.abs(component)))
        if component[anchor] < 0:
            pca.components_[index] *= -1
            coordinates[:, index] *= -1
    return pca, coordinates


def pca_report(pca: PCA, feature_names: Sequence[str]) -> dict[str, Any]:
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    return {
        "explained_variance_ratio": {
            f"PC{index + 1}": float(value) for index, value in enumerate(ratios)
        },
        "cumulative_explained_variance": {
            f"PC{index + 1}": float(value) for index, value in enumerate(cumulative)
        },
        "components_required": {
            str(threshold): int(np.searchsorted(cumulative, threshold) + 1)
            for threshold in (0.70, 0.80, 0.90)
        },
        "loadings": {
            f"PC{index + 1}": {
                feature: float(loading)
                for feature, loading in zip(feature_names, component, strict=True)
            }
            for index, component in enumerate(pca.components_)
        },
    }


def euclidean_distance_contribution(
    values: np.ndarray, feature_names: Sequence[str]
) -> dict[str, Any]:
    """Report each dimension's share of average pairwise squared distance."""
    if len(values) < 2:
        raise ValueError("Distance contribution requires at least two rows")
    differences = values[:, None, :] - values[None, :, :]
    upper = np.triu_indices(len(values), k=1)
    mean_squared = np.square(differences[upper]).mean(axis=0)
    total = float(mean_squared.sum())
    shares = mean_squared / total
    by_feature = {
        feature: {
            "mean_pairwise_squared_difference": float(squared),
            "proportion_of_total_squared_distance": float(share),
        }
        for feature, squared, share in zip(
            feature_names, mean_squared, shares, strict=True
        )
    }
    correlated_pair = [
        "expected_completion_rate_position_z",
        "progressive_pass_rate_position_z",
    ]
    return {
        "by_feature": by_feature,
        "correlated_pair": correlated_pair,
        "correlated_pair_combined_proportion": float(
            sum(by_feature[feature]["proportion_of_total_squared_distance"]
                for feature in correlated_pair)
        ),
    }


def fit_labels(
    values: np.ndarray,
    method: str,
    clusters: int,
    seed: int = 42,
    *,
    intensive: bool = False,
) -> tuple[np.ndarray, object]:
    """Fit one controlled clustering candidate on an already-normalized matrix."""
    if method == "kmeans":
        model = KMeans(
            n_clusters=clusters,
            n_init=50 if intensive else 20,
            random_state=seed,
        )
    elif method == "gmm":
        model = GaussianMixture(
            n_components=clusters,
            covariance_type="full",
            n_init=10 if intensive else 3,
            random_state=seed,
            reg_covar=1e-6,
        )
    elif method == "ward":
        model = AgglomerativeClustering(n_clusters=clusters, linkage="ward")
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    labels = model.fit_predict(values)
    return labels.astype(int), model


def empirical_centroids(values: np.ndarray, labels: np.ndarray, clusters: int) -> np.ndarray:
    centroids = []
    for cluster in range(clusters):
        members = values[labels == cluster]
        if not len(members):
            raise ValueError(f"Cluster {cluster} has no assigned players")
        centroids.append(members.mean(axis=0))
    return np.vstack(centroids)


def canonicalize_labels(
    values: np.ndarray, labels: np.ndarray, clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assign deterministic IDs by lexicographic empirical-centroid ordering."""
    centroids = empirical_centroids(values, labels, clusters)
    order = sorted(range(clusters), key=lambda cluster: tuple(centroids[cluster]))
    mapping = {old: new for new, old in enumerate(order)}
    canonical = np.array([mapping[int(label)] for label in labels], dtype=int)
    return canonical, empirical_centroids(values, canonical, clusters)


def align_labels(
    reference: np.ndarray, candidate: np.ndarray, clusters: int | None = None
) -> tuple[np.ndarray, dict[int, int]]:
    """Align candidate IDs to reference IDs by maximum-overlap Hungarian matching."""
    if len(reference) != len(candidate):
        raise ValueError("Aligned label arrays must have equal length")
    size = clusters or max(int(reference.max()), int(candidate.max())) + 1
    contingency = np.zeros((size, size), dtype=int)
    for truth, predicted in zip(reference, candidate, strict=True):
        contingency[int(truth), int(predicted)] += 1
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {int(candidate_label): int(reference_label) for reference_label, candidate_label in zip(rows, columns, strict=True)}
    aligned = np.array([mapping.get(int(label), int(label)) for label in candidate], dtype=int)
    return aligned, mapping


def internal_metrics(
    values: np.ndarray,
    labels: np.ndarray,
    model: object | None = None,
) -> dict[str, float]:
    metrics = {
        "silhouette": float(silhouette_score(values, labels)),
        "davies_bouldin": float(davies_bouldin_score(values, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(values, labels)),
    }
    if isinstance(model, GaussianMixture):
        metrics["bic"] = float(model.bic(values))
        metrics["aic"] = float(model.aic(values))
    return metrics


def repeated_seed_stability(
    values: np.ndarray,
    method: str,
    clusters: int,
    seeds: int = SEED_COUNT,
) -> dict[str, float | int]:
    """Compare stochastic fits to a fixed reference after explicit label alignment."""
    reference, _ = fit_labels(values, method, clusters, seed=42, intensive=True)
    scores = []
    for seed in range(seeds):
        candidate, _ = fit_labels(values, method, clusters, seed=seed)
        aligned, _ = align_labels(reference, candidate, clusters)
        scores.append(float(adjusted_rand_score(reference, aligned)))
    return {
        "seeds": seeds,
        "reference_seed": 42,
        "mean_ari": float(np.mean(scores)),
        "median_ari": float(np.median(scores)),
        "minimum_ari": float(np.min(scores)),
    }


def resample_stability(
    values: np.ndarray,
    method: str,
    clusters: int,
    reference_labels: np.ndarray,
    *,
    resamples: int = RESAMPLE_COUNT,
    fraction: float = RESAMPLE_FRACTION,
) -> dict[str, Any]:
    """Measure 80% subsample assignment repeatability against the full-data fit."""
    sample_size = int(np.floor(len(values) * fraction))
    ari_scores: list[float] = []
    player_matches = np.zeros(len(values), dtype=int)
    player_appearances = np.zeros(len(values), dtype=int)
    cluster_matches = np.zeros(clusters, dtype=int)
    cluster_appearances = np.zeros(clusters, dtype=int)
    for seed in range(resamples):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(values), size=sample_size, replace=False))
        candidate, _ = fit_labels(values[indices], method, clusters, seed=seed)
        aligned, _ = align_labels(reference_labels[indices], candidate, clusters)
        ari_scores.append(float(adjusted_rand_score(reference_labels[indices], aligned)))
        matches = aligned == reference_labels[indices]
        player_matches[indices] += matches.astype(int)
        player_appearances[indices] += 1
        for cluster in range(clusters):
            cluster_mask = reference_labels[indices] == cluster
            cluster_matches[cluster] += int(matches[cluster_mask].sum())
            cluster_appearances[cluster] += int(cluster_mask.sum())
    player_stability = player_matches / np.maximum(player_appearances, 1)
    cluster_stability = cluster_matches / np.maximum(cluster_appearances, 1)
    return {
        "resamples": resamples,
        "sample_fraction": fraction,
        "sample_size": sample_size,
        "mean_ari": float(np.mean(ari_scores)),
        "median_ari": float(np.median(ari_scores)),
        "minimum_ari": float(np.min(ari_scores)),
        "cluster_assignment_stability": {
            str(cluster + 1): float(cluster_stability[cluster])
            for cluster in range(clusters)
        },
        "player_assignment_stability": [float(value) for value in player_stability],
        "player_assignment_stability_summary": {
            "mean": float(player_stability.mean()),
            "median": float(np.median(player_stability)),
            "minimum": float(player_stability.min()),
        },
    }


def cluster_sizes(labels: np.ndarray, clusters: int) -> dict[str, int]:
    return {
        str(cluster + 1): int((labels == cluster).sum()) for cluster in range(clusters)
    }


def position_composition(
    labels: np.ndarray, positions: pd.Series, clusters: int
) -> dict[str, Any]:
    output = {}
    position_values = positions.to_numpy()
    for cluster in range(clusters):
        cluster_positions = pd.Series(position_values[labels == cluster])
        counts = {group: int(cluster_positions.eq(group).sum()) for group in POSITION_GROUPS}
        size = len(cluster_positions)
        output[str(cluster + 1)] = {
            "counts": counts,
            "percentages": {
                group: float(100.0 * count / size) for group, count in counts.items()
            },
        }
    return output


def distance_diagnostics(
    values: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> pd.DataFrame:
    distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
    assigned = distances[np.arange(len(values)), labels]
    masked = distances.copy()
    masked[np.arange(len(values)), labels] = np.inf
    second = masked.min(axis=1)
    margin = (second - assigned) / np.maximum(second, EPSILON)
    return pd.DataFrame({
        "assigned_centroid_distance": assigned,
        "second_centroid_distance": second,
        "separation_margin": margin,
    })


def representative_players(
    frame: pd.DataFrame,
    labels: np.ndarray,
    distances: pd.DataFrame,
    clusters: int,
    count: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for cluster in range(clusters):
        indices = np.flatnonzero(labels == cluster)
        ordered = indices[
            np.argsort(distances.loc[indices, "assigned_centroid_distance"].to_numpy())
        ][:count]
        output[str(cluster + 1)] = [
            {
                "player_id": int(frame.iloc[index]["player_id"]),
                "player_name": str(frame.iloc[index]["player_name"]),
                "position_group": str(frame.iloc[index]["position_group"]),
                "centroid_distance": float(
                    distances.iloc[index]["assigned_centroid_distance"]
                ),
            }
            for index in ordered
        ]
    return output


def centroid_report(
    centroids: np.ndarray, feature_names: Sequence[str]
) -> dict[str, dict[str, float]]:
    return {
        str(cluster + 1): {
            feature.removesuffix("_position_z"): float(value)
            for feature, value in zip(feature_names, centroid, strict=True)
        }
        for cluster, centroid in enumerate(centroids)
    }


def distinguishing_features(
    centroids: np.ndarray, feature_names: Sequence[str], count: int = 4
) -> dict[str, list[dict[str, float | str]]]:
    output = {}
    for cluster, centroid in enumerate(centroids):
        indices = np.argsort(np.abs(centroid))[::-1][:count]
        output[str(cluster + 1)] = [
            {
                "feature": feature_names[index].removesuffix("_position_z"),
                "position_z": float(centroid[index]),
            }
            for index in indices
        ]
    return output


def candidate_result(
    values: np.ndarray,
    positions: pd.Series,
    method: str,
    clusters: int,
    *,
    run_resamples: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    labels, model = fit_labels(values, method, clusters, intensive=True)
    labels, centroids = canonicalize_labels(values, labels, clusters)
    sizes = cluster_sizes(labels, clusters)
    result: dict[str, Any] = {
        "method": method,
        "k": clusters,
        **internal_metrics(values, labels, model),
        "cluster_sizes": sizes,
        "small_cluster_concern": min(sizes.values()) < MIN_CLUSTER_SIZE,
        "position_adjusted_rand": float(adjusted_rand_score(positions, labels)),
        "position_normalized_mutual_information": float(
            normalized_mutual_info_score(positions, labels)
        ),
        "position_composition": position_composition(labels, positions, clusters),
        "repeated_seed_stability": (
            repeated_seed_stability(values, method, clusters)
            if method != "ward"
            else {"seeds": 1, "reference_seed": None, "mean_ari": 1.0,
                  "median_ari": 1.0, "minimum_ari": 1.0}
        ),
    }
    if run_resamples:
        result["resample_stability"] = resample_stability(
            values, method, clusters, labels
        )
    return result, labels, centroids


def sensitivity_result(
    frame: pd.DataFrame,
    features: list[str],
    reference_labels: np.ndarray,
    method: str,
    clusters: int,
) -> dict[str, Any]:
    values = frame[features].to_numpy(dtype=float)
    result, labels, centroids = candidate_result(
        values, frame["position_group"], method, clusters, run_resamples=True
    )
    aligned, _ = align_labels(reference_labels, labels, clusters)
    result.update({
        "features": features,
        "assignment_ari_vs_seven_features": float(
            adjusted_rand_score(reference_labels, aligned)
        ),
        "centroids": centroid_report(centroids, features),
    })
    return result


def margin_summary(values: pd.Series) -> dict[str, float]:
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


def run_analysis(
    matrix_path: Path = DEFAULT_MATRIX,
    profiles_path: Path = DEFAULT_PROFILES,
    output_path: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
    *,
    selected_method: str = "kmeans",
    selected_k: int = 2,
    proposed_names: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = pd.read_parquet(matrix_path)
    validate_clustering_matrix(matrix)
    profiles = pd.read_parquet(profiles_path)
    names = profiles[["player_id", "player_name"]]
    frame = matrix.merge(names, on="player_id", how="left", validate="one_to_one")
    if frame["player_name"].isna().any():
        raise ValueError("Every clustering player requires a display name")

    values = frame[NORMALIZED_FEATURES].to_numpy(dtype=float)
    pca, coordinates = fit_pca(values, NORMALIZED_FEATURES)
    candidates = []
    fits: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    preliminary = []
    for method in METHODS:
        for clusters in K_VALUES:
            result, labels, centroids = candidate_result(
                values, frame["position_group"], method, clusters, run_resamples=False
            )
            preliminary.append((result, labels, centroids))
    # Serious solutions are the strongest stable two-cluster result per method,
    # plus K-Means k=3 as the only plausible higher-granularity alternative.
    # Every grid result remains reported and every sub-10 cluster remains flagged.
    for result, labels, centroids in preliminary:
        key = (result["method"], result["k"])
        result["serious_candidate"] = key in SERIOUS_CANDIDATES
        if result["serious_candidate"]:
            result["resample_stability"] = resample_stability(
                values, result["method"], result["k"], labels
            )
        candidates.append(result)
        fits[key] = (labels, centroids)

    for result in candidates:
        if not result["serious_candidate"]:
            continue
        labels, centroids = fits[(result["method"], result["k"])]
        candidate_distances = distance_diagnostics(values, labels, centroids)
        result["cluster_centroids"] = centroid_report(centroids, NORMALIZED_FEATURES)
        result["distinguishing_features"] = distinguishing_features(
            centroids, NORMALIZED_FEATURES
        )
        result["representative_players"] = representative_players(
            frame, labels, candidate_distances, result["k"]
        )

    if selected_method not in METHODS or selected_k not in K_VALUES:
        raise ValueError("Selected solution must come from the controlled candidate grid")
    selected_labels, selected_centroids = fits[(selected_method, selected_k)]
    selected_candidate = next(
        result for result in candidates
        if result["method"] == selected_method and result["k"] == selected_k
    )
    if "resample_stability" not in selected_candidate:
        selected_candidate["resample_stability"] = resample_stability(
            values, selected_method, selected_k, selected_labels
        )
    distances = distance_diagnostics(values, selected_labels, selected_centroids)

    six_features = [feature for feature in NORMALIZED_FEATURES if feature != FINAL_THIRD_FEATURE]
    correlation_features = [
        feature for feature in NORMALIZED_FEATURES
        if feature != CORRELATION_SENSITIVITY_EXCLUDED
    ]
    matrix_b = sensitivity_result(
        frame, six_features, selected_labels, selected_method, selected_k
    )
    correlation_sensitivity = sensitivity_result(
        frame, correlation_features, selected_labels, selected_method, selected_k
    )
    def_mid = frame["position_group"].isin(("DEF", "MID"))
    def_mid_values = values[def_mid]
    def_mid_result, def_mid_labels, _ = candidate_result(
        def_mid_values,
        frame.loc[def_mid, "position_group"],
        selected_method,
        selected_k,
        run_resamples=True,
    )
    def_mid_result["players"] = int(def_mid.sum())
    def_mid_result["assignment_ari_vs_full_overlap"] = float(
        adjusted_rand_score(selected_labels[def_mid], def_mid_labels)
    )

    output = frame[["player_id", "position_group"]].copy()
    output["PC1"] = coordinates[:, 0]
    output["PC2"] = coordinates[:, 1]
    output["PC3"] = coordinates[:, 2]
    output["candidate_cluster_id"] = selected_labels + 1
    output["assigned_centroid_distance"] = distances["assigned_centroid_distance"]
    output["second_centroid_distance"] = distances["second_centroid_distance"]
    output["separation_margin"] = distances["separation_margin"]
    player_stability = selected_candidate["resample_stability"][
        "player_assignment_stability"
    ]
    output["resample_assignment_stability"] = player_stability

    boundary_indices = np.argsort(output["separation_margin"].to_numpy())[:10]
    boundary_players = [
        {
            "player_id": int(frame.iloc[index]["player_id"]),
            "player_name": str(frame.iloc[index]["player_name"]),
            "position_group": str(frame.iloc[index]["position_group"]),
            "cluster_id": int(selected_labels[index] + 1),
            "separation_margin": float(output.iloc[index]["separation_margin"]),
            "resample_assignment_stability": float(player_stability[index]),
        }
        for index in boundary_indices
    ]
    centroids = centroid_report(selected_centroids, NORMALIZED_FEATURES)
    metadata: dict[str, Any] = {
        "version": "V3.2B-analysis",
        "scope": "Analysis artifacts only; no application labels or production persistence.",
        "input": {
            "path": repository_relative_path(matrix_path),
            "rows": len(frame),
            "position_counts": {
                str(group): int(count)
                for group, count in frame["position_group"].value_counts().items()
            },
            "features": NORMALIZED_FEATURES,
            "metadata_only": ["player_id", "position_group"],
            "threshold_policy": {"passes": 50, "carries": 29},
            "reststandardized_across_outfield": False,
        },
        "feature_statistics": feature_statistics(frame),
        "euclidean_distance_contribution": euclidean_distance_contribution(
            values, NORMALIZED_FEATURES
        ),
        "pca": pca_report(pca, NORMALIZED_FEATURES),
        "candidate_models": candidates,
        "serious_candidate_policy": (
            "The strongest stable k=2 solution from each method plus K-Means k=3 "
            "as the plausible higher-granularity alternative."
        ),
        "selected_method": selected_method,
        "selected_k": selected_k,
        "selection_rationale": (
            "K-Means k=2 was selected for its deterministic 25-seed agreement, "
            "strongest internal separation, balanced cluster sizes, substantially "
            "better 80% resample stability than more granular candidates, clear "
            "position-relative centroid interpretation, and negligible broad-position "
            "rediscovery. Stability was weighted above adding extra clusters."
        ),
        "selected_metrics": selected_candidate,
        "cluster_sizes": cluster_sizes(selected_labels, selected_k),
        "cluster_centroids": centroids,
        "distinguishing_features": distinguishing_features(
            selected_centroids, NORMALIZED_FEATURES
        ),
        "position_composition": position_composition(
            selected_labels, frame["position_group"], selected_k
        ),
        "representative_players": representative_players(
            frame, selected_labels, distances, selected_k
        ),
        "boundary_players": boundary_players,
        "separation_margin_distribution": margin_summary(output["separation_margin"]),
        "seven_vs_six_feature_sensitivity": {
            "excluded_feature": FINAL_THIRD_FEATURE,
            **matrix_b,
        },
        "correlation_sensitivity": {
            "excluded_feature": CORRELATION_SENSITIVITY_EXCLUDED,
            **correlation_sensitivity,
        },
        "def_mid_sensitivity": def_mid_result,
        "proposed_archetype_names": proposed_names or DEFAULT_PROPOSED_NAMES,
        "limitations": [
            "Only 11 forwards meet the frozen clustering-only thresholds.",
            "Final-third entries per 100 passes had weak split-half stability in V3.2A.",
            "Clusters describe style relative to broad-position peers, not quality or ability.",
            "Resample assignment stability is repeatability, not true-membership probability.",
            "The source is one 34-match product competition sample centered on Bayer Leverkusen fixtures.",
        ],
    }
    _atomic_parquet(output, output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("V3.2B archetype analysis complete")
    print(f"Input players: {len(frame)}")
    print(f"Selected candidate: {selected_method}, k={selected_k}")
    print(f"Cluster sizes: {metadata['cluster_sizes']}")
    print(f"Player analysis: {output_path}")
    print(f"Analysis metadata: {metadata_path}")
    return output, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--selected-method", choices=METHODS, default="kmeans")
    parser.add_argument("--selected-k", choices=K_VALUES, type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        args.matrix,
        args.profiles,
        args.output,
        args.metadata,
        selected_method=args.selected_method,
        selected_k=args.selected_k,
    )


if __name__ == "__main__":
    main()
