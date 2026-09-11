from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.player_archetype_analysis import (
    NORMALIZED_FEATURES,
    align_labels,
    canonicalize_labels,
    distance_diagnostics,
    empirical_centroids,
    euclidean_distance_contribution,
    feature_statistics,
    fit_labels,
    fit_pca,
    position_composition,
    representative_players,
    resample_stability,
    validate_clustering_matrix,
)


def valid_matrix() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for group, count in (("DEF", 75), ("MID", 47), ("FWD", 11)):
        values = rng.normal(size=(count, len(NORMALIZED_FEATURES)))
        values = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
        for vector in values:
            row = {"player_id": len(rows) + 1, "position_group": group}
            row.update(dict(zip(NORMALIZED_FEATURES, vector, strict=True)))
            rows.append(row)
    return pd.DataFrame(rows)


def separated_values() -> np.ndarray:
    rng = np.random.default_rng(11)
    return np.vstack([
        rng.normal(-4, 0.15, size=(20, 3)),
        rng.normal(0, 0.15, size=(20, 3)),
        rng.normal(4, 0.15, size=(20, 3)),
    ])


def test_clustering_matrix_integrity_and_metadata_exclusion() -> None:
    frame = valid_matrix()
    validate_clustering_matrix(frame)
    assert "player_id" not in NORMALIZED_FEATURES
    assert "position_group" not in NORMALIZED_FEATURES
    assert all(feature.endswith("_position_z") for feature in NORMALIZED_FEATURES)
    frame.loc[0, NORMALIZED_FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_clustering_matrix(frame)


def test_pca_is_deterministic_and_does_not_change_input() -> None:
    values = valid_matrix()[NORMALIZED_FEATURES].to_numpy()
    original = values.copy()
    first_pca, first = fit_pca(values, NORMALIZED_FEATURES)
    second_pca, second = fit_pca(values, NORMALIZED_FEATURES)
    assert np.allclose(first, second)
    assert np.allclose(first_pca.components_, second_pca.components_)
    assert np.array_equal(values, original)


def test_kmeans_is_deterministic_for_fixed_seed() -> None:
    values = separated_values()
    first, _ = fit_labels(values, "kmeans", 3, seed=42)
    second, _ = fit_labels(values, "kmeans", 3, seed=42)
    assert adjusted_partition_equal(first, second)


def adjusted_partition_equal(left: np.ndarray, right: np.ndarray) -> bool:
    aligned, _ = align_labels(left, right, 3)
    return bool(np.array_equal(left, aligned))


def test_label_alignment_recovers_permutation() -> None:
    reference = np.array([0, 0, 1, 1, 2, 2])
    candidate = np.array([2, 2, 0, 0, 1, 1])
    aligned, mapping = align_labels(reference, candidate, 3)
    assert np.array_equal(aligned, reference)
    assert mapping == {2: 0, 0: 1, 1: 2}


def test_centroids_and_canonical_labels() -> None:
    values = np.array([[0.0], [2.0], [10.0], [12.0]])
    labels = np.array([1, 1, 0, 0])
    assert np.allclose(empirical_centroids(values, labels, 2), [[11.0], [1.0]])
    canonical, centroids = canonicalize_labels(values, labels, 2)
    assert canonical.tolist() == [0, 0, 1, 1]
    assert np.allclose(centroids, [[1.0], [11.0]])


def test_resample_stability_is_high_for_separated_clusters() -> None:
    values = separated_values()
    labels, _ = fit_labels(values, "kmeans", 3, seed=42)
    result = resample_stability(
        values, "kmeans", 3, labels, resamples=12, fraction=0.8
    )
    assert result["mean_ari"] > 0.99
    assert result["player_assignment_stability_summary"]["minimum"] > 0.99


def test_distance_margin_and_representative_selection() -> None:
    values = np.array([[0.0], [1.0], [9.0], [10.0]])
    labels = np.array([0, 0, 1, 1])
    centroids = empirical_centroids(values, labels, 2)
    distances = distance_diagnostics(values, labels, centroids)
    assert distances["separation_margin"].between(0, 1).all()
    frame = pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "player_name": ["A", "B", "C", "D"],
        "position_group": ["DEF", "DEF", "MID", "MID"],
    })
    representatives = representative_players(frame, labels, distances, 2, count=1)
    assert representatives["1"][0]["player_id"] in {1, 2}
    assert representatives["2"][0]["player_id"] in {3, 4}


def test_position_composition_reports_counts_and_percentages() -> None:
    labels = np.array([0, 0, 1, 1])
    positions = pd.Series(["DEF", "MID", "MID", "FWD"])
    result = position_composition(labels, positions, 2)
    assert result["1"]["counts"] == {"DEF": 1, "MID": 1, "FWD": 0}
    assert result["1"]["percentages"]["DEF"] == 50.0
    assert result["2"]["percentages"]["FWD"] == 50.0


def test_feature_statistics_verify_position_normalization() -> None:
    result = feature_statistics(valid_matrix())
    for group in ("DEF", "MID", "FWD"):
        for metric in result["by_position"][group].values():
            assert metric["mean"] == pytest.approx(0.0, abs=1e-12)
            assert metric["std"] == pytest.approx(1.0)


def test_feature_set_sensitivity_can_drop_dimension_without_mutation() -> None:
    values = separated_values()
    original = values.copy()
    full, _ = fit_labels(values, "ward", 3)
    reduced, _ = fit_labels(values[:, :2], "ward", 3)
    aligned, _ = align_labels(full, reduced, 3)
    assert np.array_equal(full, aligned)
    assert np.array_equal(values, original)


def test_euclidean_distance_contributions_sum_to_one() -> None:
    values = valid_matrix()[NORMALIZED_FEATURES].to_numpy()
    result = euclidean_distance_contribution(values, NORMALIZED_FEATURES)
    shares = [
        feature["proportion_of_total_squared_distance"]
        for feature in result["by_feature"].values()
    ]
    assert sum(shares) == pytest.approx(1.0)
    assert result["correlated_pair_combined_proportion"] == pytest.approx(2 / 7)
