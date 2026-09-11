from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.player_similarity import DEFAULT_OUTPUT_PATH as PRODUCTION_SIMILARITIES
from analytics.player_similarity_analysis import (
    CORE_FEATURES,
    DEFAULT_FEATURE_OUTPUT,
    DEFAULT_METADATA_OUTPUT,
    DEFAULT_NEIGHBOR_OUTPUT,
    build_match_resamples,
    compare_rankings,
    distance_matrix,
    normalize_features,
    rank_neighbors,
    select_methodology,
    validate_feature_sets,
)


def normalized_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "position_group": ["DEF", "DEF", "DEF", "DEF"],
            "f1": [0.0, 0.0, 1.0, 3.0],
            "f2": [0.0, 0.0, 1.0, -3.0],
        }
    )


def test_style_feature_contract_rejects_quality_or_identity_leakage() -> None:
    validate_feature_sets()
    with pytest.raises(ValueError, match="Prohibited"):
        validate_feature_sets({"bad": [*CORE_FEATURES, "completion_above_expected_pp"]})
    with pytest.raises(ValueError, match="Prohibited"):
        validate_feature_sets({"bad": [*CORE_FEATURES, "archetype_id"]})


def test_position_normalization_is_fit_independently_per_group() -> None:
    source = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "position_group": ["DEF", "DEF", "MID", "MID"],
            "style": [0.0, 2.0, 100.0, 104.0],
        }
    )
    normalized, statistics = normalize_features(source, ["style"], by_position=True)
    for _, group in normalized.groupby("position_group"):
        assert group["style"].mean() == pytest.approx(0.0)
        assert group["style"].std(ddof=0) == pytest.approx(1.0)
    assert statistics["DEF"]["style"] == {"mean": 1.0, "std": 1.0}
    assert statistics["MID"]["style"] == {"mean": 102.0, "std": 2.0}


def test_declared_distance_methods_have_expected_geometry() -> None:
    values = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    assert distance_matrix(values, "rms_euclidean")[0, 1] == pytest.approx(1.0)
    assert distance_matrix(values, "mean_manhattan")[0, 1] == pytest.approx(1.0)
    assert distance_matrix(values, "cosine")[0, 1] == pytest.approx(1.0)


def test_rankings_are_deterministic_same_position_and_identity_free() -> None:
    source = normalized_profiles()
    first, first_d50 = rank_neighbors(source, ["f1", "f2"], "rms_euclidean")
    second, second_d50 = rank_neighbors(source, ["f1", "f2"], "rms_euclidean")
    pd.testing.assert_frame_equal(first, second)
    assert first_d50 == pytest.approx(second_d50)
    assert not first["player_id"].eq(first["similar_player_id"]).any()
    assert first["position_group"].eq(first["similar_position_group"]).all()
    assert first["similarity_score"].between(0, 100).all()
    assert first.loc[first["distance"].eq(0), "similarity_score"].eq(100).all()


def test_score_mapping_uses_cohort_median_distance() -> None:
    rankings, d50 = rank_neighbors(normalized_profiles(), ["f1", "f2"], "rms_euclidean")
    row = rankings.iloc[-1]
    expected = 100 * np.exp(-np.log(2) * row["distance"] / d50)
    assert row["similarity_score"] == pytest.approx(expected)
    assert 100 * np.exp(-np.log(2) * d50 / d50) == pytest.approx(50.0)


def test_exhaustive_neighbor_overlap_is_not_reported_as_stability() -> None:
    rankings, _ = rank_neighbors(normalized_profiles(), ["f1", "f2"], "rms_euclidean")
    compared = compare_rankings(rankings, rankings, top_ks=(2, 3))
    assert compared["top_2_overlap"].eq(1.0).all()
    assert compared["top_3_overlap"].isna().all()


def test_match_resampling_is_deterministic_and_keeps_whole_match_ids() -> None:
    first = build_match_resamples([1, 2, 3, 4, 5], count=4, fraction=0.6, seed=7)
    second = build_match_resamples([1, 2, 3, 4, 5], count=4, fraction=0.6, seed=7)
    assert first == second
    assert all(len(sample) == 3 for sample in first)
    assert all(len(sample) == len(set(sample)) for sample in first)
    assert all(set(sample).issubset({1, 2, 3, 4, 5}) for sample in first)


def test_selection_uses_stability_then_parsimony_and_interpretability() -> None:
    rows = [
        {
            "key": "larger_cosine",
            "feature_set": "larger",
            "features": ["a", "b", "c", "d", "e", "f", "g"],
            "feature_count": 7,
            "distance_method": "cosine",
            "normalization": "position",
            "split_top_6_overlap_mean": 0.30,
            "split_top_6_overlap_median": 0.50,
            "split_rank_spearman_mean": 0.40,
            "resample_top_6_overlap_mean": 0.70,
        },
        {
            "key": "core_rms",
            "feature_set": "core",
            "features": ["a", "b", "c", "d", "e", "f"],
            "feature_count": 6,
            "distance_method": "rms_euclidean",
            "normalization": "position",
            "split_top_6_overlap_mean": 0.395,
            "split_top_6_overlap_median": 0.50,
            "split_rank_spearman_mean": 0.40,
            "resample_top_6_overlap_mean": 0.595,
        },
    ]
    selected = select_methodology(rows)
    assert selected["key"] == "core_rms"
    assert selected["best_observed_stability_score"] == pytest.approx(0.50)
    assert selected["defensible"] is False


def test_research_outputs_cannot_overwrite_production_similarity_artifact() -> None:
    assert DEFAULT_FEATURE_OUTPUT != PRODUCTION_SIMILARITIES
    assert DEFAULT_NEIGHBOR_OUTPUT != PRODUCTION_SIMILARITIES
    assert DEFAULT_METADATA_OUTPUT.name != "player_similarity_metadata.json"
