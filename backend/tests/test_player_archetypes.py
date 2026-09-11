from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.player_archetypes import (
    CARRY_THRESHOLD,
    NORMALIZED_FEATURES,
    PASS_THRESHOLD,
    PRODUCTION_FEATURES,
    distance_diagnostics,
    eligibility_mask,
    fit_production_model,
    normalize_by_position,
    semantic_cluster_mapping,
    validate_source,
)


def source_frame() -> pd.DataFrame:
    rows = []
    for group, shift in (("DEF", 0.0), ("MID", 0.1), ("FWD", 0.2)):
        for index in range(4):
            rows.append(
                {
                    "player_id": len(rows) + 1,
                    "position_group": group,
                    "pass_attempts": PASS_THRESHOLD + index,
                    "carries": CARRY_THRESHOLD + index,
                    "expected_completion_rate": 0.75 + shift + index * 0.01,
                    "pressure_pass_rate": 0.10 + shift + index * 0.01,
                    "progressive_pass_rate": 0.12 + shift + index * 0.02,
                    "long_pass_rate": 0.08 + shift + index * 0.02,
                    "positive_forward_distance_per_100_passes": 500 + shift + index * 20,
                    "carry_share_of_actions": 0.15 + shift + index * 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_final_six_feature_contract_excludes_final_third_and_performance() -> None:
    assert PRODUCTION_FEATURES == [
        "expected_completion_rate",
        "pressure_pass_rate",
        "progressive_pass_rate",
        "long_pass_rate",
        "positive_forward_distance_per_100_passes",
        "carry_share_of_actions",
    ]
    assert "final_third_entries_per_100_passes" not in PRODUCTION_FEATURES
    assert len(NORMALIZED_FEATURES) == 6
    validate_source(source_frame())


def test_position_normalization_uses_population_statistics() -> None:
    normalized, statistics = normalize_by_position(source_frame())
    for group, rows in normalized.groupby("position_group"):
        assert np.allclose(rows[NORMALIZED_FEATURES].mean(), 0.0)
        assert np.allclose(rows[NORMALIZED_FEATURES].std(ddof=0), 1.0)
        assert statistics[group]["expected_completion_rate"]["std"] > 0


def test_semantic_mapping_requires_supported_centroid_directions() -> None:
    centroids = np.array(
        [
            [-0.5, -0.1, 0.7, 0.6, 0.8, -0.3],
            [0.4, 0.1, -0.6, -0.5, -0.7, 0.2],
        ]
    )
    assert semantic_cluster_mapping(centroids) == {
        0: "direct_progressor",
        1: "safe_circulator",
    }
    centroids[0, PRODUCTION_FEATURES.index("expected_completion_rate")] = 0.9
    with pytest.raises(ValueError, match="do not support"):
        semantic_cluster_mapping(centroids)


def test_deterministic_fit_assignment_distances_and_margin() -> None:
    rng = np.random.default_rng(5)
    values = np.vstack([rng.normal(-2, 0.1, (10, 6)), rng.normal(2, 0.1, (10, 6))])
    first = fit_production_model(values)
    second = fit_production_model(values)
    assert np.array_equal(first.labels_, second.labels_)
    assert np.allclose(first.cluster_centers_, second.cluster_centers_)
    diagnostics = distance_diagnostics(values, first.labels_, first.cluster_centers_)
    assert diagnostics["centroid_distance"].ge(0).all()
    assert diagnostics["second_centroid_distance"].ge(
        diagnostics["centroid_distance"]
    ).all()
    assert diagnostics["separation_margin"].between(0, 1).all()


def test_eligibility_excludes_goalkeepers_and_insufficient_samples() -> None:
    frame = source_frame().iloc[:4].copy()
    frame.loc[0, "position_group"] = "GK"
    frame.loc[1, "pass_attempts"] = PASS_THRESHOLD - 1
    frame.loc[2, "carries"] = CARRY_THRESHOLD - 1
    frame.loc[3, PRODUCTION_FEATURES[0]] = np.nan
    assert eligibility_mask(frame).tolist() == [False, False, False, False]
