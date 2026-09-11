import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.player_similarity import (
    HIGHER_SUPPORT_MIN_MATCHES,
    METHODOLOGY_VERSION,
    MIN_SIMILARITY_ATTEMPTS,
    MIN_SIMILARITY_CARRIES,
    SIMILARITY_FEATURE_COLUMNS,
    broad_position_group,
    build_similarity_recommendations,
    normalize_by_position,
    prepare_similarity_pool,
    rms_distance,
    run_pipeline,
    sample_support,
    similarity_score,
)


def sample_features() -> pd.DataFrame:
    positions = [
        ("Goalkeeper", "GK"),
        ("Center Back", "DEF"),
        ("Left Center Back", "DEF"),
        ("Right Back", "DEF"),
        ("Left Back", "DEF"),
        ("Center Midfield", "MID"),
        ("Center Attacking Midfield", "MID"),
        ("Center Defensive Midfield", "MID"),
        ("Right Midfield", "MID"),
        ("Center Forward", "FWD"),
        ("Left Wing", "FWD"),
        ("Right Wing", "FWD"),
        ("Left Center Forward", "FWD"),
    ]
    rows: list[dict[str, object]] = []
    for index, (position, group) in enumerate(positions, start=1):
        rows.append(
            {
                "player_id": index,
                "player_name": f"Player {index}",
                "team_name": f"Team {index % 3}",
                "position": position,
                "position_group": group,
                "pass_attempts": MIN_SIMILARITY_ATTEMPTS + index,
                "carries": MIN_SIMILARITY_CARRIES + index,
                "matches_observed": 2 if index % 2 else 5,
                "expected_completion_rate": 0.70 + index * 0.01,
                "pressure_pass_rate": 0.05 + index * 0.006,
                "progressive_pass_rate": 0.04 + index * 0.005,
                "long_pass_rate": 0.08 + index * 0.004,
                "positive_forward_distance_per_100_passes": 400 + index * 20,
                "carry_share_of_actions": 0.15 + index * 0.01,
            }
        )
    rows.append(
        {
            **rows[-1],
            "player_id": 99,
            "player_name": "Low Sample",
            "pass_attempts": MIN_SIMILARITY_ATTEMPTS - 1,
        }
    )
    return pd.DataFrame(rows)


def sample_analysis(features: pd.DataFrame) -> pd.DataFrame:
    eligible = prepare_similarity_pool(features)
    rows: list[dict[str, object]] = []
    for _, player in eligible.iterrows():
        for analysis_type, overlap in (
            ("split_half_player", 0.25),
            ("resample_player_mean", 0.70),
        ):
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "feature_set": "core_6",
                    "distance_method": "rms_euclidean",
                    "normalization": "position",
                    "player_id": player["player_id"],
                    "position_group": player["position_group"],
                    "top_6_overlap": overlap,
                }
            )
    return pd.DataFrame(rows)


def test_broad_position_mapping() -> None:
    assert broad_position_group("Goalkeeper") == "GK"
    assert broad_position_group("Right Wing Back") == "DEF"
    assert broad_position_group("Left Defensive Midfield") == "MID"
    assert broad_position_group("Right Wing") == "FWD"


def test_exact_six_feature_contract_has_no_leakage() -> None:
    assert SIMILARITY_FEATURE_COLUMNS == [
        "expected_completion_rate",
        "pressure_pass_rate",
        "progressive_pass_rate",
        "long_pass_rate",
        "positive_forward_distance_per_100_passes",
        "carry_share_of_actions",
    ]


def test_eligibility_excludes_goalkeepers_and_low_sample_players() -> None:
    pool = prepare_similarity_pool(sample_features())
    assert 1 not in set(pool["player_id"])
    assert 99 not in set(pool["player_id"])
    assert set(pool["position_group"]) == {"DEF", "MID", "FWD"}


def test_population_normalization_is_position_specific() -> None:
    pool = prepare_similarity_pool(sample_features())
    normalized, statistics = normalize_by_position(pool)
    for _, group in normalized.groupby("position_group"):
        np.testing.assert_allclose(
            group[SIMILARITY_FEATURE_COLUMNS].mean(axis=0), 0, atol=1e-12
        )
        np.testing.assert_allclose(
            group[SIMILARITY_FEATURE_COLUMNS].std(axis=0, ddof=0), 1, atol=1e-12
        )
    assert set(statistics) == {"DEF", "MID", "FWD"}


def test_rms_distance_and_score_mapping() -> None:
    left = np.zeros(6)
    right = np.ones(6)
    assert rms_distance(left, left) == 0
    assert rms_distance(left, right) == 1
    assert similarity_score(0, 1.25) == 100
    assert similarity_score(1.25, 1.25) == pytest.approx(50)
    assert similarity_score(2, 1.25) < similarity_score(1, 1.25)


def test_recommendations_are_same_position_deterministic_and_distance_ranked() -> None:
    first, _, _, _ = build_similarity_recommendations(sample_features(), top_n=3)
    second, _, _, _ = build_similarity_recommendations(sample_features(), top_n=3)
    pd.testing.assert_frame_equal(first, second)
    assert not first["player_id"].eq(first["similar_player_id"]).any()
    assert first["position_group"].eq(first["similar_position_group"]).all()
    assert first["same_position_group"].all()
    assert not first["position_group"].eq("GK").any()
    for _, group in first.groupby("player_id"):
        assert group["rms_distance"].is_monotonic_increasing


def test_explanations_are_smallest_z_gaps_and_contributions_sum_to_one() -> None:
    recommendations, normalized, _, _ = build_similarity_recommendations(
        sample_features(), top_n=1
    )
    row = recommendations.iloc[0]
    query = normalized.set_index("player_id").loc[row["player_id"]]
    candidate = normalized.set_index("player_id").loc[row["similar_player_id"]]
    gaps = (query[SIMILARITY_FEATURE_COLUMNS] - candidate[SIMILARITY_FEATURE_COLUMNS]).abs()
    expected = sorted(SIMILARITY_FEATURE_COLUMNS, key=lambda feature: gaps[feature])[:3]
    assert [row[f"closest_feature_{index}"] for index in (1, 2, 3)] == expected
    contributions = json.loads(row["distance_contributions"])
    assert sum(contributions.values()) == pytest.approx(1)


def test_pair_support_uses_weaker_profile_without_changing_score() -> None:
    recommendations, _, _, report = build_similarity_recommendations(
        sample_features(), top_n=3
    )
    expected = recommendations[
        ["query_matches_observed", "candidate_matches_observed"]
    ].min(axis=1)
    assert recommendations["pair_support_matches"].equals(expected)
    assert sample_support(HIGHER_SUPPORT_MIN_MATCHES - 1)[0] == "limited"
    assert sample_support(HIGHER_SUPPORT_MIN_MATCHES)[0] == "higher"
    np.testing.assert_allclose(
        recommendations["similarity_score"],
        recommendations["rms_distance"].map(
            lambda distance: similarity_score(distance, report.d50)
        ),
    )


def test_artifacts_can_be_loaded(tmp_path: Path) -> None:
    input_path = tmp_path / "features.parquet"
    analysis_path = tmp_path / "analysis.parquet"
    output_path = tmp_path / "similarities.parquet"
    normalized_path = tmp_path / "normalized.parquet"
    metadata_path = tmp_path / "metadata.json"
    features = sample_features()
    features.to_parquet(input_path, index=False)
    sample_analysis(features).to_parquet(analysis_path, index=False)

    recommendations, report = run_pipeline(
        input_path, output_path, normalized_path, metadata_path, analysis_path
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert len(pd.read_parquet(output_path)) == len(recommendations)
    assert len(pd.read_parquet(normalized_path)) == report.eligible_players
    assert metadata["methodology_version"] == METHODOLOGY_VERSION
    assert metadata["feature_order"] == SIMILARITY_FEATURE_COLUMNS
    assert metadata["sample_support_calibration"]["selected_threshold"] == 3
