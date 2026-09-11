"""Build V3.3B production player-style similarity recommendations.

Similarity uses six pre-outcome passing/carrying tendencies. Eligible outfield
players are normalized within broad position and compared only with players in
that same group. Sample support never changes distances, scores, or ranks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analytics.artifact_paths import repository_relative_path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPOSITORY_ROOT / "data" / "processed"
DEFAULT_INPUT_PATH = PROCESSED / "player_similarity_features_v3.parquet"
DEFAULT_ANALYSIS_PATH = PROCESSED / "player_similarity_analysis_v3.parquet"
DEFAULT_NORMALIZED_PATH = PROCESSED / "player_similarity_features_v33b.parquet"
DEFAULT_OUTPUT_PATH = PROCESSED / "player_similarities.parquet"
DEFAULT_METADATA_PATH = REPOSITORY_ROOT / "models" / "player_similarity_metadata.json"

METHODOLOGY_VERSION = "V3.3B"
MIN_SIMILARITY_ATTEMPTS = 50
MIN_SIMILARITY_CARRIES = 29
HIGHER_SUPPORT_MIN_MATCHES = 3
TOP_RECOMMENDATIONS = 10
OUTFIELD_GROUPS = ("DEF", "MID", "FWD")

SIMILARITY_FEATURE_COLUMNS = [
    "expected_completion_rate",
    "pressure_pass_rate",
    "progressive_pass_rate",
    "long_pass_rate",
    "positive_forward_distance_per_100_passes",
    "carry_share_of_actions",
]

PROHIBITED_SIMILARITY_COLUMNS = {
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "archetype_id",
    "archetype_name",
    "completion_above_expected_pp",
    "goals_minus_xg",
    "attacking_value_per_100_actions",
    "pass_value_per_100_passes",
    "carry_value_per_100_carries",
    "progressive_value_per_100_actions",
    "pressure_value_per_100_actions",
    "completed",
    "goal",
    "pass_outcome",
}

POSITION_GROUP_MAPPING = {
    "Goalkeeper": "GK",
    "Center Back": "DEF",
    "Left Back": "DEF",
    "Left Center Back": "DEF",
    "Left Wing Back": "DEF",
    "Right Back": "DEF",
    "Right Center Back": "DEF",
    "Right Wing Back": "DEF",
    "Center Attacking Midfield": "MID",
    "Center Defensive Midfield": "MID",
    "Left Attacking Midfield": "MID",
    "Left Center Midfield": "MID",
    "Left Defensive Midfield": "MID",
    "Left Midfield": "MID",
    "Right Attacking Midfield": "MID",
    "Right Center Midfield": "MID",
    "Right Defensive Midfield": "MID",
    "Right Midfield": "MID",
    "Center Forward": "FWD",
    "Left Center Forward": "FWD",
    "Left Wing": "FWD",
    "Right Center Forward": "FWD",
    "Right Wing": "FWD",
}

OUTPUT_COLUMNS = [
    "player_id",
    "player_name",
    "team_name",
    "position_group",
    "similar_player_id",
    "similar_player_name",
    "similar_team_name",
    "similar_position_group",
    "rank",
    "rms_distance",
    "similarity_score",
    "same_position_group",
    "closest_feature_1",
    "closest_feature_2",
    "closest_feature_3",
    "query_matches_observed",
    "candidate_matches_observed",
    "pair_support_matches",
    "sample_support",
    "sample_support_explanation",
    "methodology_version",
    "distance_contributions",
]


@dataclass(frozen=True)
class SimilarityReport:
    total_profiles: int
    eligible_players: int
    eligible_by_position_group: dict[str, int]
    recommendation_rows: int
    d50: float
    minimum_similarity_score: float
    median_similarity_score: float
    maximum_similarity_score: float
    limited_support_rows: int
    higher_support_rows: int


def broad_position_group(position: object) -> str:
    """Map an inspected StatsBomb display position into GK/DEF/MID/FWD."""
    if position is None or pd.isna(position):
        raise ValueError("Position cannot be missing for similarity grouping")
    value = str(position).strip()
    try:
        return POSITION_GROUP_MAPPING[value]
    except KeyError as error:
        raise ValueError(f"Unmapped StatsBomb position: {value!r}") from error


def validate_similarity_features() -> None:
    """Prevent performance, identity, outcome, and archetype leakage."""
    if len(SIMILARITY_FEATURE_COLUMNS) != 6:
        raise ValueError("V3.3B requires exactly six style features")
    if len(SIMILARITY_FEATURE_COLUMNS) != len(set(SIMILARITY_FEATURE_COLUMNS)):
        raise ValueError("Similarity features must be unique")
    prohibited = PROHIBITED_SIMILARITY_COLUMNS.intersection(SIMILARITY_FEATURE_COLUMNS)
    if prohibited:
        raise ValueError(f"Prohibited similarity features configured: {sorted(prohibited)}")


def prepare_similarity_pool(features: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen 50-pass/29-carry outfield eligibility policy."""
    validate_similarity_features()
    required = {
        "player_id",
        "player_name",
        "team_name",
        "position",
        "position_group",
        "pass_attempts",
        "carries",
        "matches_observed",
        *SIMILARITY_FEATURE_COLUMNS,
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Similarity features are missing columns: {sorted(missing)}")
    pool = features.loc[
        features["position_group"].isin(OUTFIELD_GROUPS)
        & features["pass_attempts"].ge(MIN_SIMILARITY_ATTEMPTS)
        & features["carries"].ge(MIN_SIMILARITY_CARRIES)
        & features[SIMILARITY_FEATURE_COLUMNS].notna().all(axis=1)
    ].copy()
    if pool["player_id"].duplicated().any():
        raise ValueError("Eligible similarity players must be unique by player_id")
    if not set(pool["position_group"]).issubset(OUTFIELD_GROUPS):
        raise AssertionError("Goalkeepers cannot enter the production similarity cohort")
    return pool.sort_values("player_id", kind="stable").reset_index(drop=True)


def normalize_by_position(
    pool: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, float]]]]:
    """Fit population z-scores separately within DEF, MID, and FWD."""
    normalized = pool[
        [
            "player_id",
            "player_name",
            "team_name",
            "position",
            "position_group",
            "pass_attempts",
            "carries",
            "matches_observed",
        ]
    ].copy()
    statistics: dict[str, dict[str, dict[str, float]]] = {}
    for position_group, group in pool.groupby("position_group", sort=True):
        statistics[str(position_group)] = {}
        for feature in SIMILARITY_FEATURE_COLUMNS:
            values = pd.to_numeric(group[feature], errors="raise").astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if not np.isfinite(std) or std <= 0:
                raise ValueError(f"Cannot normalize constant feature {feature} in {position_group}")
            statistics[str(position_group)][feature] = {"mean": mean, "std": std}
            normalized.loc[group.index, feature] = (values - mean) / std
            normalized.loc[group.index, f"{feature}_raw"] = values
    if normalized[SIMILARITY_FEATURE_COLUMNS].isna().any().any():
        raise AssertionError("Position normalization produced missing values")
    return normalized, statistics


def rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return root-mean-square Euclidean distance between equal-length vectors."""
    if left.shape != right.shape or left.size == 0:
        raise ValueError("RMS distance requires non-empty vectors with matching shapes")
    return float(np.sqrt(np.mean(np.square(left - right))))


def production_d50(normalized: pd.DataFrame) -> float:
    """Return the median unique same-position pair distance in the eligible cohort."""
    values: list[float] = []
    for _, group in normalized.groupby("position_group", sort=True):
        vectors = group[SIMILARITY_FEATURE_COLUMNS].to_numpy(dtype=float)
        for left in range(len(vectors)):
            for right in range(left + 1, len(vectors)):
                distance = rms_distance(vectors[left], vectors[right])
                if np.isfinite(distance) and distance > 0:
                    values.append(distance)
    if not values:
        raise ValueError("A positive same-position pair distance is required")
    return float(np.median(values))


def similarity_score(distance: float, d50: float) -> float:
    """Map distance to a non-probabilistic 0–100 cohort-calibrated index."""
    if distance < 0 or d50 <= 0:
        raise ValueError("Distance must be nonnegative and D50 must be positive")
    return float(100.0 * np.exp(-np.log(2.0) * distance / d50))


def sample_support(pair_support_matches: int) -> tuple[str, str]:
    """Describe evidence volume without modifying the style score."""
    if pair_support_matches < 0:
        raise ValueError("Observed match count cannot be negative")
    if pair_support_matches < HIGHER_SUPPORT_MIN_MATCHES:
        return (
            "limited",
            "Limited sample: one or both profiles use fewer than 3 observed matches.",
        )
    return (
        "higher",
        "Higher sample support: both profiles use at least 3 observed matches.",
    )


def build_similarity_recommendations(
    features: pd.DataFrame,
    top_n: int = TOP_RECOMMENDATIONS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, dict[str, float]]],
    SimilarityReport,
]:
    """Rank deterministic same-position neighbors using pure six-feature distance."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    pool = prepare_similarity_pool(features)
    normalized, statistics = normalize_by_position(pool)
    d50 = production_d50(normalized)
    rows: list[dict[str, Any]] = []
    for player_index, player in normalized.iterrows():
        candidates = normalized.loc[
            normalized["position_group"].eq(player["position_group"])
            & normalized.index.to_series().ne(player_index)
        ]
        player_vector = player[SIMILARITY_FEATURE_COLUMNS].to_numpy(dtype=float)
        ranked: list[tuple[float, int, pd.Series[Any], np.ndarray]] = []
        for _, candidate in candidates.iterrows():
            candidate_vector = candidate[SIMILARITY_FEATURE_COLUMNS].to_numpy(dtype=float)
            ranked.append(
                (
                    rms_distance(player_vector, candidate_vector),
                    int(candidate["player_id"]),
                    candidate,
                    candidate_vector,
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1]))
        for rank, (distance, _, candidate, candidate_vector) in enumerate(
            ranked[:top_n], start=1
        ):
            gaps = np.abs(player_vector - candidate_vector)
            closest_indices = sorted(
                range(len(gaps)), key=lambda index: (float(gaps[index]), index)
            )[:3]
            squared = np.square(gaps)
            total_squared = float(squared.sum())
            contributions = (
                squared / total_squared if total_squared > 0 else np.zeros_like(squared)
            )
            query_matches = int(player["matches_observed"])
            candidate_matches = int(candidate["matches_observed"])
            pair_matches = min(query_matches, candidate_matches)
            support, support_explanation = sample_support(pair_matches)
            rows.append(
                {
                    "player_id": int(player["player_id"]),
                    "player_name": str(player["player_name"]),
                    "team_name": str(player["team_name"]),
                    "position_group": str(player["position_group"]),
                    "similar_player_id": int(candidate["player_id"]),
                    "similar_player_name": str(candidate["player_name"]),
                    "similar_team_name": str(candidate["team_name"]),
                    "similar_position_group": str(candidate["position_group"]),
                    "rank": rank,
                    "rms_distance": distance,
                    "similarity_score": similarity_score(distance, d50),
                    "same_position_group": True,
                    "closest_feature_1": SIMILARITY_FEATURE_COLUMNS[closest_indices[0]],
                    "closest_feature_2": SIMILARITY_FEATURE_COLUMNS[closest_indices[1]],
                    "closest_feature_3": SIMILARITY_FEATURE_COLUMNS[closest_indices[2]],
                    "query_matches_observed": query_matches,
                    "candidate_matches_observed": candidate_matches,
                    "pair_support_matches": pair_matches,
                    "sample_support": support,
                    "sample_support_explanation": support_explanation,
                    "methodology_version": METHODOLOGY_VERSION,
                    "distance_contributions": json.dumps(
                        {
                            feature: float(contribution)
                            for feature, contribution in zip(
                                SIMILARITY_FEATURE_COLUMNS, contributions, strict=True
                            )
                        },
                        sort_keys=True,
                    ),
                }
            )
    recommendations = pd.DataFrame.from_records(rows, columns=OUTPUT_COLUMNS)
    validate_recommendations(recommendations, pool, d50)
    counts = pool["position_group"].value_counts().reindex(OUTFIELD_GROUPS, fill_value=0)
    report = SimilarityReport(
        total_profiles=len(features),
        eligible_players=len(pool),
        eligible_by_position_group={key: int(value) for key, value in counts.items()},
        recommendation_rows=len(recommendations),
        d50=d50,
        minimum_similarity_score=float(recommendations["similarity_score"].min()),
        median_similarity_score=float(recommendations["similarity_score"].median()),
        maximum_similarity_score=float(recommendations["similarity_score"].max()),
        limited_support_rows=int(recommendations["sample_support"].eq("limited").sum()),
        higher_support_rows=int(recommendations["sample_support"].eq("higher").sum()),
    )
    return recommendations, normalized, statistics, report


def validate_recommendations(
    recommendations: pd.DataFrame,
    pool: pd.DataFrame,
    d50: float,
) -> None:
    """Enforce production similarity invariants."""
    if list(recommendations.columns) != OUTPUT_COLUMNS:
        raise AssertionError("Similarity output columns do not match the schema")
    if recommendations.empty:
        raise AssertionError("Production recommendations cannot be empty")
    eligible_ids = set(pool["player_id"].astype(int))
    if not set(recommendations["player_id"].astype(int)).issubset(eligible_ids):
        raise AssertionError("An ineligible player received recommendations")
    if recommendations["player_id"].eq(recommendations["similar_player_id"]).any():
        raise AssertionError("Players cannot be recommended to themselves")
    if recommendations.duplicated(["player_id", "similar_player_id"]).any():
        raise AssertionError("Neighbor pairs must be unique")
    if recommendations.duplicated(["player_id", "rank"]).any():
        raise AssertionError("Ranks must be unique within each player")
    if not recommendations["same_position_group"].all():
        raise AssertionError("V3.3B supports same-position comparisons only")
    if not recommendations["position_group"].eq(
        recommendations["similar_position_group"]
    ).all():
        raise AssertionError("Cross-position recommendations are prohibited")
    if recommendations["position_group"].eq("GK").any():
        raise AssertionError("Goalkeepers are excluded from V3.3B")
    if not recommendations["rms_distance"].ge(0).all():
        raise AssertionError("RMS distance cannot be negative")
    if not recommendations["similarity_score"].between(0, 100).all():
        raise AssertionError("Similarity score must remain within [0, 100]")
    expected_scores = recommendations["rms_distance"].map(
        lambda distance: similarity_score(float(distance), d50)
    )
    if not np.allclose(expected_scores, recommendations["similarity_score"]):
        raise AssertionError("Similarity scores do not match the distance-only formula")
    expected_pair_support = recommendations[
        ["query_matches_observed", "candidate_matches_observed"]
    ].min(axis=1)
    if not recommendations["pair_support_matches"].eq(expected_pair_support).all():
        raise AssertionError("Pair support must use the weaker-supported profile")
    for _, group in recommendations.groupby("player_id", sort=False):
        if group["rank"].tolist() != list(range(1, len(group) + 1)):
            raise AssertionError("Ranks must be consecutive from one")
        if not group["rms_distance"].is_monotonic_increasing:
            raise AssertionError("Ranks must follow ascending pure style distance")


def calibrate_sample_support(
    analysis: pd.DataFrame,
    pool: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize V3.3A stability at predeclared observed-match boundaries."""
    required = {
        "analysis_type",
        "feature_set",
        "distance_method",
        "normalization",
        "player_id",
        "position_group",
        "top_6_overlap",
    }
    missing = required.difference(analysis.columns)
    if missing:
        raise ValueError(f"V3.3A analysis is missing columns: {sorted(missing)}")
    selected = analysis.loc[
        analysis["feature_set"].eq("core_6")
        & analysis["distance_method"].eq("rms_euclidean")
        & analysis["normalization"].eq("position")
    ].merge(
        pool[["player_id", "position_group", "matches_observed"]],
        on=["player_id", "position_group"],
        how="inner",
        validate="many_to_one",
    )
    thresholds: dict[str, Any] = {}
    for threshold in (2, 3, 4, 5, 6, 8, 10, 15):
        players = pool.loc[pool["matches_observed"].ge(threshold)]
        stability: dict[str, Any] = {}
        for analysis_type in ("split_half_player", "resample_player_mean"):
            values = selected.loc[
                selected["analysis_type"].eq(analysis_type)
                & selected["matches_observed"].ge(threshold),
                "top_6_overlap",
            ].dropna()
            stability[analysis_type] = {
                "measurable_players": len(values),
                "mean_top_6_overlap": float(values.mean()) if len(values) else None,
                "median_top_6_overlap": float(values.median()) if len(values) else None,
            }
        counts = players["position_group"].value_counts().reindex(OUTFIELD_GROUPS, fill_value=0)
        thresholds[str(threshold)] = {
            "eligible_players": len(players),
            "position_counts": {key: int(value) for key, value in counts.items()},
            "mean_observed_matches": (
                float(players["matches_observed"].mean()) if len(players) else None
            ),
            "median_observed_matches": (
                float(players["matches_observed"].median()) if len(players) else None
            ),
            "stability": stability,
            "meaningful_conclusion": len(players) >= 20,
        }
    return {
        "candidate_thresholds": thresholds,
        "selected_threshold": HIGHER_SUPPORT_MIN_MATCHES,
        "tiers": {
            "limited": f"pair_support_matches < {HIGHER_SUPPORT_MIN_MATCHES}",
            "higher": f"pair_support_matches >= {HIGHER_SUPPORT_MIN_MATCHES}",
        },
        "reason": (
            "Three matches is the first candidate boundary with materially stronger "
            "chronological and resampling overlap. Only 20 current players clear it, "
            "so the data does not support finer moderate/high tiers."
        ),
    }


def build_metadata(
    report: SimilarityReport,
    statistics: dict[str, dict[str, dict[str, float]]],
    support_calibration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "feature_names": SIMILARITY_FEATURE_COLUMNS,
        "feature_order": SIMILARITY_FEATURE_COLUMNS,
        "eligibility": {
            "position_groups": list(OUTFIELD_GROUPS),
            "minimum_pass_attempts": MIN_SIMILARITY_ATTEMPTS,
            "minimum_carries": MIN_SIMILARITY_CARRIES,
            "goalkeepers_available": False,
        },
        "candidate_rule": "eligible players in the same broad position group only",
        "normalization": "population z-scores fitted separately within DEF, MID, and FWD",
        "normalization_statistics": statistics,
        "distance_definition": "sqrt(mean((z_player - z_candidate)^2))",
        "d50": report.d50,
        "display_score_formula": "100 * exp(-ln(2) * rms_distance / D50)",
        "display_score_interpretation": (
            "100 is identical; 50 is the median same-position pair distance; not a probability"
        ),
        "ranking_rule": "ascending RMS distance, then player_id; sample support is excluded",
        "closest_style_dimensions": "three smallest absolute position-z-score gaps",
        "distance_contributions": "squared feature gap divided by total squared gap",
        "sample_support_calibration": support_calibration,
        "cohort_counts": {
            "total_profiles": report.total_profiles,
            "eligible_players": report.eligible_players,
            "by_position_group": report.eligible_by_position_group,
        },
        "known_limitation": (
            "Match coverage is uneven; nearest-neighbor ranks for limited-sample profiles "
            "may change as additional matches are observed."
        ),
        "excluded_concepts": [
            "performance",
            "outcomes",
            "player identity",
            "team identity",
            "archetype",
            "sample support",
        ],
        "report": asdict(report),
    }


def print_report(report: SimilarityReport) -> None:
    print("V3.3B player-style similarity generation complete")
    print(f"Total profiles: {report.total_profiles:,}")
    print(f"Eligible players: {report.eligible_players:,}")
    print(f"Eligible by position: {report.eligible_by_position_group}")
    print(f"Production D50: {report.d50:.6f}")
    print(f"Recommendation rows: {report.recommendation_rows:,}")
    print(
        "Similarity score minimum/median/maximum: "
        f"{report.minimum_similarity_score:.2f} / "
        f"{report.median_similarity_score:.2f} / "
        f"{report.maximum_similarity_score:.2f}"
    )
    print(
        "Sample-support rows (limited/higher): "
        f"{report.limited_support_rows:,} / {report.higher_support_rows:,}"
    )


def run_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    normalized_path: Path = DEFAULT_NORMALIZED_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    analysis_path: Path = DEFAULT_ANALYSIS_PATH,
) -> tuple[pd.DataFrame, SimilarityReport]:
    features = pd.read_parquet(input_path)
    recommendations, normalized, statistics, report = build_similarity_recommendations(features)
    analysis = pd.read_parquet(analysis_path)
    pool = prepare_similarity_pool(features)
    calibration = calibrate_sample_support(analysis, pool)
    metadata = build_metadata(report, statistics, calibration)
    metadata.update(
        {
            "input_path": repository_relative_path(input_path),
            "normalized_feature_path": repository_relative_path(normalized_path),
            "output_path": repository_relative_path(output_path),
            "analysis_path": repository_relative_path(analysis_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_parquet(output_path, index=False)
    normalized.to_parquet(normalized_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print_report(report)
    print(f"Saved similarities: {output_path}")
    print(f"Saved normalized features: {normalized_path}")
    print(f"Saved metadata: {metadata_path}")
    return recommendations, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.input, args.output, args.normalized, args.metadata, args.analysis)


if __name__ == "__main__":
    main()
