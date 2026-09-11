"""V4.0 research for team style and player-to-role fit feasibility.

This module is deliberately research-only. It builds match-grouped diagnostics from
the frozen product cohort and reuses the exact V3.3B six-feature style space. It does
not write to PostgreSQL or alter any production model, API, or frontend artifact.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analytics.artifact_paths import repository_relative_path
from analytics.player_feature_registry import PERFORMANCE_FEATURES
from analytics.player_similarity import (
    OUTFIELD_GROUPS,
    PROHIBITED_SIMILARITY_COLUMNS,
    SIMILARITY_FEATURE_COLUMNS,
    normalize_by_position,
    prepare_similarity_pool,
)
from analytics.player_similarity_analysis import recompute_similarity_features
from analytics.player_style_stability import (
    assign_player_match_halves,
    build_match_chronology,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

DEFAULT_PASSES = PROCESSED / "pass_oof_predictions.parquet"
DEFAULT_ACTIONS = PROCESSED / "attacking_actions.parquet"
DEFAULT_SHOTS = PROCESSED / "product_shot_predictions.parquet"
DEFAULT_PLAYERS = PROCESSED / "player_similarity_features_v3.parquet"
DEFAULT_ARCHETYPES = PROCESSED / "player_archetypes.parquet"
DEFAULT_CHRONOLOGY = PROCESSED / "product_match_chronology.parquet"
DEFAULT_TEAM_MATCH_OUTPUT = PROCESSED / "team_match_style_profiles_v40.parquet"
DEFAULT_TEAM_RESEARCH_OUTPUT = PROCESSED / "leverkusen_team_style_research_v40.parquet"
DEFAULT_ROLE_OUTPUT = PROCESSED / "leverkusen_role_profiles_v40.parquet"
DEFAULT_FIT_OUTPUT = PROCESSED / "player_role_fit_research_v40.parquet"
DEFAULT_COVERAGE_OUTPUT = PROCESSED / "team_coverage_v40.parquet"
DEFAULT_METADATA_OUTPUT = MODELS / "v40_team_role_research.json"

METHODOLOGY_VERSION = "V4.0"
TARGET_TEAM_NAME = "Bayer Leverkusen"
LONG_PASS_LENGTH = 30.0
FINAL_THIRD_X = 80.0
RESAMPLE_COUNT = 100
RESAMPLE_FRACTION = 0.8
RANDOM_SEED = 42

FIT_FEATURES = tuple(SIMILARITY_FEATURE_COLUMNS)
OPTIONAL_STYLE_FEATURES = (
    "progressive_carry_rate",
    "progressive_action_rate",
    "pressure_action_rate",
)
DESCRIPTIVE_FEATURES = (
    *FIT_FEATURES,
    "average_forward_distance",
    "final_third_entries_per_100_passes",
    *OPTIONAL_STYLE_FEATURES,
    "shots_per_match",
    "xg_per_shot",
    "xg_per_match",
    "attacking_value_per_100_actions",
)
DISTANCE_METHODS = ("rms", "manhattan", "cosine")
ROLE_GROUPS = tuple(OUTFIELD_GROUPS)

IDENTITY_COLUMNS = {
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "position_group",
}


def validate_fit_contract(features: Sequence[str] = FIT_FEATURES) -> None:
    """Enforce the frozen six-dimensional style-only V3.3B contract."""
    if tuple(features) != FIT_FEATURES or len(features) != 6:
        raise ValueError("V4.0 primary fit must use the exact ordered V3.3B six features")
    prohibited = (
        set(PERFORMANCE_FEATURES)
        | PROHIBITED_SIMILARITY_COLUMNS
        | IDENTITY_COLUMNS
        | {"archetype", "archetype_id", "archetype_name"}
    )
    leaked = prohibited.intersection(features)
    if leaked:
        raise ValueError(f"Non-style fields entered the role-fit vector: {sorted(leaked)}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).div(denominator.replace(0, np.nan).astype(float))


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def enrich_actions(actions: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Attach the frozen primary broad position and display metadata to actions."""
    lookup = players[
        ["player_id", "player_name", "team_name", "position", "position_group"]
    ].drop_duplicates("player_id")
    if lookup["player_id"].duplicated().any():
        raise ValueError("Player metadata must be unique")
    result = actions.merge(lookup, on="player_id", how="left", validate="many_to_one")
    # Four opponent carries belong to players without a frozen V3 product profile.
    # Preserve them for team-level counts, but never infer an outfield role for them.
    result["player_name"] = result["player_name"].fillna("Unknown")
    result["position"] = result["position"].fillna("Unknown")
    result["position_group"] = result["position_group"].fillna("Unknown")
    return result


def _aggregate_profiles(
    passes: pd.DataFrame,
    actions: pd.DataFrame,
    group_keys: list[str],
    shots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate exact V3.3B features plus explicitly descriptive diagnostics."""
    if passes.empty:
        return pd.DataFrame(columns=[*group_keys, *DESCRIPTIVE_FEATURES])

    pass_source = passes.copy()
    pass_source["long_pass"] = pass_source["pass_length"].ge(LONG_PASS_LENGTH)
    pass_source["positive_forward"] = pass_source["forward_distance"].clip(lower=0)
    pass_source["final_third_entry"] = pass_source["start_x"].lt(FINAL_THIRD_X) & pass_source[
        "end_x"
    ].ge(FINAL_THIRD_X)
    pass_profile = pass_source.groupby(group_keys, observed=True, sort=True).agg(
        pass_attempts=("match_id", "size"),
        matches_observed=("match_id", "nunique"),
        expected_completion_rate=("expected_completion", "mean"),
        pressure_pass_rate=("under_pressure", "mean"),
        progressive_pass_rate=("progressive", "mean"),
        long_pass_rate=("long_pass", "mean"),
        positive_forward_distance=("positive_forward", "sum"),
        average_forward_distance=("forward_distance", "mean"),
        final_third_entries=("final_third_entry", "sum"),
    ).reset_index()
    pass_profile["positive_forward_distance_per_100_passes"] = 100 * _safe_divide(
        pass_profile["positive_forward_distance"], pass_profile["pass_attempts"]
    )
    pass_profile["final_third_entries_per_100_passes"] = 100 * _safe_divide(
        pass_profile["final_third_entries"], pass_profile["pass_attempts"]
    )

    action_source = actions.copy()
    action_source["is_carry"] = action_source["action_type"].eq("Carry")
    action_source["progressive_carry"] = action_source["is_carry"] & action_source[
        "progressive"
    ].fillna(False)
    action_profile = action_source.groupby(group_keys, observed=True, sort=True).agg(
        attacking_actions=("action_id", "size"),
        carries=("is_carry", "sum"),
        progressive_carries=("progressive_carry", "sum"),
        progressive_actions=("progressive", "sum"),
        pressure_actions=("under_pressure", "sum"),
        attacking_value=("attacking_value", "sum"),
        action_players=("player_id", "nunique"),
    ).reset_index()
    action_profile["carry_share_of_actions"] = _safe_divide(
        action_profile["carries"], action_profile["attacking_actions"]
    )
    action_profile["progressive_carry_rate"] = _safe_divide(
        action_profile["progressive_carries"], action_profile["carries"]
    )
    action_profile["progressive_action_rate"] = _safe_divide(
        action_profile["progressive_actions"], action_profile["attacking_actions"]
    )
    action_profile["pressure_action_rate"] = _safe_divide(
        action_profile["pressure_actions"], action_profile["attacking_actions"]
    )
    action_profile["attacking_value_per_100_actions"] = 100 * _safe_divide(
        action_profile["attacking_value"], action_profile["attacking_actions"]
    )
    result = pass_profile.merge(action_profile, on=group_keys, how="left", validate="one_to_one")

    if shots is not None:
        shot_source = shots.copy()
        shot_source["modeled_shot"] = shot_source["expected_goal"].notna()
        shot_source["shot_xg"] = shot_source["expected_goal"].fillna(0.0)
        shot_profile = shot_source.groupby(group_keys, observed=True, sort=True).agg(
            shots=("shot_id", "size"),
            modeled_shots=("modeled_shot", "sum"),
            goals=("goal", "sum"),
            total_xg=("shot_xg", "sum"),
        ).reset_index()
        result = result.merge(shot_profile, on=group_keys, how="left", validate="one_to_one")
    for column in ("shots", "modeled_shots", "goals", "total_xg"):
        if column not in result:
            result[column] = 0
        result[column] = result[column].fillna(0)
    result["shots_per_match"] = _safe_divide(result["shots"], result["matches_observed"])
    result["xg_per_shot"] = _safe_divide(result["total_xg"], result["modeled_shots"])
    result["xg_per_match"] = _safe_divide(result["total_xg"], result["matches_observed"])
    return result


def build_team_match_profiles(
    passes: pd.DataFrame,
    actions: pd.DataFrame,
    shots: pd.DataFrame,
    chronology: pd.DataFrame,
) -> pd.DataFrame:
    """Build exactly one contextual team profile for each side in each match."""
    result = _aggregate_profiles(passes, actions, ["match_id", "team_id"], shots)
    team_names = pd.concat(
        [
            passes[["team_id", "team_name"]],
            shots[["team_id", "team_name"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    if team_names["team_id"].duplicated().any():
        raise ValueError("A team_id maps to multiple team names")
    result = result.merge(team_names, on="team_id", how="left", validate="many_to_one")
    pairs = result[["match_id", "team_id", "team_name"]].merge(
        result[["match_id", "team_id", "team_name"]].rename(
            columns={"team_id": "opponent_id", "team_name": "opponent_name"}
        ),
        on="match_id",
    )
    pairs = pairs.loc[pairs["team_id"].ne(pairs["opponent_id"])]
    result = result.merge(
        pairs[["match_id", "team_id", "opponent_id", "opponent_name"]],
        on=["match_id", "team_id"],
        how="left",
        validate="one_to_one",
    )
    dates = chronology[["match_id", "match_date", "kick_off", "home_team", "away_team"]]
    result = result.merge(dates, on="match_id", how="left", validate="many_to_one")
    result["home_away"] = np.where(result["team_name"].eq(result["home_team"]), "home", "away")
    if result.duplicated(["match_id", "team_id"]).any():
        raise AssertionError("Team-match profiles must be unique")
    if result["opponent_id"].isna().any() or result["match_date"].isna().any():
        raise AssertionError("Every team-match profile requires opponent and chronology")
    return result.sort_values(["match_date", "kick_off", "match_id", "team_id"], kind="stable")


def build_team_coverage(
    passes: pd.DataFrame, actions: pd.DataFrame, shots: pd.DataFrame
) -> pd.DataFrame:
    """Audit actual match/event coverage without implying full-season opponents."""
    names = passes[["team_id", "team_name"]].drop_duplicates("team_id")
    pass_counts = passes.groupby("team_id").agg(
        matches_observed=("match_id", "nunique"),
        passes=("match_id", "size"),
        pass_players=("player_id", "nunique"),
    )
    action_counts = actions.groupby("team_id").agg(
        attacking_actions=("action_id", "size"),
        carries=("action_type", lambda values: int(values.eq("Carry").sum())),
        action_players=("player_id", "nunique"),
    )
    shot_counts = shots.groupby("team_id").agg(
        shots=("shot_id", "size"), shot_players=("player_id", "nunique")
    )
    result = names.merge(pass_counts, on="team_id", validate="one_to_one")
    result = result.merge(action_counts, on="team_id", how="left", validate="one_to_one")
    result = result.merge(shot_counts, on="team_id", how="left", validate="one_to_one")
    result["unique_players"] = result[
        ["pass_players", "action_players", "shot_players"]
    ].max(axis=1)
    result["total_events_available"] = result["attacking_actions"] + result["shots"]
    columns = [
        "team_id", "team_name", "matches_observed", "total_events_available",
        "passes", "carries", "shots", "attacking_actions", "unique_players",
    ]
    return result[columns].sort_values("team_name", kind="stable").reset_index(drop=True)


def deterministic_match_resamples(
    match_ids: Iterable[int],
    count: int = RESAMPLE_COUNT,
    fraction: float = RESAMPLE_FRACTION,
    seed: int = RANDOM_SEED,
) -> list[tuple[int, ...]]:
    """Sample whole matches without replacement, deterministically."""
    values = np.array(sorted({int(value) for value in match_ids}), dtype=int)
    if len(values) < 2 or not 0 < fraction <= 1 or count < 1:
        raise ValueError("Resampling requires >=2 matches, positive count, and 0<fraction<=1")
    size = max(1, round(len(values) * fraction))
    rng = np.random.default_rng(seed)
    return [tuple(sorted(rng.choice(values, size=size, replace=False).tolist())) for _ in range(count)]


def normalize_profile(
    profile: pd.Series | dict[str, float],
    position_group: str,
    statistics: dict[str, dict[str, dict[str, float]]],
) -> np.ndarray:
    """Apply the exact eligible-player V3.3B position population statistics."""
    if position_group not in ROLE_GROUPS:
        raise ValueError("Role normalization is outfield-only")
    return np.array(
        [
            (float(profile[feature]) - statistics[position_group][feature]["mean"])
            / statistics[position_group][feature]["std"]
            for feature in FIT_FEATURES
        ],
        dtype=float,
    )


def vector_distance(left: np.ndarray, right: np.ndarray, method: str = "rms") -> float:
    """Compare style vectors without identity, performance, or sample-size inputs."""
    if left.shape != right.shape or left.size != len(FIT_FEATURES):
        raise ValueError("Distances require compatible six-feature vectors")
    delta = left - right
    if method == "rms":
        return float(np.sqrt(np.mean(np.square(delta))))
    if method == "manhattan":
        return float(np.mean(np.abs(delta)))
    if method == "cosine":
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float(1 - np.dot(left, right) / denominator) if denominator else float("nan")
    raise ValueError(f"Unknown distance method: {method}")


def _vector_summary(full: np.ndarray, alternative: np.ndarray) -> dict[str, float | None]:
    denominator = float(np.linalg.norm(full) * np.linalg.norm(alternative))
    cosine_similarity = float(np.dot(full, alternative) / denominator) if denominator else None
    rank = pd.Series(full).corr(pd.Series(alternative), method="spearman")
    return {
        "rms_z_distance": vector_distance(full, alternative, "rms"),
        "cosine_similarity": cosine_similarity,
        "feature_rank_spearman": None if pd.isna(rank) else float(rank),
    }


def _profile_for_subset(
    passes: pd.DataFrame,
    actions: pd.DataFrame,
    shots: pd.DataFrame,
    team_id: int,
    match_ids: Iterable[int] | None = None,
    position_group: str | None = None,
    excluded_player_id: int | None = None,
) -> pd.Series:
    pass_mask = passes["team_id"].eq(team_id)
    action_mask = actions["team_id"].eq(team_id)
    shot_mask = shots["team_id"].eq(team_id)
    if match_ids is not None:
        selected = {int(value) for value in match_ids}
        pass_mask &= passes["match_id"].isin(selected)
        action_mask &= actions["match_id"].isin(selected)
        shot_mask &= shots["match_id"].isin(selected)
    if position_group is not None:
        pass_mask &= passes["position_group"].eq(position_group)
        action_mask &= actions["position_group"].eq(position_group)
        shot_mask &= shots["position_group"].eq(position_group)
    if excluded_player_id is not None:
        pass_mask &= passes["player_id"].ne(excluded_player_id)
        action_mask &= actions["player_id"].ne(excluded_player_id)
        shot_mask &= shots["player_id"].ne(excluded_player_id)
    frame = _aggregate_profiles(
        passes.loc[pass_mask].assign(_group=0),
        actions.loc[action_mask].assign(_group=0),
        ["_group"],
        shots.loc[shot_mask].assign(_group=0),
    )
    if len(frame) != 1:
        raise ValueError("Requested subset did not produce exactly one profile")
    return frame.iloc[0]


def _record_profile(
    profile: pd.Series,
    analysis_type: str,
    sample_id: str,
    position_group: str | None,
    statistics: dict[str, dict[str, dict[str, float]]],
    **extra: Any,
) -> dict[str, Any]:
    record = {"analysis_type": analysis_type, "sample_id": sample_id, **extra}
    record["position_group"] = position_group
    for column in profile.index:
        if column != "_group":
            value = profile[column]
            record[column] = value.item() if hasattr(value, "item") else value
    if position_group in ROLE_GROUPS:
        vector = normalize_profile(profile, position_group, statistics)
        for feature, value in zip(FIT_FEATURES, vector, strict=True):
            record[f"{feature}_z"] = float(value)
    return record


def rank_external_candidates(
    normalized_players: pd.DataFrame,
    raw_players: pd.DataFrame,
    role_raw: pd.Series,
    role_group: str,
    target_team_name: str,
    statistics: dict[str, dict[str, dict[str, float]]],
    method: str = "rms",
) -> pd.DataFrame:
    """Rank same-position eligible external players against one role vector."""
    if role_group not in ROLE_GROUPS:
        raise ValueError("External role fit is outfield-only")
    role_z = normalize_profile(role_raw, role_group, statistics)
    candidates = normalized_players.loc[
        normalized_players["position_group"].eq(role_group)
        & normalized_players["team_name"].ne(target_team_name)
    ].copy()
    raw_lookup = raw_players.set_index("player_id")
    rows: list[dict[str, Any]] = []
    for player in candidates.itertuples(index=False):
        player_z = np.array([float(getattr(player, feature)) for feature in FIT_FEATURES])
        gaps = np.abs(player_z - role_z)
        squared = np.square(gaps)
        total = float(squared.sum())
        order = np.argsort(gaps, kind="stable")
        source = raw_lookup.loc[int(player.player_id)]
        rows.append(
            {
                "analysis_type": "external_fit",
                "distance_method": method,
                "position_group": role_group,
                "player_id": int(player.player_id),
                "player_name": str(player.player_name),
                "team_name": str(player.team_name),
                "matches_observed": int(source["matches_observed"]),
                "pass_attempts": int(source["pass_attempts"]),
                "carries": int(source["carries"]),
                "distance": vector_distance(player_z, role_z, method),
                "closest_feature_1": FIT_FEATURES[int(order[0])],
                "closest_feature_2": FIT_FEATURES[int(order[1])],
                "closest_feature_3": FIT_FEATURES[int(order[2])],
                "largest_gap_feature": FIT_FEATURES[int(order[-1])],
                "feature_gaps": json.dumps(
                    {feature: float(value) for feature, value in zip(FIT_FEATURES, gaps, strict=True)},
                    sort_keys=True,
                ),
                "distance_contributions": json.dumps(
                    {
                        feature: float(value / total) if total else 0.0
                        for feature, value in zip(FIT_FEATURES, squared, strict=True)
                    },
                    sort_keys=True,
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(["distance", "player_id"], kind="stable")
    result["rank"] = np.arange(1, len(result) + 1)
    if result["team_name"].eq(target_team_name).any():
        raise AssertionError("Current target-team players entered external rankings")
    if not result["position_group"].eq(role_group).all():
        raise AssertionError("External fit must remain same-position")
    return result


def _ranking_comparison(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    merged = reference[["player_id", "rank"]].merge(
        candidate[["player_id", "rank"]], on="player_id", suffixes=("_full", "_variant")
    )
    output: dict[str, Any] = {"candidates": len(merged)}
    for size in (5, 6, 10):
        actual = min(size, len(merged))
        left = set(reference.nsmallest(actual, "rank")["player_id"])
        right = set(candidate.nsmallest(actual, "rank")["player_id"])
        union = left | right
        output[f"top_{size}_overlap"] = len(left & right) / actual if actual else None
        output[f"top_{size}_jaccard"] = len(left & right) / len(union) if union else None
    rho = merged["rank_full"].corr(merged["rank_variant"], method="spearman")
    output["rank_spearman"] = None if pd.isna(rho) else float(rho)
    return output


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    series = pd.Series(list(values), dtype=float).dropna()
    if series.empty:
        return {"count": 0, "min": None, "p10": None, "p25": None, "median": None,
                "p75": None, "p90": None, "max": None, "mean": None, "std": None}
    return {
        "count": len(series), "min": float(series.min()),
        "p10": float(series.quantile(0.10)), "p25": float(series.quantile(0.25)),
        "median": float(series.median()), "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)), "max": float(series.max()),
        "mean": float(series.mean()), "std": float(series.std(ddof=0)),
    }


def _feature_variation(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    output: dict[str, Any] = {}
    for feature in (*FIT_FEATURES, *OPTIONAL_STYLE_FEATURES):
        if feature in frame:
            output[feature] = _distribution(frame[feature])
    return output


def _role_contributions(actions: pd.DataFrame, team_id: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = actions.loc[
        actions["team_id"].eq(team_id) & actions["position_group"].isin(ROLE_GROUPS)
    ]
    grouped = source.groupby(
        ["position_group", "player_id", "player_name", "position"], observed=True
    ).agg(actions=("action_id", "size"), matches=("match_id", "nunique"),
          carries=("action_type", lambda x: int(x.eq("Carry").sum())))
    grouped = grouped.reset_index()
    grouped["action_share"] = grouped["actions"] / grouped.groupby("position_group")[
        "actions"
    ].transform("sum")
    summary: dict[str, Any] = {}
    for group, frame in grouped.groupby("position_group", sort=True):
        shares = frame.sort_values("action_share", ascending=False)["action_share"]
        summary[str(group)] = {
            "players": len(frame),
            "top_player_share": float(shares.iloc[0]),
            "top_two_share": float(shares.head(2).sum()),
            "hhi": float(np.square(shares).sum()),
            "dominance_note": (
                "concentrated" if shares.iloc[0] >= 0.40 or shares.head(2).sum() >= 0.65
                else "distributed"
            ),
        }
    return grouped.sort_values(["position_group", "actions"], ascending=[True, False]), summary


def _aggregation_audit(
    full_roles: dict[str, pd.Series],
    target_players: pd.DataFrame,
    statistics: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in ROLE_GROUPS:
        players = target_players.loc[target_players["position_group"].eq(group)]
        if players.empty:
            continue
        vectors = players[list(FIT_FEATURES)].to_numpy(float)
        weights = {
            "equal_player": np.ones(len(players)),
            "action_weighted": players["attacking_actions"].to_numpy(float),
            "match_weighted": players["matches_observed"].to_numpy(float),
        }
        pooled_z = normalize_profile(full_roles[group], group, statistics)
        group_result: dict[str, Any] = {}
        for label, weight in weights.items():
            raw = np.average(vectors, axis=0, weights=weight)
            normalized = np.array(
                [
                    (raw[index] - statistics[group][feature]["mean"])
                    / statistics[group][feature]["std"]
                    for index, feature in enumerate(FIT_FEATURES)
                ]
            )
            group_result[label] = {
                "raw": {feature: float(value) for feature, value in zip(FIT_FEATURES, raw, strict=True)},
                "rms_z_movement_from_pooled": vector_distance(pooled_z, normalized),
            }
        output[group] = group_result
    return output


def _optional_feature_audit(players: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "progressive_pass_rate", "pressure_pass_rate", *OPTIONAL_STYLE_FEATURES
    ]
    correlations = players.loc[players["position_group"].isin(ROLE_GROUPS), columns].corr()
    return {
        "availability": {column: int(players[column].notna().sum()) for column in columns},
        "correlations": {
            "progressive_pass_vs_progressive_action": float(
                correlations.loc["progressive_pass_rate", "progressive_action_rate"]
            ),
            "pressure_pass_vs_pressure_action": float(
                correlations.loc["pressure_pass_rate", "pressure_action_rate"]
            ),
            "progressive_carry_vs_progressive_action": float(
                correlations.loc["progressive_carry_rate", "progressive_action_rate"]
            ),
        },
        "decision": "retain_primary_six",
        "reason": "Optional rates are descriptive; action rates overlap existing pass tendencies.",
    }


def run_research(
    passes_path: Path = DEFAULT_PASSES,
    actions_path: Path = DEFAULT_ACTIONS,
    shots_path: Path = DEFAULT_SHOTS,
    players_path: Path = DEFAULT_PLAYERS,
    chronology_path: Path = DEFAULT_CHRONOLOGY,
    archetypes_path: Path = DEFAULT_ARCHETYPES,
    target_team_name: str = TARGET_TEAM_NAME,
) -> dict[str, Any]:
    """Run the complete bounded V4.0 feasibility audit and persist research outputs."""
    validate_fit_contract()
    passes = pd.read_parquet(passes_path)
    actions_raw = pd.read_parquet(actions_path)
    shots = pd.read_parquet(shots_path)
    players = pd.read_parquet(players_path)
    chronology_raw = pd.read_parquet(chronology_path)
    actions = enrich_actions(actions_raw, players)
    player_lookup = players[["player_id", "position_group"]].drop_duplicates("player_id")
    passes = passes.merge(player_lookup, on="player_id", how="left", validate="many_to_one")
    shots = shots.merge(player_lookup, on="player_id", how="left", validate="many_to_one")
    if passes["position_group"].isna().any() or actions["position_group"].isna().any():
        raise ValueError("Frozen broad-position mapping is incomplete")

    match_ids = set(passes["match_id"].astype(int))
    chronology = build_match_chronology(chronology_raw, match_ids)
    coverage = build_team_coverage(passes, actions, shots)
    target_rows = coverage.loc[coverage["team_name"].eq(target_team_name)]
    if len(target_rows) != 1:
        raise ValueError(f"Could not resolve exactly one target team named {target_team_name!r}")
    target_team_id = int(target_rows.iloc[0]["team_id"])
    target_match_ids = chronology["match_id"].astype(int).tolist()
    if set(passes.loc[passes["team_id"].eq(target_team_id), "match_id"].astype(int)) != set(
        target_match_ids
    ):
        raise AssertionError("Target-team match coverage does not match chronology")

    team_match = build_team_match_profiles(passes, actions, shots, chronology)
    eligible = prepare_similarity_pool(players)
    normalized_players, statistics = normalize_by_position(eligible)
    target_player_metadata = players[
        ["player_id", "player_name", "team_name", "position", "position_group"]
    ].drop_duplicates("player_id")
    target_players = _aggregate_profiles(
        passes.loc[passes["team_id"].eq(target_team_id)],
        actions.loc[actions["team_id"].eq(target_team_id)],
        ["player_id"],
        shots.loc[shots["team_id"].eq(target_team_id)],
    ).merge(
        target_player_metadata.drop(columns="team_name"),
        on="player_id", how="left", validate="one_to_one",
    )
    target_players["team_name"] = target_team_name
    target_players = target_players.loc[target_players["position_group"].isin(ROLE_GROUPS)]
    contributions, concentration = _role_contributions(actions, target_team_id)

    full_team = _profile_for_subset(passes, actions, shots, target_team_id)
    team_records = [
        _record_profile(
            full_team, "full", "full", None, statistics, matches=len(target_match_ids)
        )
    ]
    midpoint = len(target_match_ids) // 2
    halves = {"first": target_match_ids[:midpoint], "second": target_match_ids[midpoint:]}
    half_profiles: dict[str, pd.Series] = {}
    for label, ids in halves.items():
        profile = _profile_for_subset(passes, actions, shots, target_team_id, ids)
        half_profiles[label] = profile
        team_records.append(
            _record_profile(profile, "chronological_half", label, None, statistics, matches=len(ids))
        )

    resamples = deterministic_match_resamples(target_match_ids)
    team_resample_distances: list[float] = []
    for index, ids in enumerate(resamples):
        profile = _profile_for_subset(passes, actions, shots, target_team_id, ids)
        record = _record_profile(
            profile, "match_resample", f"resample_{index:03d}", None, statistics,
            matches=len(ids), match_ids=json.dumps(ids),
        )
        variant = np.array(
            [
                (float(profile[f]) - float(full_team[f]))
                / float(team_match.loc[team_match["team_id"].eq(target_team_id), f].std(ddof=0))
                if team_match.loc[team_match["team_id"].eq(target_team_id), f].std(ddof=0) > 0 else 0
                for f in FIT_FEATURES
            ]
        )
        distance = float(np.sqrt(np.mean(np.square(variant))))
        record["rms_standardized_distance"] = distance
        team_resample_distances.append(distance)
        team_records.append(record)

    team_lomo: list[dict[str, Any]] = []
    for omitted in target_match_ids:
        ids = [value for value in target_match_ids if value != omitted]
        profile = _profile_for_subset(passes, actions, shots, target_team_id, ids)
        record = _record_profile(
            profile, "leave_one_match_out", str(omitted), None, statistics,
            matches=len(ids), omitted_match_id=omitted,
        )
        deltas = []
        for feature in FIT_FEATURES:
            scale = float(team_match.loc[team_match["team_id"].eq(target_team_id), feature].std(ddof=0))
            deltas.append((float(profile[feature]) - float(full_team[feature])) / scale if scale else 0)
        record["rms_standardized_distance"] = float(np.sqrt(np.mean(np.square(deltas))))
        team_lomo.append(record)
        team_records.append(record)

    role_records: list[dict[str, Any]] = []
    full_roles: dict[str, pd.Series] = {}
    role_stability: dict[str, Any] = {}
    fit_frames: list[pd.DataFrame] = []
    target_rank_stability: dict[str, Any] = {}
    role_resamples_by_group: dict[str, list[dict[str, Any]]] = {}
    role_lopo_summary: dict[str, Any] = {}
    current_fit_summary: dict[str, Any] = {}

    for group in ROLE_GROUPS:
        full_role = _profile_for_subset(
            passes, actions, shots, target_team_id, position_group=group
        )
        full_roles[group] = full_role
        full_z = normalize_profile(full_role, group, statistics)
        role_records.append(_record_profile(full_role, "full", "full", group, statistics))
        group_halves: dict[str, Any] = {}
        variant_rankings: list[dict[str, Any]] = []
        reference = rank_external_candidates(
            normalized_players, eligible, full_role, group, target_team_name, statistics, "rms"
        )
        for method in DISTANCE_METHODS:
            fit_frames.append(
                rank_external_candidates(
                    normalized_players, eligible, full_role, group, target_team_name,
                    statistics, method,
                )
            )
        for label, ids in halves.items():
            profile = _profile_for_subset(
                passes, actions, shots, target_team_id, ids, position_group=group
            )
            z = normalize_profile(profile, group, statistics)
            role_records.append(
                _record_profile(profile, "chronological_half", label, group, statistics)
            )
            comparison = _vector_summary(full_z, z)
            comparison["action_count"] = int(profile["attacking_actions"])
            comparison["contributors"] = int(profile["action_players"])
            comparison["raw_features"] = {f: float(profile[f]) for f in FIT_FEATURES}
            comparison["z_features"] = {f: float(v) for f, v in zip(FIT_FEATURES, z, strict=True)}
            group_halves[label] = comparison
            ranking = rank_external_candidates(
                normalized_players, eligible, profile, group, target_team_name, statistics, "rms"
            )
            variant_rankings.append({"variant": f"half_{label}", **_ranking_comparison(reference, ranking)})

        group_resample_records: list[dict[str, Any]] = []
        for index, ids in enumerate(resamples):
            profile = _profile_for_subset(
                passes, actions, shots, target_team_id, ids, position_group=group
            )
            z = normalize_profile(profile, group, statistics)
            record = _record_profile(
                profile, "match_resample", f"resample_{index:03d}", group, statistics,
                match_ids=json.dumps(ids),
            )
            record["rms_z_distance"] = vector_distance(full_z, z)
            group_resample_records.append(record)
            role_records.append(record)
            ranking = rank_external_candidates(
                normalized_players, eligible, profile, group, target_team_name, statistics, "rms"
            )
            variant_rankings.append({"variant": f"resample_{index:03d}", **_ranking_comparison(reference, ranking)})
        role_resamples_by_group[group] = group_resample_records

        match_influence: list[dict[str, Any]] = []
        for omitted in target_match_ids:
            ids = [value for value in target_match_ids if value != omitted]
            profile = _profile_for_subset(
                passes, actions, shots, target_team_id, ids, position_group=group
            )
            z = normalize_profile(profile, group, statistics)
            record = _record_profile(
                profile, "leave_one_match_out", str(omitted), group, statistics,
                omitted_match_id=omitted,
            )
            record["rms_z_distance"] = vector_distance(full_z, z)
            role_records.append(record)
            match_influence.append({"match_id": omitted, "rms_z_distance": record["rms_z_distance"]})
            ranking = rank_external_candidates(
                normalized_players, eligible, profile, group, target_team_name, statistics, "rms"
            )
            variant_rankings.append({"variant": f"lomo_{omitted}", **_ranking_comparison(reference, ranking)})

        lopo: list[dict[str, Any]] = []
        role_players = contributions.loc[contributions["position_group"].eq(group)]
        for player in role_players.itertuples(index=False):
            profile = _profile_for_subset(
                passes, actions, shots, target_team_id, position_group=group,
                excluded_player_id=int(player.player_id),
            )
            z = normalize_profile(profile, group, statistics)
            gaps = np.abs(z - full_z)
            order = np.argsort(gaps, kind="stable")
            record = _record_profile(
                profile, "leave_one_player_out", str(player.player_id), group, statistics,
                excluded_player_id=int(player.player_id), excluded_player_name=str(player.player_name),
                excluded_action_share=float(player.action_share),
            )
            record["rms_z_distance"] = vector_distance(full_z, z)
            record["largest_changed_feature"] = FIT_FEATURES[int(order[-1])]
            role_records.append(record)
            lopo.append(
                {
                    "player_id": int(player.player_id), "player_name": str(player.player_name),
                    "action_share": float(player.action_share),
                    "rms_z_distance": record["rms_z_distance"],
                    "largest_changed_feature": record["largest_changed_feature"],
                    "remaining_actions": int(profile["attacking_actions"]),
                    "remaining_players": int(profile["action_players"]),
                }
            )
            eligible_player = eligible.loc[eligible["player_id"].eq(int(player.player_id))]
            target_player = target_players.loc[
                target_players["player_id"].eq(int(player.player_id))
            ]
            if not eligible_player.empty and not target_player.empty:
                target_player_row = target_player.iloc[0]
                player_z = normalize_profile(target_player_row, group, statistics)
                current_distance = vector_distance(player_z, z)
                fit_frames.append(
                    pd.DataFrame(
                        [{
                            "analysis_type": "current_player_leave_self_out",
                            "distance_method": "rms", "position_group": group,
                            "player_id": int(player.player_id),
                            "player_name": str(player.player_name),
                            "team_name": target_team_name,
                            "matches_observed": int(target_player_row["matches_observed"]),
                            "pass_attempts": int(target_player_row["pass_attempts"]),
                            "carries": int(target_player_row["carries"]),
                            "distance": current_distance, "rank": np.nan,
                            "role_remaining_actions": int(profile["attacking_actions"]),
                            "role_remaining_players": int(profile["action_players"]),
                        }]
                    )
                )
        role_lopo_summary[group] = sorted(lopo, key=lambda x: x["rms_z_distance"], reverse=True)
        current_fit_summary[group] = [
            row for row in role_lopo_summary[group] if row["remaining_players"] >= 1
        ]
        rank_frame = pd.DataFrame(variant_rankings)
        target_rank_stability[group] = {
            metric: _distribution(rank_frame[metric])
            for metric in (
                "top_5_overlap", "top_6_overlap", "top_10_overlap",
                "top_5_jaccard", "top_6_jaccard", "top_10_jaccard", "rank_spearman",
            )
        }
        role_stability[group] = {
            "halves": group_halves,
            "resampling": {
                "rms_z_distance": _distribution(
                    record["rms_z_distance"] for record in group_resample_records
                ),
                "feature_variation": _feature_variation(group_resample_records),
            },
            "leave_one_match_out": {
                "rms_z_distance": _distribution(item["rms_z_distance"] for item in match_influence),
                "most_influential": sorted(
                    match_influence, key=lambda x: x["rms_z_distance"], reverse=True
                )[:5],
            },
        }

    # Hold the target role fixed and vary each eligible external player's chronology.
    # This isolates candidate-profile instability from target-role instability.
    eligible_ids = set(eligible["player_id"].astype(int))
    assignments = assign_player_match_halves(
        passes, actions_raw, chronology, eligible_ids
    )
    half_features = recompute_similarity_features(
        passes, actions_raw, players.loc[players["player_id"].isin(eligible_ids)], assignments
    )
    candidate_profile_stability: dict[str, Any] = {}
    for group in ROLE_GROUPS:
        source = half_features.loc[
            half_features["position_group"].eq(group)
            & half_features["team_name"].ne(target_team_name)
        ].copy()
        complete_ids = source.groupby("player_id")["half"].nunique().loc[lambda x: x.eq(2)].index
        source = source.loc[source["player_id"].isin(complete_ids)]
        role_z = normalize_profile(full_roles[group], group, statistics)
        half_rankings: dict[str, pd.DataFrame] = {}
        for half in ("first", "second"):
            rows: list[dict[str, Any]] = []
            for player in source.loc[source["half"].eq(half)].itertuples(index=False):
                player_z = normalize_profile(player._asdict(), group, statistics)
                rows.append(
                    {
                        "analysis_type": "candidate_profile_half_fit",
                        "sample_id": half,
                        "distance_method": "rms",
                        "position_group": group,
                        "player_id": int(player.player_id),
                        "player_name": str(player.player_name),
                        "team_name": str(player.team_name),
                        "matches_observed": int(player.matches_observed),
                        "pass_attempts": int(player.pass_attempts),
                        "carries": int(player.carries),
                        "distance": vector_distance(player_z, role_z),
                    }
                )
            ranking = pd.DataFrame(rows).sort_values(["distance", "player_id"], kind="stable")
            ranking["rank"] = np.arange(1, len(ranking) + 1)
            half_rankings[half] = ranking
            fit_frames.append(ranking)
        comparison = _ranking_comparison(half_rankings["first"], half_rankings["second"])
        paired = half_rankings["first"][["player_id", "distance"]].merge(
            half_rankings["second"][["player_id", "distance"]],
            on="player_id", suffixes=("_first", "_second"), validate="one_to_one",
        ).merge(
            eligible[["player_id", "matches_observed"]], on="player_id", validate="one_to_one"
        )
        paired["absolute_distance_change"] = (
            paired["distance_first"] - paired["distance_second"]
        ).abs()
        candidate_profile_stability[group] = {
            "players_with_two_halves": len(paired),
            "ranking_comparison": comparison,
            "absolute_distance_change": _distribution(paired["absolute_distance_change"]),
            "by_full_match_support": {
                "2_plus": _distribution(
                    paired.loc[paired["matches_observed"].ge(2), "absolute_distance_change"]
                ),
                "3_plus": _distribution(
                    paired.loc[paired["matches_observed"].ge(3), "absolute_distance_change"]
                ),
            },
        }

    fits = pd.concat(fit_frames, ignore_index=True, sort=False)
    external_rms = fits.loc[
        fits["analysis_type"].eq("external_fit") & fits["distance_method"].eq("rms")
    ]
    fit_distributions = {
        group: _distribution(external_rms.loc[external_rms["position_group"].eq(group), "distance"])
        for group in ROLE_GROUPS
    }
    method_comparison = {}
    for group in ROLE_GROUPS:
        pivot = fits.loc[
            fits["analysis_type"].eq("external_fit") & fits["position_group"].eq(group)
        ].pivot(index="player_id", columns="distance_method", values="rank")
        method_comparison[group] = {
            f"rms_vs_{method}_rank_spearman": float(pivot["rms"].corr(pivot[method], method="spearman"))
            for method in ("manhattan", "cosine")
        }

    archetypes = pd.read_parquet(archetypes_path)
    archetype_audit: dict[str, Any] = {}
    for group in ROLE_GROUPS:
        top = external_rms.loc[external_rms["position_group"].eq(group)].nsmallest(10, "rank")
        joined = top.merge(
            archetypes[["player_id", "archetype_name"]], on="player_id", how="left"
        )
        archetype_audit[group] = {
            str(key): int(value)
            for key, value in joined["archetype_name"].fillna("Unavailable").value_counts().items()
        }

    aggregation = _aggregation_audit(full_roles, target_players, statistics)
    optional_audit = _optional_feature_audit(players)
    team_frame = pd.DataFrame(team_records)
    role_frame = pd.DataFrame(role_records)
    target_matches = team_match.loc[team_match["team_id"].eq(target_team_id)]
    team_half_summary = {
        feature: {
            "first": float(half_profiles["first"][feature]),
            "second": float(half_profiles["second"][feature]),
            "absolute_difference": abs(float(half_profiles["first"][feature]) - float(half_profiles["second"][feature])),
            "standardized_difference": (
                abs(float(half_profiles["first"][feature]) - float(half_profiles["second"][feature]))
                / float(target_matches[feature].std(ddof=0))
                if float(target_matches[feature].std(ddof=0)) > 0 else None
            ),
            "match_variance": float(target_matches[feature].var(ddof=0)),
            "coefficient_of_variation": (
                float(target_matches[feature].std(ddof=0) / abs(target_matches[feature].mean()))
                if target_matches[feature].mean() else None
            ),
        }
        for feature in FIT_FEATURES
    }
    first_standardized = np.array([
        (float(half_profiles["first"][f]) - float(target_matches[f].mean()))
        / float(target_matches[f].std(ddof=0)) if target_matches[f].std(ddof=0) else 0
        for f in FIT_FEATURES
    ])
    second_standardized = np.array([
        (float(half_profiles["second"][f]) - float(target_matches[f].mean()))
        / float(target_matches[f].std(ddof=0)) if target_matches[f].std(ddof=0) else 0
        for f in FIT_FEATURES
    ])

    home_away = {
        venue: {
            feature: float(frame[feature].mean())
            for feature in FIT_FEATURES
        }
        for venue, frame in target_matches.groupby("home_away", sort=True)
    }
    detailed_positions = (
        passes.loc[passes["team_id"].eq(target_team_id)]
        .groupby(["player_id", "player_name"], sort=True)["position"]
        .agg(lambda values: sorted({str(value) for value in values.dropna()}))
        .reset_index(name="detailed_positions")
    )
    detailed_positions["detailed_position_count"] = detailed_positions[
        "detailed_positions"
    ].map(len)
    opponent_counts = coverage.loc[coverage["team_id"].ne(target_team_id), "matches_observed"]
    role_pair_distances = {}
    for left_index, left in enumerate(ROLE_GROUPS):
        for right in ROLE_GROUPS[left_index + 1:]:
            # Each role is position-relative, but the shared axes make this a useful diagnostic.
            role_pair_distances[f"{left}_vs_{right}"] = vector_distance(
                normalize_profile(full_roles[left], left, statistics),
                normalize_profile(full_roles[right], right, statistics),
            )

    # Candidate profile uncertainty is inherited from the frozen V3.3A/B grouped audit.
    similarity_metadata = json.loads(
        (MODELS / "player_similarity_metadata.json").read_text(encoding="utf-8")
    )
    support = similarity_metadata["sample_support_calibration"]
    decisions = {
        "V4.1": "GO WITH LIMITATIONS",
        "V4.2": {group: "PENDING_EVIDENCE_RULE" for group in ROLE_GROUPS},
        "V4.3": {group: "PENDING_EVIDENCE_RULE" for group in ROLE_GROUPS},
        "V4.4": "PENDING_EVIDENCE_RULE",
    }
    # Conservative, declared diagnostics: role halves, resamples, ranking stability, contributors.
    for group in ROLE_GROUPS:
        half_distance = vector_distance(
            normalize_profile(
                role_frame.loc[
                    role_frame["position_group"].eq(group)
                    & role_frame["analysis_type"].eq("chronological_half")
                    & role_frame["sample_id"].eq("first")
                ].iloc[0], group, statistics,
            ),
            normalize_profile(
                role_frame.loc[
                    role_frame["position_group"].eq(group)
                    & role_frame["analysis_type"].eq("chronological_half")
                    & role_frame["sample_id"].eq("second")
                ].iloc[0], group, statistics,
            ),
        )
        median_resample = role_stability[group]["resampling"]["rms_z_distance"]["median"]
        median_top6 = target_rank_stability[group]["top_6_overlap"]["median"]
        candidate_rank_spearman = candidate_profile_stability[group][
            "ranking_comparison"
        ]["rank_spearman"]
        candidate_top6 = candidate_profile_stability[group]["ranking_comparison"][
            "top_6_overlap"
        ]
        enough_players = concentration[group]["players"] >= 3
        concentrated = concentration[group]["dominance_note"] == "concentrated"
        stable_role = half_distance <= 1.0 and median_resample <= 0.35
        stable_rank = median_top6 is not None and median_top6 >= 0.67
        stable_candidates = (
            candidate_rank_spearman is not None
            and candidate_rank_spearman >= 0.70
            and candidate_top6 is not None
            and candidate_top6 >= 0.67
        )
        decisions["V4.2"][group] = (
            "GO" if stable_role and enough_players and not concentrated
            else "GO WITH LIMITATIONS"
            if median_resample <= 0.50 and enough_players else "NO-GO"
        )
        decisions["V4.3"][group] = (
            "GO WITH LIMITATIONS"
            if stable_role
            and stable_rank
            and stable_candidates
            and enough_players
            and not concentrated
            else "NO-GO"
        )
    role_stage_decisions = [decisions["V4.2"][group] for group in ROLE_GROUPS]
    fit_stage_decisions = [decisions["V4.3"][group] for group in ROLE_GROUPS]
    decisions["V4.2"]["overall"] = (
        "GO" if all(value == "GO" for value in role_stage_decisions)
        else "GO WITH LIMITATIONS"
        if any(value.startswith("GO") for value in role_stage_decisions)
        else "NO-GO"
    )
    decisions["V4.3"]["overall"] = (
        "GO WITH LIMITATIONS"
        if any(value.startswith("GO") for value in fit_stage_decisions)
        else "NO-GO"
    )
    if any(value.startswith("GO") for value in fit_stage_decisions):
        decisions["V4.4"] = "GO WITH SAMPLE-SUPPORT GUARDRAILS"
    else:
        decisions["V4.4"] = "NO-GO"

    metadata: dict[str, Any] = {
        "methodology_version": METHODOLOGY_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Research only; no production persistence, API, frontend, or model changes.",
        "source_product_cohort": "Bundesliga 2023/24, 34 Leverkusen-centered fixtures",
        "target": {"team_id": target_team_id, "team_name": target_team_name},
        "coverage": {
            "target_matches": len(target_match_ids),
            "chronology_start": str(chronology.iloc[0]["match_date"]),
            "chronology_end": str(chronology.iloc[-1]["match_date"]),
            "chronological_halves": [len(halves["first"]), len(halves["second"])],
            "opponent_match_distribution": _distribution(opponent_counts),
            "team_table": coverage.to_dict(orient="records"),
            "opponent_language": "Observed only in matches against Bayer Leverkusen.",
        },
        "feature_contract": {
            "descriptive_team_metrics": list(DESCRIPTIVE_FEATURES),
            "primary_role_fit_features": list(FIT_FEATURES),
            "feature_definitions": {
                "long_pass_rate": f"pass_length >= {LONG_PASS_LENGTH:g}",
                "positive_forward_distance_per_100_passes": "sum(max(forward_distance, 0)) / passes * 100",
                "final_third_entries_per_100_passes": f"start_x < {FINAL_THIRD_X:g} and end_x >= {FINAL_THIRD_X:g}",
                "carry_share_of_actions": "carries / (passes + carries)",
            },
            "optional_style_audit": optional_audit,
            "excluded_from_fit": sorted(
                set(PERFORMANCE_FEATURES) | IDENTITY_COLUMNS | {"archetype", "outcomes"}
            ),
        },
        "normalization": {
            "method": "V3.3B eligible-player population z-scores by broad position",
            "statistics": statistics,
            "compatible_player_role_space": True,
        },
        "resampling": {
            "unit": "match", "count": RESAMPLE_COUNT,
            "fraction": RESAMPLE_FRACTION, "seed": RANDOM_SEED,
            "matches_per_resample": len(resamples[0]),
        },
        "role_contributions": concentration,
        "role_contribution_rows": contributions.to_dict(orient="records"),
        "detailed_position_audit": {
            "players": detailed_positions.to_dict(orient="records"),
            "players_in_multiple_detailed_positions": detailed_positions.loc[
                detailed_positions["detailed_position_count"].gt(1)
            ].to_dict(orient="records"),
            "broad_position_policy": "Frozen V3 primary event-count mode with deterministic tie-breaking.",
        },
        "team_stability": {
            "split_half_features": team_half_summary,
            "split_half_vector": _vector_summary(first_standardized, second_standardized),
            "resampling": {
                "rms_standardized_distance": _distribution(team_resample_distances),
                "feature_variation": _feature_variation(
                    [row for row in team_records if row["analysis_type"] == "match_resample"]
                ),
            },
            "leave_one_match_out": {
                "rms_standardized_distance": _distribution(
                    row["rms_standardized_distance"] for row in team_lomo
                ),
                "most_influential": sorted(
                    [
                        {"match_id": int(row["omitted_match_id"]),
                         "rms_standardized_distance": row["rms_standardized_distance"]}
                        for row in team_lomo
                    ],
                    key=lambda x: x["rms_standardized_distance"], reverse=True,
                )[:5],
            },
            "home_away_mean_features": home_away,
            "team_match_context": "68 team-match observations; not Bundesliga season percentiles.",
        },
        "role_stability": role_stability,
        "role_aggregation_methods": aggregation,
        "leave_one_player_out_role_sensitivity": role_lopo_summary,
        "current_player_leave_self_out": "Rows are stored in player_role_fit_research_v40.parquet.",
        "external_fit": {
            "rules": "V3.3B eligible, same position, outfield, excludes current Leverkusen players",
            "methods": list(DISTANCE_METHODS),
            "primary": "rms",
            "method_rank_comparison": method_comparison,
            "rms_distance_distributions": fit_distributions,
            "mapping_recommendation": "Retain raw RMS internally; postpone a display score until more candidate match coverage exists.",
            "explanation": "Smallest absolute position-z gaps are alignments; largest gaps are differences; squared-gap shares explain RMS distance.",
        },
        "target_role_ranking_stability": target_rank_stability,
        "candidate_player_profile_stability": {
            "role_fit_split_half": candidate_profile_stability,
            "v33b_grouped_similarity_calibration": support,
            "end_to_end_decision": "Not pooled into one score; target-role and candidate-profile uncertainty remain separate.",
        },
        "sample_support_recommendation": {
            "decision": "show observed match count and retain V3.3B limited/higher labels only as descriptive evidence",
            "limited": "<3 observed candidate matches",
            "higher": ">=3 observed candidate matches",
            "fit_penalty": False,
            "reason": "V3.3B grouped evidence improves at 3 matches, but only 20 eligible players clear it.",
        },
        "archetype_post_hoc_audit": archetype_audit,
        "team_role_distinction": {"pairwise_role_rms_z_distances": role_pair_distances},
        "go_no_go": decisions,
        "generalization": "All builders accept team_id and position_group; complete league data could run unchanged for any team.",
        "limitations": [
            "Only Bayer Leverkusen has full-season coverage; opponents have two matches each.",
            "FWD has only three contributing Leverkusen players and eleven eligible external candidates.",
            "Most external candidates have fewer than three observed matches.",
            "Primary position is the frozen event-count mode, not match-specific tactical role.",
            "No authoritative minutes are available, so no minute weighting is used.",
            "Score-state adjustment was not attempted because a governed score-state field is absent.",
            "Expected-completion, xG, and action-value inputs remain frozen OOF estimates.",
        ],
        "artifacts": {
            "team_match_profiles": repository_relative_path(DEFAULT_TEAM_MATCH_OUTPUT),
            "team_research": repository_relative_path(DEFAULT_TEAM_RESEARCH_OUTPUT),
            "role_profiles": repository_relative_path(DEFAULT_ROLE_OUTPUT),
            "player_role_fit": repository_relative_path(DEFAULT_FIT_OUTPUT),
            "coverage": repository_relative_path(DEFAULT_COVERAGE_OUTPUT),
            "metadata": repository_relative_path(DEFAULT_METADATA_OUTPUT),
        },
    }

    _atomic_parquet(team_match, DEFAULT_TEAM_MATCH_OUTPUT)
    _atomic_parquet(team_frame, DEFAULT_TEAM_RESEARCH_OUTPUT)
    _atomic_parquet(role_frame, DEFAULT_ROLE_OUTPUT)
    _atomic_parquet(fits, DEFAULT_FIT_OUTPUT)
    _atomic_parquet(coverage, DEFAULT_COVERAGE_OUTPUT)
    _atomic_json(metadata, DEFAULT_METADATA_OUTPUT)
    print("V4.0 team/role feasibility research complete")
    print(f"Target: {target_team_name} ({target_team_id})")
    print(f"Target matches: {len(target_match_ids):,}")
    print(f"Team-match profiles: {len(team_match):,}")
    print(f"Role research rows: {len(role_frame):,}")
    print(f"Fit research rows: {len(fits):,}")
    print(f"Metadata: {DEFAULT_METADATA_OUTPUT}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=Path, default=DEFAULT_PASSES)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--shots", type=Path, default=DEFAULT_SHOTS)
    parser.add_argument("--players", type=Path, default=DEFAULT_PLAYERS)
    parser.add_argument("--chronology", type=Path, default=DEFAULT_CHRONOLOGY)
    parser.add_argument("--archetypes", type=Path, default=DEFAULT_ARCHETYPES)
    parser.add_argument("--target-team-name", default=TARGET_TEAM_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_research(
        args.passes, args.actions, args.shots, args.players, args.chronology,
        args.archetypes, args.target_team_name,
    )


if __name__ == "__main__":
    main()
