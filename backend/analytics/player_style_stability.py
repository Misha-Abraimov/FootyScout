"""Audit V3 style-feature stability and clustering cohort coverage.

This is an analysis-only V3.2A pipeline. It does not alter the frozen V3.1
percentile contract, run dimensionality reduction, or fit a clustering model.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analytics.artifact_paths import repository_relative_path
from analytics.player_feature_registry import (
    PERFORMANCE_FEATURES,
    STYLE_CLUSTERING_FEATURES,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_PASSES = PROCESSED / "pass_oof_predictions.parquet"
DEFAULT_ACTIONS = PROCESSED / "attacking_actions.parquet"
DEFAULT_PROFILES = PROCESSED / "player_intelligence_profiles.parquet"
DEFAULT_MATCHES = PROCESSED / "product_match_chronology.parquet"
DEFAULT_EXPANDED = PROCESSED / "player_style_features_expanded.parquet"
DEFAULT_NORMALIZED = PROCESSED / "player_style_features_position_normalized.parquet"
DEFAULT_AUDIT = ROOT / "models" / "player_style_stability_audit.json"

PASS_THRESHOLDS = (25, 50, 75, 100)
CARRY_THRESHOLDS = (15, 20, 29)
OUTFIELD_GROUPS = ("DEF", "MID", "FWD")
POSITION_GROUPS = ("GK", *OUTFIELD_GROUPS)
LONG_PASS_LENGTH = 30.0
FINAL_THIRD_X = 80.0
PASS_STYLE_FEATURES = tuple(
    feature for feature in STYLE_CLUSTERING_FEATURES if feature != "carry_share_of_actions"
)

MATCH_COLUMNS = ["match_id", "match_date", "kick_off", "home_team", "away_team"]
EXPANDED_METADATA = ["player_id", "position_group"]
COUNT_COLUMNS = ["pass_attempts", "actions", "carries", "matches_observed"]


def build_match_chronology(matches: pd.DataFrame, match_ids: Iterable[int]) -> pd.DataFrame:
    """Return a complete, uniquely dated chronological match lookup."""
    missing = set(MATCH_COLUMNS).difference(matches.columns)
    if missing:
        raise ValueError(f"Match catalogue is missing columns: {sorted(missing)}")
    expected = {int(match_id) for match_id in match_ids}
    result = matches.loc[matches["match_id"].isin(expected), MATCH_COLUMNS].copy()
    result["match_id"] = pd.to_numeric(result["match_id"], errors="raise").astype(int)
    result["match_date"] = pd.to_datetime(result["match_date"], errors="raise")
    result["kick_off"] = result["kick_off"].astype("string").fillna("")
    if result["match_id"].duplicated().any():
        raise ValueError("Match chronology must contain one row per match")
    found = set(result["match_id"])
    if found != expected:
        raise ValueError(f"Match chronology is incomplete; missing IDs: {sorted(expected - found)}")
    return result.sort_values(
        ["match_date", "kick_off", "match_id"], kind="stable"
    ).reset_index(drop=True)


def fetch_product_match_chronology(match_ids: Iterable[int]) -> pd.DataFrame:
    """Resolve Bundesliga 2023/24 dates from the StatsBomb Open Data catalogue."""
    from statsbombpy import sb

    matches = sb.matches(competition_id=9, season_id=281, fmt="dataframe")
    return build_match_chronology(matches, match_ids)


def assign_player_match_halves(
    passes: pd.DataFrame,
    actions: pd.DataFrame,
    chronology: pd.DataFrame,
    player_ids: set[int],
) -> pd.DataFrame:
    """Split each player's observed matches chronologically into disjoint halves.

    With an odd number of matches, the later half receives the extra match. A
    one-match player consequently has no first-half sample and is excluded from
    split-half correlations without being deleted from full-cohort analysis.
    """
    pass_pairs = passes.loc[
        passes["player_id"].isin(player_ids), ["player_id", "match_id"]
    ]
    action_pairs = actions.loc[
        actions["player_id"].isin(player_ids), ["player_id", "match_id"]
    ]
    pairs = pd.concat([pass_pairs, action_pairs], ignore_index=True).drop_duplicates()
    pairs["player_id"] = pd.to_numeric(pairs["player_id"], errors="raise").astype(int)
    pairs["match_id"] = pd.to_numeric(pairs["match_id"], errors="raise").astype(int)
    pairs = pairs.merge(
        chronology[["match_id", "match_date", "kick_off"]],
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    if pairs["match_date"].isna().any():
        raise ValueError("Every observed player-match must have a chronological date")
    pairs = pairs.sort_values(
        ["player_id", "match_date", "kick_off", "match_id"], kind="stable"
    )
    pairs["match_number"] = pairs.groupby("player_id").cumcount()
    pairs["match_count"] = pairs.groupby("player_id")["match_id"].transform("size")
    pairs["half"] = np.where(
        pairs["match_number"] < (pairs["match_count"] // 2), "first", "second"
    )
    if pairs.duplicated(["player_id", "match_id"]).any():
        raise AssertionError("Each player-match must be assigned to exactly one half")
    return pairs[["player_id", "match_id", "half", "match_number", "match_count"]]


def _aggregate_pass_style(passes: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    source = passes.copy()
    source["long_pass"] = pd.to_numeric(source["pass_length"], errors="coerce").ge(
        LONG_PASS_LENGTH
    )
    source["positive_forward_distance"] = pd.to_numeric(
        source["forward_distance"], errors="coerce"
    ).clip(lower=0)
    source["final_third_entry"] = (
        pd.to_numeric(source["start_x"], errors="coerce").lt(FINAL_THIRD_X)
        & pd.to_numeric(source["end_x"], errors="coerce").ge(FINAL_THIRD_X)
    )
    grouped = source.groupby(keys, sort=True, observed=True)
    result = grouped.agg(
        pass_attempts=("match_id", "size"),
        matches_observed=("match_id", "nunique"),
        expected_completion_rate=("expected_completion", "mean"),
        pressure_pass_rate=("under_pressure", "mean"),
        progressive_pass_rate=("progressive", "mean"),
        long_pass_rate=("long_pass", "mean"),
        positive_forward_distance=("positive_forward_distance", "sum"),
        final_third_entries=("final_third_entry", "sum"),
    ).reset_index()
    result["positive_forward_distance_per_100_passes"] = (
        100.0 * result["positive_forward_distance"] / result["pass_attempts"]
    )
    result["final_third_entries_per_100_passes"] = (
        100.0 * result["final_third_entries"] / result["pass_attempts"]
    )
    return result.drop(columns=["positive_forward_distance", "final_third_entries"])


def _aggregate_action_style(actions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    source = actions.copy()
    source["is_carry"] = source["action_type"].eq("Carry")
    result = source.groupby(keys, sort=True, observed=True).agg(
        actions=("action_type", "size"),
        carries=("is_carry", "sum"),
    ).reset_index()
    result["carry_share_of_actions"] = result["carries"] / result["actions"].replace(0, np.nan)
    return result


def recompute_style_features(
    passes: pd.DataFrame,
    actions: pd.DataFrame,
    profiles: pd.DataFrame,
    half_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Recompute the seven frozen candidate style features from event-level rows."""
    required_passes = {
        "player_id", "match_id", "expected_completion", "under_pressure", "progressive",
        "pass_length", "forward_distance", "start_x", "end_x",
    }
    required_actions = {"player_id", "match_id", "action_type"}
    required_profiles = {"player_id", "position_group"}
    for frame, required, label in (
        (passes, required_passes, "passes"),
        (actions, required_actions, "actions"),
        (profiles, required_profiles, "profiles"),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
    if profiles["player_id"].duplicated().any():
        raise ValueError("Profiles must contain one row per player")

    cohort = profiles[["player_id", "position_group"]].copy()
    cohort["player_id"] = cohort["player_id"].astype(int)
    player_ids = set(cohort["player_id"])
    pass_source = passes.loc[passes["player_id"].isin(player_ids)].copy()
    action_source = actions.loc[actions["player_id"].isin(player_ids)].copy()
    keys = ["player_id"]
    if half_assignments is not None:
        lookup = half_assignments[["player_id", "match_id", "half"]]
        pass_source = pass_source.merge(
            lookup, on=["player_id", "match_id"], how="left", validate="many_to_one"
        )
        action_source = action_source.merge(
            lookup, on=["player_id", "match_id"], how="left", validate="many_to_one"
        )
        if pass_source["half"].isna().any() or action_source["half"].isna().any():
            raise ValueError("Every source event must map to a player-specific half")
        keys.append("half")

    pass_features = _aggregate_pass_style(pass_source, keys)
    action_features = _aggregate_action_style(action_source, keys)
    result = pass_features.merge(action_features, on=keys, how="left", validate="one_to_one")
    result["actions"] = result["actions"].fillna(0).astype(int)
    result["carries"] = result["carries"].fillna(0).astype(int)
    result = result.merge(cohort, on="player_id", how="left", validate="many_to_one")
    columns = ["player_id", "position_group"]
    if half_assignments is not None:
        columns.append("half")
    columns += [*COUNT_COLUMNS, *STYLE_CLUSTERING_FEATURES]
    # matches_observed comes from passes; every product-cohort player has a pass.
    sort_columns = ["player_id"]
    if half_assignments is not None:
        sort_columns.append("half")
    result = result[columns].sort_values(sort_columns, kind="stable").reset_index(drop=True)
    forbidden = set(PERFORMANCE_FEATURES).intersection(result.columns)
    if forbidden:
        raise AssertionError(f"Performance features entered style analysis: {sorted(forbidden)}")
    return result


def validate_against_frozen(recomputed: pd.DataFrame, frozen: pd.DataFrame) -> None:
    """Prove that analysis recomputation matches the frozen V3.1 raw metrics."""
    expected = frozen[["player_id", *STYLE_CLUSTERING_FEATURES]].copy()
    merged = recomputed.merge(expected, on="player_id", suffixes=("", "__frozen"))
    if len(merged) != len(frozen) or merged["player_id"].duplicated().any():
        raise AssertionError("Recomputed style cohort does not match frozen V3.1 players")
    for feature in STYLE_CLUSTERING_FEATURES:
        left = pd.to_numeric(merged[feature], errors="coerce")
        right = pd.to_numeric(merged[f"{feature}__frozen"], errors="coerce")
        # The frozen profile sums float32 OOF probabilities before division,
        # while groupby mean may accumulate in a different order.
        if not np.allclose(left, right, rtol=1e-7, atol=2e-7, equal_nan=True):
            raise AssertionError(f"Recomputed feature differs from frozen V3.1: {feature}")


def eligibility_mask(frame: pd.DataFrame, pass_threshold: int, carry_threshold: int) -> pd.Series:
    """Return clustering-only complete-case eligibility without imputation."""
    feature_complete = frame[STYLE_CLUSTERING_FEATURES].notna().all(axis=1)
    return (
        feature_complete
        & frame["pass_attempts"].ge(pass_threshold)
        & frame["carries"].ge(carry_threshold)
    )


def _safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float | None:
    if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
        return None
    value = left.corr(right, method=method)
    return None if pd.isna(value) else float(value)


def feature_stability(
    full: pd.DataFrame,
    halves: pd.DataFrame,
    pass_threshold: int,
    carry_threshold: int,
    *,
    require_complete_cohort: bool,
) -> dict[str, dict[str, float | int | None]]:
    """Calculate split-half stability under one clustering-only threshold pair."""
    pivot = halves.pivot(index="player_id", columns="half")
    complete_ids = set(full.loc[eligibility_mask(full, pass_threshold, carry_threshold), "player_id"])
    records: dict[str, dict[str, float | int | None]] = {}
    for feature in STYLE_CLUSTERING_FEATURES:
        if require_complete_cohort:
            eligible_ids = complete_ids
        elif feature in PASS_STYLE_FEATURES:
            eligible_ids = set(full.loc[
                full["pass_attempts"].ge(pass_threshold) & full[feature].notna(), "player_id"
            ])
        else:
            eligible_ids = set(full.loc[
                full["carries"].ge(carry_threshold) & full[feature].notna(), "player_id"
            ])
        available = pivot.index.intersection(sorted(eligible_ids))
        first = pd.to_numeric(pivot.loc[available, (feature, "first")], errors="coerce")
        second = pd.to_numeric(pivot.loc[available, (feature, "second")], errors="coerce")
        valid = first.notna() & second.notna()
        first = first.loc[valid]
        second = second.loc[valid]
        records[feature] = {
            "eligible_players": len(first),
            "pearson": _safe_correlation(first, second, "pearson"),
            "spearman": _safe_correlation(first, second, "spearman"),
            "median_absolute_difference": (
                float((first - second).abs().median()) if len(first) else None
            ),
        }
    return records


def cohort_sizes(
    frame: pd.DataFrame, pass_threshold: int, carry_threshold: int
) -> dict[str, int]:
    eligible = frame.loc[eligibility_mask(frame, pass_threshold, carry_threshold)]
    counts = eligible["position_group"].value_counts()
    result = {group: int(counts.get(group, 0)) for group in POSITION_GROUPS}
    result["ALL_OUTFIELD"] = int(eligible["position_group"].isin(OUTFIELD_GROUPS).sum())
    return result


def feature_position_eligibility(
    frame: pd.DataFrame, pass_threshold: int, carry_threshold: int
) -> dict[str, dict[str, dict[str, int]]]:
    output: dict[str, dict[str, dict[str, int]]] = {}
    for feature in STYLE_CLUSTERING_FEATURES:
        threshold_mask = (
            frame["pass_attempts"].ge(pass_threshold)
            if feature in PASS_STYLE_FEATURES
            else frame["carries"].ge(carry_threshold)
        )
        available = frame[feature].notna() & threshold_mask
        output[feature] = {}
        for group in POSITION_GROUPS:
            group_mask = frame["position_group"].eq(group)
            output[feature][group] = {
                "players": int(group_mask.sum()),
                "eligible": int((group_mask & available).sum()),
                "metric_missing": int((group_mask & frame[feature].isna()).sum()),
                "below_sample_threshold": int(
                    (group_mask & frame[feature].notna() & ~threshold_mask).sum()
                ),
            }
    return output


def _eta_squared(values: pd.Series, groups: pd.Series) -> float | None:
    valid = values.notna() & groups.notna()
    values = values.loc[valid].astype(float)
    groups = groups.loc[valid]
    if len(values) < 2:
        return None
    total = float(((values - values.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    between = sum(
        len(group) * float(group.mean() - values.mean()) ** 2
        for _, group in values.groupby(groups)
    )
    return float(between / total)


def position_differences(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize broad-position location differences and one-way eta squared."""
    output: dict[str, Any] = {}
    for feature in STYLE_CLUSTERING_FEATURES:
        feature_result: dict[str, Any] = {}
        for group in OUTFIELD_GROUPS:
            values = frame.loc[frame["position_group"].eq(group), feature].dropna()
            feature_result[group] = {
                "count": len(values),
                "mean": float(values.mean()) if len(values) else None,
                "median": float(values.median()) if len(values) else None,
            }
        feature_result["eta_squared"] = _eta_squared(
            frame[feature], frame["position_group"]
        )
        output[feature] = feature_result
    return output


def position_prediction_diagnostic(
    frame: pd.DataFrame, feature_columns: list[str] | None = None
) -> dict[str, float | int | None]:
    """Cross-validate a disposable broad-position classifier on style features."""
    features = feature_columns or STYLE_CLUSTERING_FEATURES
    source = frame.loc[frame["position_group"].isin(OUTFIELD_GROUPS)].dropna(
        subset=features
    )
    counts = source["position_group"].value_counts()
    if source.empty or len(counts) < 2 or int(counts.min()) < 2:
        return {"players": len(source), "folds": None, "accuracy": None,
                "balanced_accuracy": None, "macro_f1": None, "majority_baseline": None}
    folds = min(5, int(counts.min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2_000, random_state=42)
    )
    labels = source["position_group"].astype(str)
    predictions = cross_val_predict(model, source[features], labels, cv=cv)
    return {
        "players": len(source),
        "folds": folds,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "majority_baseline": float(counts.max() / len(source)),
    }


def redundancy_pairs(frame: pd.DataFrame, threshold: float = 0.85) -> list[dict[str, Any]]:
    correlations = frame[STYLE_CLUSTERING_FEATURES].corr(method="pearson")
    pairs = []
    for index, left in enumerate(STYLE_CLUSTERING_FEATURES):
        for right in STYLE_CLUSTERING_FEATURES[index + 1:]:
            value = correlations.at[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                pairs.append({"left": left, "right": right, "pearson": float(value)})
    return pairs


def position_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Z-score features within broad position, without filling missing values."""
    result = frame[["player_id", "position_group"]].copy()
    for feature in STYLE_CLUSTERING_FEATURES:
        grouped = frame.groupby("position_group")[feature]
        means = grouped.transform("mean")
        standard_deviations = grouped.transform(lambda values: values.std(ddof=0))
        result[f"{feature}_position_z"] = (frame[feature] - means) / standard_deviations.replace(
            0, np.nan
        )
    return result


def build_expanded_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[[*EXPANDED_METADATA, *COUNT_COLUMNS, *STYLE_CLUSTERING_FEATURES]].copy()
    for pass_threshold in PASS_THRESHOLDS:
        result[f"pass_features__eligible_{pass_threshold}"] = (
            result[list(PASS_STYLE_FEATURES)].notna().all(axis=1)
            & result["pass_attempts"].ge(pass_threshold)
        )
    for carry_threshold in CARRY_THRESHOLDS:
        result[f"carry_share_of_actions__eligible_{carry_threshold}"] = (
            result["carry_share_of_actions"].notna()
            & result["carries"].ge(carry_threshold)
        )
    for pass_threshold in PASS_THRESHOLDS:
        for carry_threshold in CARRY_THRESHOLDS:
            label = f"pass_{pass_threshold}__carry_{carry_threshold}"
            result[f"complete__{label}"] = eligibility_mask(
                result, pass_threshold, carry_threshold
            )
            reasons = []
            for row in result.itertuples(index=False):
                row_reasons = []
                if row.pass_attempts < pass_threshold:
                    row_reasons.append(f"six_pass_features:passes<{pass_threshold}")
                missing_pass = [
                    feature for feature in PASS_STYLE_FEATURES if pd.isna(getattr(row, feature))
                ]
                if missing_pass:
                    row_reasons.append("missing:" + ",".join(missing_pass))
                if row.carries < carry_threshold:
                    row_reasons.append(f"carry_share_of_actions:carries<{carry_threshold}")
                if pd.isna(row.carry_share_of_actions):
                    row_reasons.append("missing:carry_share_of_actions")
                reasons.append("eligible" if not row_reasons else ";".join(row_reasons))
            result[f"ineligibility__{label}"] = reasons
    return result.sort_values("player_id", kind="stable").reset_index(drop=True)


def build_audit(
    full: pd.DataFrame,
    halves: pd.DataFrame,
) -> dict[str, Any]:
    threshold_results = []
    for pass_threshold in PASS_THRESHOLDS:
        for carry_threshold in CARRY_THRESHOLDS:
            complete = full.loc[eligibility_mask(full, pass_threshold, carry_threshold)]
            outfield = complete.loc[complete["position_group"].isin(OUTFIELD_GROUPS)]
            complete_stability = feature_stability(
                full, halves, pass_threshold, carry_threshold, require_complete_cohort=True
            )
            correlations = [
                value["spearman"] for value in complete_stability.values()
                if value["spearman"] is not None
            ]
            threshold_results.append({
                "pass_threshold": pass_threshold,
                "carry_threshold": carry_threshold,
                "cohort_sizes": cohort_sizes(full, pass_threshold, carry_threshold),
                "feature_specific_stability": feature_stability(
                    full, halves, pass_threshold, carry_threshold,
                    require_complete_cohort=False,
                ),
                "complete_cohort_stability": complete_stability,
                "median_complete_cohort_spearman": (
                    float(np.median(correlations)) if correlations else None
                ),
                "feature_position_eligibility": feature_position_eligibility(
                    full, pass_threshold, carry_threshold
                ),
                "position_differences": position_differences(outfield),
                "position_prediction": position_prediction_diagnostic(outfield),
                "redundancy_pairs": redundancy_pairs(outfield),
            })
    return {
        "version": "V3.2A",
        "scope": "Analysis only; no PCA or clustering was run.",
        "style_features": STYLE_CLUSTERING_FEATURES,
        "pass_thresholds": list(PASS_THRESHOLDS),
        "carry_thresholds": list(CARRY_THRESHOLDS),
        "threshold_basis": {
            "passing_features": "full-sample pass_attempts",
            "carry_share_of_actions": "full-sample carries",
            "split_half": "first floor(n/2) and last ceil(n/2) observed matches by date",
        },
        "players": len(full),
        "players_with_two_observed_matches": int(
            full["player_id"].isin(
                halves.groupby("player_id")["half"].nunique().loc[lambda x: x.eq(2)].index
            ).sum()
        ),
        "threshold_results": threshold_results,
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def run_pipeline(
    passes_path: Path = DEFAULT_PASSES,
    actions_path: Path = DEFAULT_ACTIONS,
    profiles_path: Path = DEFAULT_PROFILES,
    matches_path: Path = DEFAULT_MATCHES,
    expanded_output: Path = DEFAULT_EXPANDED,
    audit_output: Path = DEFAULT_AUDIT,
    normalized_output: Path | None = None,
    normalized_pass_threshold: int | None = None,
    normalized_carry_threshold: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    passes = pd.read_parquet(passes_path)
    actions = pd.read_parquet(actions_path)
    profiles = pd.read_parquet(profiles_path)
    match_ids = set(passes["match_id"].dropna().astype(int))
    if matches_path.exists():
        chronology = build_match_chronology(pd.read_parquet(matches_path), match_ids)
    else:
        chronology = fetch_product_match_chronology(match_ids)
        _atomic_parquet(chronology, matches_path)

    player_ids = set(profiles["player_id"].astype(int))
    assignments = assign_player_match_halves(passes, actions, chronology, player_ids)
    full = recompute_style_features(passes, actions, profiles)
    halves = recompute_style_features(passes, actions, profiles, assignments)
    frozen = pd.read_parquet(PROCESSED / "player_style_features.parquet")
    validate_against_frozen(full, frozen)
    expanded = build_expanded_matrix(full)
    audit = build_audit(full, halves)
    _atomic_parquet(expanded, expanded_output)
    normalized = None
    if normalized_output is not None:
        if normalized_pass_threshold not in PASS_THRESHOLDS:
            raise ValueError("Normalized pass threshold must be one of the audited candidates")
        if normalized_carry_threshold not in CARRY_THRESHOLDS:
            raise ValueError("Normalized carry threshold must be one of the audited candidates")
        mask = eligibility_mask(full, normalized_pass_threshold, normalized_carry_threshold)
        cohort = full.loc[mask & full["position_group"].isin(OUTFIELD_GROUPS)].copy()
        normalized = position_normalize(cohort)
        _atomic_parquet(normalized, normalized_output)
        normalized_columns = [
            f"{feature}_position_z" for feature in STYLE_CLUSTERING_FEATURES
        ]
        diagnostic_frame = normalized.rename(columns={
            f"{feature}_position_z": feature for feature in STYLE_CLUSTERING_FEATURES
        })
        audit["position_normalized_candidate"] = {
            "pass_threshold": normalized_pass_threshold,
            "carry_threshold": normalized_carry_threshold,
            "players": len(normalized),
            "position_groups": {
                str(group): int(count)
                for group, count in normalized["position_group"].value_counts().items()
            },
            "fit_scope": "Eligible combined outfield analysis cohort only.",
            "position_prediction": position_prediction_diagnostic(
                normalized, normalized_columns
            ),
            "position_differences": position_differences(diagnostic_frame),
            "redundancy_pairs": redundancy_pairs(diagnostic_frame),
            "artifact": repository_relative_path(normalized_output),
        }

    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("V3.2A style stability audit complete")
    print(f"Players audited: {len(full):,}")
    print(f"Expanded matrix: {expanded_output}")
    print(f"Audit metadata: {audit_output}")
    if normalized is not None:
        print(f"Position-normalized matrix: {normalized_output} ({len(normalized):,} players)")
    return expanded, normalized, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=Path, default=DEFAULT_PASSES)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--expanded-output", type=Path, default=DEFAULT_EXPANDED)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--normalized-output", type=Path)
    parser.add_argument("--normalized-pass-threshold", type=int)
    parser.add_argument("--normalized-carry-threshold", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        args.passes,
        args.actions,
        args.profiles,
        args.matches,
        args.expanded_output,
        args.audit_output,
        args.normalized_output,
        args.normalized_pass_threshold,
        args.normalized_carry_threshold,
    )


if __name__ == "__main__":
    main()
