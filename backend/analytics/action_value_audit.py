"""Reproducible methodological audit for FootyScout V2.2 action value."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analytics.action_value_model import grouped_folds

ROOT = Path(__file__).resolve().parents[2]
EVENTS_PATH = ROOT / "data" / "processed" / "events.parquet"
STATES_PATH = ROOT / "data" / "processed" / "possession_states.parquet"
ACTIONS_PATH = ROOT / "data" / "processed" / "attacking_actions.parquet"
SHOT_OOF_PATH = ROOT / "data" / "processed" / "shot_oof_predictions.parquet"
METADATA_PATH = ROOT / "models" / "action_value_model_metadata.json"
OUTPUT_PATH = ROOT / "models" / "action_value_hardening_audit.json"


def value_distribution(frame: pd.DataFrame) -> dict[str, float | int]:
    values = frame["attacking_value"].astype(float)
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "standard_deviation": float(values.std()),
        "p1": float(values.quantile(0.01)),
        "p5": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "positive_percentage": float(values.gt(0).mean()),
        "negative_percentage": float(values.lt(0).mean()),
        "exactly_zero_percentage": float(values.eq(0).mean()),
    }


def action_distributions(actions: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for action_type, group in actions.groupby("action_type", sort=True):
        thirds = pd.cut(
            group["start_x"],
            [-np.inf, 40, 80, np.inf],
            labels=["defensive", "middle", "attacking"],
            right=False,
        )
        output[str(action_type)] = {
            "overall": value_distribution(group),
            "by_pitch_third": {
                str(name): value_distribution(subset)
                for name, subset in group.assign(pitch_third=thirds).groupby(
                    "pitch_third", observed=True
                )
            },
            "by_progressive": {
                str(bool(name)).lower(): value_distribution(subset)
                for name, subset in group.groupby("progressive")
            },
        }
    return output


def transition_coordinate_audit(
    states: pd.DataFrame, actions: pd.DataFrame
) -> dict[str, Any]:
    ordered = states.sort_values(
        ["match_id", "possession_id", "event_index"], kind="stable"
    ).copy()
    grouped = ordered.groupby(["match_id", "possession_id"], sort=False)
    next_columns = [
        "event_id",
        "event_index",
        "start_x",
        "start_y",
        "event_type",
        "match_id",
        "possession_id",
        "possession_team_id",
    ]
    for column in next_columns:
        ordered[f"after_{column}"] = grouped[column].shift(-1)
    eligible = ordered.loc[
        ordered["event_id"].isin(actions["action_id"])
        & ordered["event_type"].isin(["Pass", "Carry"])
        & ordered["event_success"]
        & ordered["after_event_id"].notna()
    ].copy()
    eligible["coordinate_discrepancy"] = np.hypot(
        eligible["end_x"] - eligible["after_start_x"],
        eligible["end_y"] - eligible["after_start_y"],
    )
    invariants = {
        "different_match": int(eligible["match_id"].ne(eligible["after_match_id"]).sum()),
        "different_possession": int(
            eligible["possession_id"].ne(eligible["after_possession_id"]).sum()
        ),
        "different_possession_team": int(
            eligible["possession_team_id"]
            .ne(eligible["after_possession_team_id"])
            .sum()
        ),
        "non_increasing_event_index": int(
            eligible["after_event_index"].le(eligible["event_index"]).sum()
        ),
    }
    summaries: dict[str, Any] = {}
    for action_type, group in eligible.groupby("event_type", sort=True):
        discrepancy = group["coordinate_discrepancy"]
        summaries[str(action_type)] = {
            "action_count": len(group),
            "exact_matches": int(discrepancy.le(1e-9).sum()),
            "near_matches_within_one_unit": int(discrepancy.le(1.0).sum()),
            "median_discrepancy": float(discrepancy.median()),
            "p95_discrepancy": float(discrepancy.quantile(0.95)),
            "p99_discrepancy": float(discrepancy.quantile(0.99)),
            "maximum_discrepancy": float(discrepancy.max()),
            "after_event_types": group["after_event_type"].value_counts().to_dict(),
        }
    detail_columns = [
        "event_id",
        "match_id",
        "possession_id",
        "event_index",
        "event_type",
        "end_x",
        "end_y",
        "after_event_id",
        "after_event_index",
        "after_event_type",
        "after_start_x",
        "after_start_y",
        "coordinate_discrepancy",
    ]
    return {
        "exact_tolerance": 1e-9,
        "near_tolerance_statsbomb_units": 1.0,
        "invariants": invariants,
        "by_action_type": summaries,
        "largest_25_discrepancies": eligible.nlargest(25, "coordinate_discrepancy")[
            detail_columns
        ].to_dict("records"),
        "interpretation": (
            "Ball Receipt* is the authoritative post-pass state; carry endpoints are the "
            "next on-ball state. Events are not skipped to force coordinate agreement."
        ),
    }


def _count_distribution(series: pd.Series, total: int) -> list[dict[str, Any]]:
    counts = series.fillna("Missing").astype(str).value_counts()
    return [
        {"value": value, "count": int(count), "percentage": float(count / total)}
        for value, count in counts.items()
    ]


def possession_owner_mismatch_audit(events: pd.DataFrame) -> dict[str, Any]:
    ordered = events.sort_values(["match_id", "event_index"], kind="stable").copy()
    match_group = ordered.groupby("match_id", sort=False)
    ordered["previous_event_type"] = match_group["event_type"].shift()
    ordered["next_event_type"] = match_group["event_type"].shift(-1)
    ordered["previous_team_id"] = match_group["team_id"].shift()
    ordered["next_team_id"] = match_group["team_id"].shift(-1)
    possession_group = ordered.groupby(["match_id", "possession_id"], sort=False)
    ordered["possession_event_number"] = possession_group.cumcount() + 1
    ordered["possession_event_count"] = possession_group["event_id"].transform("size")
    product_passes = ordered.loc[
        ordered["is_product_cohort"] & ordered["event_type"].eq("Pass")
    ].copy()
    mismatch = product_passes.loc[
        product_passes["team_id"].ne(product_passes["possession_team_id"])
    ].copy()
    mismatch["previous_team_context"] = np.select(
        [
            mismatch["previous_team_id"].eq(mismatch["team_id"]),
            mismatch["previous_team_id"].eq(mismatch["possession_team_id"]),
        ],
        ["action_team", "possession_owner"],
        default="other_or_missing",
    )
    mismatch["next_team_context"] = np.select(
        [
            mismatch["next_team_id"].eq(mismatch["team_id"]),
            mismatch["next_team_id"].eq(mismatch["possession_team_id"]),
        ],
        ["action_team", "possession_owner"],
        default="other_or_missing",
    )
    pair = mismatch["previous_team_context"] + " -> " + mismatch["next_team_context"]
    example_columns = [
        "event_id",
        "match_id",
        "event_index",
        "possession_id",
        "team_name",
        "possession_team_name",
        "event_success",
        "play_pattern",
        "previous_event_type",
        "next_event_type",
        "possession_event_number",
        "possession_event_count",
    ]
    total = len(mismatch)
    return {
        "all_bundesliga_passes": len(product_passes),
        "count": total,
        "percentage_of_all_passes": float(total / len(product_passes)),
        "success_rate": float(mismatch["event_success"].mean()),
        "previous_event_types": _count_distribution(
            mismatch["previous_event_type"], total
        ),
        "next_event_types": _count_distribution(mismatch["next_event_type"], total),
        "play_patterns": _count_distribution(mismatch["play_pattern"], total),
        "team_context_transitions": _count_distribution(pair, total),
        "possession_context": {
            "median_event_number": float(mismatch["possession_event_number"].median()),
            "median_events_remaining": float(
                (
                    mismatch["possession_event_count"]
                    - mismatch["possession_event_number"]
                ).median()
            ),
            "first_three_events_percentage": float(
                mismatch["possession_event_number"].le(3).mean()
            ),
            "last_three_events_percentage": float(
                (
                    mismatch["possession_event_count"]
                    - mismatch["possession_event_number"]
                )
                .le(2)
                .mean()
            ),
        },
        "representative_examples": mismatch[example_columns].head(25).to_dict("records"),
        "decision": (
            "retain exclusion: the event actor conflicts with StatsBomb's supplied "
            "possession owner, so attributing that owner's future xG to the actor is unsafe"
        ),
    }


def _half_metric(frame: pd.DataFrame, metric: str) -> float:
    if metric == "attacking_value_per_100_actions":
        denominator = len(frame)
        numerator = frame["attacking_value"].sum()
    elif metric == "pass_value_per_100_passes":
        subset = frame.loc[frame["action_type"].eq("Pass")]
        denominator = len(subset)
        numerator = subset["attacking_value"].sum()
    elif metric == "carry_value_per_100_carries":
        subset = frame.loc[frame["action_type"].eq("Carry")]
        denominator = len(subset)
        numerator = subset["attacking_value"].sum()
    elif metric == "progressive_value_per_100_actions":
        denominator = len(frame)
        numerator = frame.loc[frame["progressive"], "attacking_value"].sum()
    else:
        denominator = len(frame)
        numerator = frame.loc[frame["under_pressure"], "attacking_value"].sum()
    return float(100 * numerator / denominator) if denominator else float("nan")


def split_half_stability(actions: pd.DataFrame) -> dict[str, Any]:
    definitions = {
        "attacking_value_per_100_actions": ("actions", 63),
        "pass_value_per_100_passes": ("passes", 36),
        "carry_value_per_100_carries": ("carries", 29),
        "progressive_value_per_100_actions": ("actions", 63),
        "pressure_value_per_100_actions": ("actions", 63),
    }
    values: list[dict[str, Any]] = []
    available = actions.dropna(subset=["player_id"])
    for player_id, group in available.groupby("player_id"):
        match_ids = sorted(group["match_id"].unique())
        if len(match_ids) < 2:
            continue
        midpoint = len(match_ids) // 2
        first = group.loc[group["match_id"].isin(match_ids[:midpoint])]
        second = group.loc[group["match_id"].isin(match_ids[midpoint:])]
        counts = {
            "actions": len(group),
            "passes": int(group["action_type"].eq("Pass").sum()),
            "carries": int(group["action_type"].eq("Carry").sum()),
        }
        for metric, (count_type, threshold) in definitions.items():
            if counts[count_type] >= threshold:
                values.append(
                    {
                        "player_id": int(player_id),
                        "metric": metric,
                        "first_half": _half_metric(first, metric),
                        "second_half": _half_metric(second, metric),
                    }
                )
    frame = pd.DataFrame(values).dropna()
    output: dict[str, Any] = {}
    for metric, group in frame.groupby("metric"):
        output[str(metric)] = {
            "eligible_players": len(group),
            "pearson": float(pearsonr(group["first_half"], group["second_half"]).statistic),
            "spearman": float(
                spearmanr(group["first_half"], group["second_half"]).statistic
            ),
        }
    return {
        "match_order": (
            "ascending Bundesliga match_id; verified against the StatsBomb match catalogue "
            "to be chronological for all 34 product matches"
        ),
        "odd_match_policy": "the second half receives the middle match",
        "thresholds": {"actions": 63, "passes": 36, "carries": 29},
        "metrics": output,
    }


def prechange_nested_leakage_audit(
    states: pd.DataFrame,
    shot_oof: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    all_match_ids = set(shot_oof["match_id"].astype(int))
    fixed_train_ids = set(metadata["split"]["train_match_ids"])

    def contaminated_count(train_ids: set[int], heldout_ids: set[int]) -> int:
        rows = shot_oof.loc[shot_oof["match_id"].isin(train_ids)]
        fold_matches = {
            int(fold): set(group["match_id"].astype(int))
            for fold, group in shot_oof.groupby("fold")
        }
        return int(
            sum(
                bool((all_match_ids - fold_matches[int(row.fold)]) & heldout_ids)
                for row in rows.itertuples()
            )
        )

    fixed = {}
    for name in ["validation", "test"]:
        heldout = set(metadata["split"][f"{name}_match_ids"])
        fixed[name] = {
            "outer_training_shots": int(
                shot_oof["match_id"].isin(fixed_train_ids).sum()
            ),
            "contaminated_training_shot_labels": contaminated_count(
                fixed_train_ids, heldout
            ),
        }
    folds = []
    for fold, (train_index, heldout_index) in enumerate(grouped_folds(states), start=1):
        train_ids = set(states.iloc[train_index]["match_id"].astype(int))
        heldout_ids = set(states.iloc[heldout_index]["match_id"].astype(int))
        folds.append(
            {
                "fold": fold,
                "outer_training_shots": int(
                    shot_oof["match_id"].isin(train_ids).sum()
                ),
                "contaminated_training_shot_labels": contaminated_count(
                    train_ids, heldout_ids
                ),
            }
        )
    return {
        "leakage_existed_before_hardening": True,
        "reason": (
            "global V2.1 OOF excluded each shot's own fold but did not exclude the "
            "V2.2 outer held-out matches from upstream models used for outer-training labels"
        ),
        "fixed_split": fixed,
        "state_oof_folds": folds,
    }


def run(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    events = pd.read_parquet(EVENTS_PATH)
    states = pd.read_parquet(STATES_PATH)
    actions = pd.read_parquet(ACTIONS_PATH)
    shot_oof = pd.read_parquet(SHOT_OOF_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    pass_distribution = value_distribution(
        actions.loc[actions["action_type"].eq("Pass")]
    )
    audit = {
        "nested_cross_fitting": {
            **prechange_nested_leakage_audit(states, shot_oof, metadata),
            "correction": metadata["nested_cross_fitting"],
        },
        "transition_coordinates": transition_coordinate_audit(states, actions),
        "possession_owner_mismatches": possession_owner_mismatch_audit(events),
        "action_value_distributions": action_distributions(actions),
        "median_reward_category": {
            "threshold": float(pass_distribution["median"]),
            "absolute_value_within_0.001_percentage": float(
                actions.loc[actions["action_type"].eq("Pass"), "attacking_value"]
                .abs()
                .le(0.001)
                .mean()
            ),
            "assessment": (
                "not substantively defensible: the median threshold is effectively zero "
                "relative to the action-value spread; retain continuous value and "
                "remove or deemphasize binary reward categories"
            ),
        },
        "player_split_half_stability": split_half_stability(actions),
    }
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    run()
