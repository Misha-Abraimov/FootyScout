"""Create reliability-aware Bundesliga player attacking-value profiles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def distribution(values: pd.Series) -> dict[str, float]:
    numeric = values.astype(float)
    return {
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "p75": float(numeric.quantile(0.75)),
        "p90": float(numeric.quantile(0.90)),
    }


def build_player_attacking_profiles(
    actions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate product-cohort actions; keep every observed player."""
    source = actions.loc[actions["player_id"].notna()].copy()
    grouped = source.groupby("player_id", sort=True)
    rows: list[dict[str, object]] = []
    for player_id, group in grouped:
        passes = group.loc[group["action_type"].eq("Pass")]
        carries = group.loc[group["action_type"].eq("Carry")]
        actions_count = len(group)
        pass_count = len(passes)
        carry_count = len(carries)
        total = float(group["attacking_value"].sum())
        pass_total = float(passes["attacking_value"].sum())
        carry_total = float(carries["attacking_value"].sum())
        progressive_total = float(group.loc[group["progressive"], "attacking_value"].sum())
        pressure_total = float(group.loc[group["under_pressure"], "attacking_value"].sum())
        rows.append(
            {
                "player_id": int(player_id),
                "matches_observed": int(group["match_id"].nunique()),
                "actions": actions_count,
                "passes": pass_count,
                "carries": carry_count,
                "total_attacking_value": total,
                "attacking_value_per_100_actions": 100 * total / actions_count,
                "total_pass_value": pass_total,
                "pass_value_per_100_passes": 100 * pass_total / pass_count if pass_count else np.nan,
                "total_carry_value": carry_total,
                "carry_value_per_100_carries": 100 * carry_total / carry_count if carry_count else np.nan,
                "positive_value_actions": int(group["attacking_value"].gt(0).sum()),
                "positive_value_action_rate": float(group["attacking_value"].gt(0).mean()),
                "progressive_action_value": progressive_total,
                "progressive_value_per_100_actions": 100 * progressive_total / actions_count,
                "pressure_action_value": pressure_total,
                "pressure_value_per_100_actions": 100 * pressure_total / actions_count,
            }
        )
    profiles = pd.DataFrame(rows)
    stats = {
        "actions": distribution(profiles["actions"]),
        "passes": distribution(profiles["passes"]),
        "carries": distribution(profiles["carries"]),
    }
    thresholds = {
        name: max(1, math.ceil(values["median"])) for name, values in stats.items()
    }
    profiles["attacking_value_reliable"] = profiles["actions"].ge(thresholds["actions"])
    profiles["pass_value_reliable"] = profiles["passes"].ge(thresholds["passes"])
    profiles["carry_value_reliable"] = profiles["carries"].ge(thresholds["carries"])
    return profiles, {"distributions": stats, "thresholds": thresholds}
