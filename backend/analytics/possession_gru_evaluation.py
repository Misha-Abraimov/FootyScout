"""Offline regression and downstream transition diagnostics; no production writes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(target, dtype=float)
    prediction = np.asarray(predicted, dtype=float)
    if len(actual) != len(prediction) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Predictions must be complete, finite and non-negative")
    positive = actual > 0
    if not positive.any():
        raise ValueError("Positive-target diagnostics require at least one positive state")
    pearson = pearsonr(actual, prediction).statistic
    spearman = spearmanr(actual, prediction).statistic
    return {
        "states": len(actual),
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
        "pearson": float(pearson) if np.isfinite(pearson) else 0.0,
        "spearman": float(spearman) if np.isfinite(spearman) else 0.0,
        "positive_states": int(positive.sum()),
        "positive_target_mae": float(mean_absolute_error(actual[positive], prediction[positive])),
        "positive_target_rmse": float(mean_squared_error(actual[positive], prediction[positive]) ** 0.5),
        "target_mean": float(actual.mean()),
        "prediction_mean": float(prediction.mean()),
        "prediction_min": float(prediction.min()),
        "prediction_p99": float(np.quantile(prediction, 0.99)),
        "prediction_max": float(prediction.max()),
    }


def action_values_from_state_predictions(
    states: pd.DataFrame, predicted: pd.DataFrame
) -> pd.DataFrame:
    """Mirror frozen V2.2 before/after alignment for analysis on any match cohort.

    A successful pass/carry uses the next eligible pre-event state from its own
    possession. Failed or terminal actions use zero. Never cross possessions.
    """
    if predicted["event_id"].duplicated().any() or states["event_id"].duplicated().any():
        raise ValueError("States and predictions must be unique by event ID")
    value_columns = [column for column in predicted if column.startswith("value_")]
    if not value_columns:
        raise ValueError("At least one value_* prediction column is required")
    merged = states.merge(
        predicted[["event_id", *value_columns]], on="event_id", validate="one_to_one"
    ).sort_values(["match_id", "possession_id", "event_index"], kind="stable")
    if len(merged) != len(states) or merged[value_columns].isna().any().any():
        raise ValueError("Every state needs every prediction")
    group = merged.groupby(["match_id", "possession_id"], sort=False)
    selected = merged.event_type.isin(["Pass", "Carry"]) & merged.end_x.notna() & merged.end_y.notna()
    actions = merged.loc[selected, [
        "event_id", "match_id", "possession_id", "event_index", "event_type", "event_success"
    ]].copy()
    for column in value_columns:
        after = group[column].shift(-1)
        continues = merged.event_success.astype(bool) & after.notna()
        actions[f"before_{column}"] = merged.loc[selected, column].to_numpy()
        actions[f"after_{column}"] = np.where(
            continues.loc[selected], after.loc[selected], 0.0
        )
        actions[f"impact_{column}"] = (
            actions[f"after_{column}"] - actions[f"before_{column}"]
        )
    return actions.reset_index(drop=True)


def action_comparison(actions: pd.DataFrame) -> dict[str, object]:
    xgb = actions["impact_value_xgboost"].to_numpy(dtype=float)
    gru = actions["impact_value_gru"].to_numpy(dtype=float)
    correlation = pearsonr(xgb, gru).statistic if len(actions) > 1 else np.nan
    summary: dict[str, object] = {
        "actions": len(actions),
        "pearson_impact": float(correlation) if np.isfinite(correlation) else None,
        "models": {},
    }
    for name in ("xgboost", "gru"):
        values = actions[f"impact_value_{name}"]
        summary["models"][name] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "positive_fraction": float(values.gt(0).mean()),
            "p01": float(values.quantile(0.01)),
            "p99": float(values.quantile(0.99)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "by_action_type": {
                str(action_type): {
                    "count": len(group),
                    "mean": float(group[f"impact_value_{name}"].mean()),
                    "median": float(group[f"impact_value_{name}"].median()),
                    "positive_fraction": float(group[f"impact_value_{name}"].gt(0).mean()),
                }
                for action_type, group in actions.groupby("event_type")
            },
        }
    differences = actions.assign(absolute_difference=np.abs(gru - xgb))
    summary["largest_disagreements"] = differences.nlargest(10, "absolute_difference")[
        ["event_id", "match_id", "event_type", "impact_value_xgboost", "impact_value_gru", "absolute_difference"]
    ].to_dict(orient="records")
    return summary
