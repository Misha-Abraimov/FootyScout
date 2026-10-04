"""Offline hurdle-head and same-possession attacking-impact diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def head_metrics(
    target: np.ndarray, probability: np.ndarray, magnitude: np.ndarray
) -> dict[str, float | int | list[dict[str, float | int]]]:
    actual = np.asarray(target, dtype=float)
    p = np.asarray(probability, dtype=float)
    mag = np.asarray(magnitude, dtype=float)
    if len(actual) != len(p) or len(p) != len(mag):
        raise ValueError("Hurdle heads must align with every target")
    if not np.isfinite(p).all() or not np.isfinite(mag).all() or (p < 0).any() or (p > 1).any() or (mag < 0).any():
        raise ValueError("Hurdle heads must be finite probabilities and non-negative magnitudes")
    positive = actual > 0
    bins = np.linspace(0, 1, 11)
    bucket = np.minimum(np.searchsorted(bins, p, side="right") - 1, 9)
    calibration = [
        {"count": int((bucket == index).sum()),
         "mean_probability": float(p[bucket == index].mean()),
         "positive_rate": float(positive[bucket == index].mean())}
        for index in range(10) if (bucket == index).any()
    ]
    ece = sum(
        row["count"] * abs(row["mean_probability"] - row["positive_rate"])
        for row in calibration
    ) / len(actual)
    return {
        "states": len(actual),
        "positive_states": int(positive.sum()),
        "roc_auc": float(roc_auc_score(positive, p)),
        "pr_auc": float(average_precision_score(positive, p)),
        "log_loss": float(log_loss(positive, p, labels=[False, True])),
        "brier_score": float(brier_score_loss(positive, p)),
        "ece_10_bin": float(ece),
        "mean_probability": float(p.mean()),
        "positive_rate": float(positive.mean()),
        "positive_magnitude_mae": float(mean_absolute_error(actual[positive], mag[positive])),
        "positive_magnitude_rmse": float(mean_squared_error(actual[positive], mag[positive]) ** 0.5),
        "positive_magnitude_prediction_mean": float(mag[positive].mean()),
        "positive_magnitude_target_mean": float(actual[positive].mean()),
        "calibration_bins": calibration,
    }


def impact_comparison(actions: pd.DataFrame) -> dict[str, object]:
    """Descriptive comparisons only; the production action definition is unchanged."""
    names = [column.removeprefix("impact_value_") for column in actions if column.startswith("impact_value_")]
    baseline = actions["impact_value_xgboost"].to_numpy(dtype=float)
    result: dict[str, object] = {"actions": len(actions), "models": {}}
    for name in names:
        column = f"impact_value_{name}"
        values = actions[column].to_numpy(dtype=float)
        model: dict[str, object] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "positive_fraction": float(np.mean(values > 0)),
            "minimum": float(np.min(values)),
            "p01": float(np.quantile(values, 0.01)),
            "p99": float(np.quantile(values, 0.99)),
            "maximum": float(np.max(values)),
            "pearson_with_xgboost": float(pearsonr(baseline, values).statistic),
            "spearman_with_xgboost": float(spearmanr(baseline, values).statistic),
            "by_type": {},
        }
        for event_type, group in actions.groupby("event_type"):
            subset = group[column].to_numpy(dtype=float)
            model["by_type"][str(event_type)] = {
                "count": len(subset), "mean": float(np.mean(subset)),
                "median": float(np.median(subset)), "std": float(np.std(subset)),
                "positive_fraction": float(np.mean(subset > 0)),
            }
        result["models"][name] = model
    for name in names:
        if name == "xgboost":
            continue
        differences = actions.assign(
            absolute_difference=(actions[f"impact_value_{name}"] - actions["impact_value_xgboost"]).abs()
        )
        result[f"largest_{name}_disagreements"] = differences.nlargest(10, "absolute_difference")[[
            "event_id", "match_id", "event_type", "impact_value_xgboost",
            f"impact_value_{name}", "absolute_difference",
        ]].to_dict(orient="records")
    return result
