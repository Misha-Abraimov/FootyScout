"""Grouped out-of-fold prediction utilities for pass completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from analytics.pass_features import METADATA_COLUMNS, MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from analytics.pass_model import (
    RANDOM_SEED,
    TrainingConfig,
    XGBoostConfig,
    build_preprocessor,
    evaluate_probabilities,
    fit_logistic_regression,
    fit_temperature,
    fit_xgboost_fixed_rounds,
    grouped_match_folds,
    predict_mlp_logits,
    predict_xgboost_margins,
    probabilities_from_logits,
    train_mlp_fixed_epochs,
)

OOF_PREDICTION_COLUMNS = [
    "pass_index",
    *METADATA_COLUMNS,
    TARGET_COLUMN,
    "expected_completion",
    "under_pressure",
    "progressive",
    "pass_length",
    "forward_distance",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "fold",
]


@dataclass(frozen=True)
class OOFResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]
    fold_summaries: list[dict[str, Any]]
    raw_logits: np.ndarray | None


def _fit_calibration_split(
    outer_training_data: pd.DataFrame,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve whole training matches for fitting a fold-specific temperature."""
    match_ids = sorted(int(value) for value in outer_training_data["match_id"].unique())
    fit_ids, calibration_ids = train_test_split(
        match_ids,
        test_size=0.1,
        random_state=random_seed,
        shuffle=True,
    )
    if set(fit_ids).intersection(calibration_ids):
        raise AssertionError("MLP fit and calibration match IDs overlap")
    fit_data = outer_training_data[outer_training_data["match_id"].isin(fit_ids)].copy()
    calibration_data = outer_training_data[
        outer_training_data["match_id"].isin(calibration_ids)
    ].copy()
    return fit_data, calibration_data


def validate_oof_predictions(predictions: pd.DataFrame, expected_rows: int) -> None:
    """Assert one valid probability for every original pass row."""
    missing_columns = set(OOF_PREDICTION_COLUMNS).difference(predictions.columns)
    if missing_columns:
        raise AssertionError(f"OOF output is missing columns: {sorted(missing_columns)}")
    if len(predictions) != expected_rows:
        raise AssertionError(
            f"OOF row count {len(predictions):,} does not match input {expected_rows:,}"
        )
    if predictions["pass_index"].isna().any() or predictions["pass_index"].duplicated().any():
        raise AssertionError("Every original pass must receive exactly one OOF prediction")
    if set(predictions["pass_index"].astype(int)) != set(range(expected_rows)):
        raise AssertionError("OOF pass indices do not cover every original row")
    probabilities = predictions["expected_completion"]
    if probabilities.isna().any() or not probabilities.between(0.0, 1.0).all():
        raise AssertionError("OOF expected completion must always be between 0 and 1")
    if predictions["fold"].isna().any():
        raise AssertionError("Every OOF prediction must identify its held-out fold")


def generate_oof_predictions(
    data: pd.DataFrame,
    selected_model: str,
    training_config: TrainingConfig,
    mlp_epochs: int,
    use_temperature_scaling: bool,
    xgboost_config: XGBoostConfig | None = None,
    xgboost_rounds: int | None = None,
    n_splits: int = 5,
    random_seed: int = RANDOM_SEED,
) -> OOFResult:
    """Cross-fit the selected model so every match is predicted out of sample."""
    if selected_model not in {"logistic_regression", "pytorch_mlp", "xgboost"}:
        raise ValueError(f"Unsupported selected model: {selected_model}")
    if selected_model == "xgboost" and (xgboost_config is None or xgboost_rounds is None):
        raise ValueError("Selected XGBoost OOF requires frozen config and boosting rounds")

    indexed_data = data.reset_index(drop=True).copy()
    indexed_data["pass_index"] = np.arange(len(indexed_data), dtype=np.int64)
    fold_outputs: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    raw_logits_by_row = np.full(len(indexed_data), np.nan, dtype=np.float32)

    for fold in grouped_match_folds(indexed_data, n_splits, random_seed):
        outer_training = indexed_data.iloc[fold.train_indices]
        heldout = indexed_data.iloc[fold.heldout_indices]
        if set(outer_training["match_id"]).intersection(heldout["match_id"]):
            raise AssertionError(f"Fold {fold.fold} leaks a held-out match into training")

        fold_seed = random_seed + fold.fold
        temperature = 1.0
        calibration_match_count = 0
        if selected_model in {"pytorch_mlp", "xgboost"} and use_temperature_scaling:
            model_fit_data, calibration_data = _fit_calibration_split(
                outer_training,
                fold_seed,
            )
            calibration_match_count = int(calibration_data["match_id"].nunique())
        else:
            model_fit_data = outer_training
            calibration_data = None

        preprocessor = build_preprocessor()
        fit_matrix = np.asarray(
            preprocessor.fit_transform(model_fit_data[MODEL_FEATURE_COLUMNS]),
            dtype=np.float32,
        )
        heldout_matrix = np.asarray(
            preprocessor.transform(heldout[MODEL_FEATURE_COLUMNS]),
            dtype=np.float32,
        )
        fit_target = model_fit_data[TARGET_COLUMN].to_numpy(dtype=np.float32)

        if selected_model == "logistic_regression":
            model = fit_logistic_regression(fit_matrix, fit_target, fold_seed)
            probabilities = model.predict_proba(heldout_matrix)[:, 1]
        elif selected_model == "pytorch_mlp":
            training_result = train_mlp_fixed_epochs(
                fit_matrix,
                fit_target,
                epochs=mlp_epochs,
                config=training_config,
                random_seed=fold_seed,
                progress_label=f"OOF fold {fold.fold}",
            )
            if calibration_data is not None:
                calibration_matrix = np.asarray(
                    preprocessor.transform(calibration_data[MODEL_FEATURE_COLUMNS]),
                    dtype=np.float32,
                )
                calibration_logits = predict_mlp_logits(
                    training_result.model,
                    calibration_matrix,
                )
                temperature = fit_temperature(
                    calibration_logits,
                    calibration_data[TARGET_COLUMN].to_numpy(dtype=np.float32),
                )
            heldout_logits = predict_mlp_logits(training_result.model, heldout_matrix)
            raw_logits_by_row[fold.heldout_indices] = heldout_logits
            probabilities = probabilities_from_logits(heldout_logits, temperature)
        else:
            assert xgboost_config is not None and xgboost_rounds is not None
            model = fit_xgboost_fixed_rounds(
                fit_matrix,
                fit_target,
                xgboost_config,
                xgboost_rounds,
                fold_seed,
            )
            if calibration_data is not None:
                calibration_matrix = np.asarray(
                    preprocessor.transform(calibration_data[MODEL_FEATURE_COLUMNS]),
                    dtype=np.float32,
                )
                temperature = fit_temperature(
                    predict_xgboost_margins(model, calibration_matrix),
                    calibration_data[TARGET_COLUMN].to_numpy(dtype=np.float32),
                )
            heldout_logits = predict_xgboost_margins(model, heldout_matrix)
            raw_logits_by_row[fold.heldout_indices] = heldout_logits
            probabilities = probabilities_from_logits(heldout_logits, temperature)

        output = heldout[
            [
                "pass_index",
                *METADATA_COLUMNS,
                TARGET_COLUMN,
                "under_pressure",
                "progressive",
                "pass_length",
                "forward_distance",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
            ]
        ].copy()
        output["expected_completion"] = probabilities
        output["fold"] = fold.fold
        fold_outputs.append(output[OOF_PREDICTION_COLUMNS])
        fold_summaries.append(
            {
                "fold": fold.fold,
                "train_match_count": len(fold.train_match_ids),
                "heldout_match_count": len(fold.heldout_match_ids),
                "model_fit_pass_count": len(model_fit_data),
                "calibration_match_count": calibration_match_count,
                "heldout_pass_count": len(heldout),
                "heldout_match_ids": list(fold.heldout_match_ids),
                "temperature": temperature if use_temperature_scaling else None,
            }
        )
        print(
            f"OOF fold {fold.fold}/{n_splits}: predicted {len(heldout):,} passes "
            f"from {len(fold.heldout_match_ids)} held-out matches"
        )

    predictions = pd.concat(fold_outputs, ignore_index=True).sort_values("pass_index")
    predictions = predictions.reset_index(drop=True)
    validate_oof_predictions(predictions, len(indexed_data))
    metrics = evaluate_probabilities(
        predictions[TARGET_COLUMN].to_numpy(),
        predictions["expected_completion"].to_numpy(),
    )
    raw_logits = raw_logits_by_row if selected_model in {"pytorch_mlp", "xgboost"} else None
    if raw_logits is not None and np.isnan(raw_logits).any():
        raise AssertionError("Every MLP OOF row must have one raw logit")
    return OOFResult(
        predictions=predictions,
        metrics=metrics,
        fold_summaries=fold_summaries,
        raw_logits=raw_logits,
    )
