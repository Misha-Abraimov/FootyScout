"""Run rigorous evaluation, OOF scoring, and production training for FootyScout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPOSITORY_ROOT / "models" / ".matplotlib"))
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from analytics.artifact_paths import repository_relative_path
from analytics.pass_features import METADATA_COLUMNS, MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from analytics.pass_model import (
    MODEL_ARCHITECTURE,
    MODEL_BOOLEAN_COLUMNS,
    MODEL_CATEGORICAL_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    RANDOM_SEED,
    XGBOOST_CANDIDATES,
    XGBOOST_VERSION,
    TrainingConfig,
    XGBoostConfig,
    build_preprocessor,
    evaluate_probabilities,
    evaluate_temperature_scaling,
    fit_logistic_regression,
    fit_preprocessor,
    fit_temperature,
    fit_xgboost_fixed_rounds,
    predict_mlp_logits,
    predict_xgboost_margins,
    probabilities_from_logits,
    save_logistic_model,
    save_preprocessor,
    save_torch_model,
    save_training_history,
    save_xgboost_model,
    select_model,
    split_by_match,
    train_logistic_regression,
    train_mlp,
    train_mlp_fixed_epochs,
    train_xgboost_candidate,
    training_config_dict,
    validate_model_contract,
    xgboost_config_dict,
    xgboost_gain_importance,
)
from analytics.pass_oof import OOFResult, generate_oof_predictions

DEFAULT_INPUT_PATH = REPOSITORY_ROOT / "data" / "processed" / "pass_features.parquet"
DEFAULT_MODELS_DIRECTORY = REPOSITORY_ROOT / "models"
DEFAULT_TEST_PREDICTIONS_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "pass_test_predictions.parquet"
)
DEFAULT_OOF_PREDICTIONS_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "pass_oof_predictions.parquet"
)

PREPROCESSOR_FILENAME = "pass_preprocessor.joblib"
LOGISTIC_FILENAME = "pass_logistic_regression.joblib"
PYTORCH_FILENAME = "pass_completion_model.pt"
METADATA_FILENAME = "pass_model_metadata.json"
CALIBRATION_FILENAME = "pass_model_calibration.png"
HISTORY_FILENAME = "pass_mlp_history.json"
FINAL_PREPROCESSOR_FILENAME = "pass_preprocessor_final.joblib"
FINAL_PYTORCH_FILENAME = "pass_completion_model_final.pt"
FINAL_LOGISTIC_FILENAME = "pass_logistic_regression_final.joblib"
FINAL_HISTORY_FILENAME = "pass_mlp_history_final.json"
XGBOOST_BENCHMARK_FILENAME = "pass_xgboost_benchmark.json"
XGBOOST_PREPROCESSOR_FILENAME = "pass_xgboost_preprocessor.joblib"
FINAL_XGBOOST_FILENAME = "pass_xgboost_model.json"
FINAL_XGBOOST_PREPROCESSOR_FILENAME = "pass_xgboost_preprocessor_final.joblib"
OOF_FOLDS = 5

TEST_PREDICTION_COLUMNS = [
    *METADATA_COLUMNS,
    TARGET_COLUMN,
    "expected_completion",
    "logistic_expected_completion",
    "pytorch_uncalibrated_expected_completion",
    "pytorch_effective_expected_completion",
    "xgboost_uncalibrated_expected_completion",
    "xgboost_effective_expected_completion",
    "under_pressure",
    "progressive",
    "pass_length",
    "forward_distance",
]


def _split_summary(frame: pd.DataFrame, match_ids: tuple[int, ...]) -> dict[str, Any]:
    return {
        "match_count": len(match_ids),
        "match_ids": list(match_ids),
        "pass_count": len(frame),
        "completion_rate": float(frame[TARGET_COLUMN].mean()),
    }


def _print_split_summary(name: str, summary: dict[str, Any]) -> None:
    print(
        f"{name}: {summary['match_count']} matches, {summary['pass_count']:,} passes, "
        f"completion rate {summary['completion_rate']:.2%}"
    )


def _print_metrics(name: str, split_name: str, metrics: dict[str, float]) -> None:
    print(f"{name} {split_name} metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")


def save_calibration_plot(
    target: np.ndarray,
    curves: list[tuple[str, np.ndarray]],
    path: Path,
) -> None:
    """Save a reliability plot using predictions never used for selection."""
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    for label, probabilities in curves:
        observed, predicted = calibration_curve(
            target,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", linewidth=2, label=label)

    axis.set(
        xlabel="Mean predicted completion probability",
        ylabel="Observed completion rate",
        title="FootyScout reliability on untouched test matches",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_test_predictions(
    test_data: pd.DataFrame,
    logistic_probabilities: np.ndarray,
    pytorch_uncalibrated: np.ndarray,
    pytorch_effective: np.ndarray,
    xgboost_uncalibrated: np.ndarray,
    xgboost_effective: np.ndarray,
    selected_model: str,
    path: Path,
) -> None:
    predictions = test_data[
        METADATA_COLUMNS
        + [TARGET_COLUMN, "under_pressure", "progressive", "pass_length", "forward_distance"]
    ].copy()
    predictions["logistic_expected_completion"] = logistic_probabilities
    predictions["pytorch_uncalibrated_expected_completion"] = pytorch_uncalibrated
    predictions["pytorch_effective_expected_completion"] = pytorch_effective
    predictions["xgboost_uncalibrated_expected_completion"] = xgboost_uncalibrated
    predictions["xgboost_effective_expected_completion"] = xgboost_effective
    selected_probabilities = {
        "logistic_regression": logistic_probabilities,
        "pytorch_mlp": pytorch_effective,
        "xgboost": xgboost_effective,
    }
    predictions["expected_completion"] = selected_probabilities[selected_model]
    predictions = predictions[TEST_PREDICTION_COLUMNS]
    if not predictions["expected_completion"].between(0.0, 1.0).all():
        raise AssertionError("Test predictions must be probabilities from 0 to 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)


def _train_production_artifacts(
    data: pd.DataFrame,
    selected_model: str,
    training_config: TrainingConfig,
    mlp_epochs: int,
    calibration_retained: bool,
    oof_result: OOFResult,
    models_directory: Path,
    xgboost_config: XGBoostConfig | None = None,
    xgboost_rounds: int | None = None,
) -> dict[str, Any]:
    """Fit the selected model on all rows strictly for future inference."""
    final_preprocessor = build_preprocessor()
    full_matrix = np.asarray(
        final_preprocessor.fit_transform(data[MODEL_FEATURE_COLUMNS]),
        dtype=np.float32,
    )
    full_target = data[TARGET_COLUMN].to_numpy(dtype=np.float32)
    final_preprocessor_path = models_directory / (
        FINAL_XGBOOST_PREPROCESSOR_FILENAME
        if selected_model == "xgboost"
        else FINAL_PREPROCESSOR_FILENAME
    )
    save_preprocessor(final_preprocessor, final_preprocessor_path)

    artifact: dict[str, Any] = {
        "training_rows": len(data),
        "evaluation_use": "none; full-data model is for future inference only",
        "preprocessor": repository_relative_path(final_preprocessor_path),
        "encoded_feature_count": int(full_matrix.shape[1]),
    }
    if selected_model == "logistic_regression":
        final_model = fit_logistic_regression(full_matrix, full_target, RANDOM_SEED)
        final_model_path = models_directory / FINAL_LOGISTIC_FILENAME
        save_logistic_model(final_model, final_model_path)
        artifact.update(
            {
                "model": "logistic_regression",
                "model_path": repository_relative_path(final_model_path),
            }
        )
    elif selected_model == "pytorch_mlp":
        production_result = train_mlp_fixed_epochs(
            full_matrix,
            full_target,
            epochs=mlp_epochs,
            config=training_config,
            random_seed=RANDOM_SEED,
            progress_label="Production MLP",
        )
        production_temperature = 1.0
        temperature_source = "not used"
        if calibration_retained:
            if oof_result.raw_logits is None:
                raise AssertionError("Calibrated production MLP requires cross-fitted logits")
            production_temperature = fit_temperature(oof_result.raw_logits, full_target)
            temperature_source = (
                "fitted after OOF evaluation using cross-fitted logits; not used for metrics"
            )
        final_model_path = models_directory / FINAL_PYTORCH_FILENAME
        final_history_path = models_directory / FINAL_HISTORY_FILENAME
        save_torch_model(
            production_result.model,
            final_model_path,
            temperature=production_temperature,
        )
        save_training_history(production_result, final_history_path)
        artifact.update(
            {
                "model": "pytorch_mlp",
                "model_path": repository_relative_path(final_model_path),
                "history_path": repository_relative_path(final_history_path),
                "epochs": mlp_epochs,
                "temperature": production_temperature,
                "temperature_source": temperature_source,
            }
        )
    else:
        if xgboost_config is None or xgboost_rounds is None:
            raise ValueError("Production XGBoost requires frozen parameters and rounds")
        final_model = fit_xgboost_fixed_rounds(
            full_matrix,
            full_target,
            xgboost_config,
            xgboost_rounds,
            RANDOM_SEED,
        )
        production_temperature = 1.0
        temperature_source = "not used"
        if calibration_retained:
            if oof_result.raw_logits is None:
                raise AssertionError("Calibrated production XGBoost requires cross-fitted margins")
            production_temperature = fit_temperature(oof_result.raw_logits, full_target)
            temperature_source = (
                "fitted after OOF evaluation using cross-fitted margins; not used for metrics"
            )
        final_model_path = models_directory / FINAL_XGBOOST_FILENAME
        save_xgboost_model(final_model, final_model_path)
        artifact.update(
            {
                "model": "xgboost",
                "model_path": repository_relative_path(final_model_path),
                "boosting_rounds": xgboost_rounds,
                "temperature": production_temperature,
                "temperature_source": temperature_source,
            }
        )
    return artifact


def train_and_save(
    input_path: Path = DEFAULT_INPUT_PATH,
    models_directory: Path = DEFAULT_MODELS_DIRECTORY,
    test_predictions_path: Path = DEFAULT_TEST_PREDICTIONS_PATH,
    oof_predictions_path: Path = DEFAULT_OOF_PREDICTIONS_PATH,
) -> dict[str, Any]:
    """Evaluate without test selection, cross-fit all rows, then train production."""
    validate_model_contract()
    data = pd.read_parquet(input_path)
    required_columns = set(METADATA_COLUMNS + MODEL_FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Feature dataset is missing columns: {sorted(missing_columns)}")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("The completed target contains missing values")

    splits = split_by_match(data, RANDOM_SEED)
    split_information = {
        "train": _split_summary(splits.train, splits.train_match_ids),
        "validation": _split_summary(splits.validation, splits.validation_match_ids),
        "test": _split_summary(splits.test, splits.test_match_ids),
    }
    print(f"Dataset: {len(data):,} passes across {data['match_id'].nunique()} matches")
    for split_name in ("train", "validation", "test"):
        _print_split_summary(split_name.capitalize(), split_information[split_name])

    preprocessor = build_preprocessor()
    matrices = fit_preprocessor(preprocessor, splits)
    print(f"Evaluation encoded feature width: {matrices.train.shape[1]}")
    training_config = TrainingConfig()

    logistic_model = train_logistic_regression(matrices, RANDOM_SEED)
    logistic_validation_probabilities = logistic_model.predict_proba(matrices.validation)[:, 1]
    logistic_validation_metrics = evaluate_probabilities(
        matrices.validation_target,
        logistic_validation_probabilities,
    )

    training_result = train_mlp(matrices, training_config)
    validation_logits = predict_mlp_logits(training_result.model, matrices.validation)
    calibration = evaluate_temperature_scaling(
        validation_logits,
        matrices.validation_target,
    )
    evaluation_temperature = calibration.temperature if calibration.retained else 1.0
    pytorch_validation_metrics = (
        calibration.calibrated_metrics if calibration.retained else calibration.uncalibrated_metrics
    )

    print(
        f"Validation class balance: {int(matrices.validation_target.sum()):,} completed / "
        f"{len(matrices.validation_target) - int(matrices.validation_target.sum()):,} incomplete "
        f"({float(matrices.validation_target.mean()):.2%} completed)"
    )
    xgboost_candidates = [
        train_xgboost_candidate(matrices, candidate) for candidate in XGBOOST_CANDIDATES
    ]
    xgboost_result = min(
        xgboost_candidates,
        key=lambda result: (result.metrics["log_loss"], result.metrics["brier_score"]),
    )
    xgboost_frozen_model = fit_xgboost_fixed_rounds(
        matrices.train,
        matrices.train_target,
        xgboost_result.config,
        xgboost_result.boosting_rounds,
        RANDOM_SEED,
    )
    xgboost_validation_margins = predict_xgboost_margins(
        xgboost_frozen_model,
        matrices.validation,
    )
    xgboost_calibration = evaluate_temperature_scaling(
        xgboost_validation_margins,
        matrices.validation_target,
    )
    xgboost_temperature = xgboost_calibration.temperature if xgboost_calibration.retained else 1.0
    xgboost_validation_metrics = (
        xgboost_calibration.calibrated_metrics
        if xgboost_calibration.retained
        else xgboost_calibration.uncalibrated_metrics
    )
    selected_calibration_retained = {
        "logistic_regression": False,
        "pytorch_mlp": calibration.retained,
        "xgboost": xgboost_calibration.retained,
    }

    selected_model, selection_reason = select_model(
        logistic_validation_metrics,
        pytorch_validation_metrics,
        xgboost_validation_metrics,
    )
    _print_metrics("Logistic Regression", "validation", logistic_validation_metrics)
    _print_metrics("PyTorch MLP uncalibrated", "validation", calibration.uncalibrated_metrics)
    _print_metrics("PyTorch MLP temperature-scaled", "validation", calibration.calibrated_metrics)
    print(
        f"Temperature scaling: {'retained' if calibration.retained else 'not retained'} "
        f"(T={calibration.temperature:.6f}; {calibration.reason})"
    )
    for index, candidate in enumerate(xgboost_candidates, start=1):
        _print_metrics(f"XGBoost candidate {index}", "validation", candidate.metrics)
        print(
            f"  best_iteration: {candidate.best_iteration}; "
            f"boosting_rounds: {candidate.boosting_rounds}; "
            f"parameters: {xgboost_config_dict(candidate.config)}"
        )
    _print_metrics(
        "XGBoost temperature-scaled",
        "validation",
        xgboost_calibration.calibrated_metrics,
    )
    print(
        f"XGBoost temperature scaling: "
        f"{'retained' if xgboost_calibration.retained else 'not retained'} "
        f"(T={xgboost_calibration.temperature:.6f}; {xgboost_calibration.reason})"
    )
    print(f"Validation-selected model: {selected_model} ({selection_reason})")
    print("Model and calibration decisions are now frozen; evaluating untouched test matches.")

    logistic_test_probabilities = logistic_model.predict_proba(matrices.test)[:, 1]
    test_logits = predict_mlp_logits(training_result.model, matrices.test)
    pytorch_test_uncalibrated = probabilities_from_logits(test_logits)
    pytorch_test_effective = probabilities_from_logits(test_logits, evaluation_temperature)
    xgboost_test_margins = predict_xgboost_margins(xgboost_frozen_model, matrices.test)
    xgboost_test_uncalibrated = probabilities_from_logits(xgboost_test_margins)
    xgboost_test_effective = probabilities_from_logits(
        xgboost_test_margins,
        xgboost_temperature,
    )
    logistic_test_metrics = evaluate_probabilities(
        matrices.test_target,
        logistic_test_probabilities,
    )
    pytorch_test_uncalibrated_metrics = evaluate_probabilities(
        matrices.test_target,
        pytorch_test_uncalibrated,
    )
    pytorch_test_effective_metrics = evaluate_probabilities(
        matrices.test_target,
        pytorch_test_effective,
    )
    xgboost_test_uncalibrated_metrics = evaluate_probabilities(
        matrices.test_target,
        xgboost_test_uncalibrated,
    )
    xgboost_test_effective_metrics = evaluate_probabilities(
        matrices.test_target,
        xgboost_test_effective,
    )
    selected_test_metrics = {
        "logistic_regression": logistic_test_metrics,
        "pytorch_mlp": pytorch_test_effective_metrics,
        "xgboost": xgboost_test_effective_metrics,
    }[selected_model]
    _print_metrics("Logistic Regression", "test", logistic_test_metrics)
    _print_metrics("PyTorch MLP effective", "test", pytorch_test_effective_metrics)
    _print_metrics("XGBoost uncalibrated", "test", xgboost_test_uncalibrated_metrics)
    _print_metrics("XGBoost effective", "test", xgboost_test_effective_metrics)

    preprocessor_path = models_directory / PREPROCESSOR_FILENAME
    logistic_path = models_directory / LOGISTIC_FILENAME
    pytorch_path = models_directory / PYTORCH_FILENAME
    metadata_path = models_directory / METADATA_FILENAME
    calibration_path = models_directory / CALIBRATION_FILENAME
    history_path = models_directory / HISTORY_FILENAME
    xgboost_path = models_directory / XGBOOST_BENCHMARK_FILENAME
    xgboost_preprocessor_path = models_directory / XGBOOST_PREPROCESSOR_FILENAME
    save_preprocessor(preprocessor, preprocessor_path)
    save_logistic_model(logistic_model, logistic_path)
    save_torch_model(
        training_result.model,
        pytorch_path,
        temperature=evaluation_temperature,
    )
    save_training_history(training_result, history_path)
    save_xgboost_model(xgboost_frozen_model, xgboost_path)
    save_preprocessor(preprocessor, xgboost_preprocessor_path)
    save_calibration_plot(
        matrices.test_target,
        [
            ("Logistic Regression", logistic_test_probabilities),
            (
                "PyTorch MLP" + (" + temperature" if calibration.retained else ""),
                pytorch_test_effective,
            ),
            (
                "XGBoost" + (" + temperature" if xgboost_calibration.retained else ""),
                xgboost_test_effective,
            ),
        ],
        calibration_path,
    )
    _save_test_predictions(
        splits.test,
        logistic_test_probabilities,
        pytorch_test_uncalibrated,
        pytorch_test_effective,
        xgboost_test_uncalibrated,
        xgboost_test_effective,
        selected_model,
        test_predictions_path,
    )

    print(f"Generating {OOF_FOLDS}-fold grouped OOF predictions using {selected_model}.")
    oof_result = generate_oof_predictions(
        data,
        selected_model=selected_model,
        training_config=training_config,
        mlp_epochs=training_result.best_epoch,
        use_temperature_scaling=selected_calibration_retained[selected_model],
        xgboost_config=xgboost_result.config if selected_model == "xgboost" else None,
        xgboost_rounds=xgboost_result.boosting_rounds if selected_model == "xgboost" else None,
        n_splits=OOF_FOLDS,
        random_seed=RANDOM_SEED,
    )
    oof_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    oof_result.predictions.to_parquet(oof_predictions_path, index=False)
    _print_metrics(selected_model, "out-of-fold", oof_result.metrics)

    player_profiles_regenerated = selected_model == "xgboost"
    if player_profiles_regenerated:
        from analytics.player_profiles import run_pipeline as run_profile_pipeline

        profile_path = REPOSITORY_ROOT / "data" / "processed" / "player_profiles.parquet"
        run_profile_pipeline(oof_predictions_path, profile_path)

    print("Training the final full-data production artifact (not used for evaluation).")
    production = _train_production_artifacts(
        data,
        selected_model,
        training_config,
        training_result.best_epoch,
        selected_calibration_retained[selected_model],
        oof_result,
        models_directory,
        xgboost_result.config,
        xgboost_result.boosting_rounds,
    )
    xgboost_importance = xgboost_gain_importance(
        xgboost_frozen_model,
        preprocessor.get_feature_names_out().tolist(),
    )

    metadata: dict[str, Any] = {
        "dataset": {
            "path": repository_relative_path(input_path),
            "pass_count": len(data),
            "match_count": int(data["match_id"].nunique()),
            "completion_rate": float(data[TARGET_COLUMN].mean()),
        },
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "models_evaluated": ["logistic_regression", "pytorch_mlp", "xgboost"],
        "numeric_feature_columns": MODEL_NUMERIC_COLUMNS,
        "boolean_feature_columns": MODEL_BOOLEAN_COLUMNS,
        "categorical_feature_columns": MODEL_CATEGORICAL_COLUMNS,
        "evaluation_encoded_feature_count": int(matrices.train.shape[1]),
        "preprocessing": {
            "evaluation_fit_split": "train only",
            "oof_fit_policy": "fit separately inside each outer training fold",
            "numeric": "median imputation then StandardScaler",
            "boolean": "explicit 0/1 conversion then most-frequent imputation",
            "categorical": (
                "most-frequent imputation then OneHotEncoder(handle_unknown='ignore', "
                "sparse_output=False)"
            ),
            "evaluation_feature_names": preprocessor.get_feature_names_out().tolist(),
        },
        "split": {
            "method": "grouped by match_id; whole matches; approximately 80/10/10",
            "random_seed": RANDOM_SEED,
            **split_information,
        },
        "pytorch": {
            "architecture": MODEL_ARCHITECTURE,
            "loss": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "training_parameters": training_config_dict(training_config),
            "epochs_trained": training_result.epochs_trained,
            "best_epoch": training_result.best_epoch,
            "best_validation_loss": training_result.best_validation_loss,
            "stopped_early": training_result.stopped_early,
        },
        "temperature_scaling": {
            "fit_split": "validation only",
            "temperature": calibration.temperature,
            "retained": calibration.retained,
            "reason": calibration.reason,
            "uncalibrated_validation_metrics": calibration.uncalibrated_metrics,
            "calibrated_validation_metrics": calibration.calibrated_metrics,
        },
        "xgboost": {
            "version": XGBOOST_VERSION,
            "random_seed": RANDOM_SEED,
            "selection_metric": "validation log loss",
            "selected_parameters": xgboost_config_dict(xgboost_result.config),
            "best_iteration": xgboost_result.best_iteration,
            "boosting_rounds": xgboost_result.boosting_rounds,
            "candidates": [
                {
                    "parameters": xgboost_config_dict(candidate.config),
                    "best_iteration": candidate.best_iteration,
                    "boosting_rounds": candidate.boosting_rounds,
                    "validation_metrics": candidate.metrics,
                }
                for candidate in xgboost_candidates
            ],
            "calibration": {
                "method": "temperature_scaling_on_raw_margins",
                "fit_split": "validation only",
                "temperature": xgboost_calibration.temperature,
                "retained": xgboost_calibration.retained,
                "reason": xgboost_calibration.reason,
                "uncalibrated_validation_metrics": xgboost_calibration.uncalibrated_metrics,
                "calibrated_validation_metrics": xgboost_calibration.calibrated_metrics,
            },
            "feature_importance": {
                "importance_type": "gain",
                "interpretation": "model feature importance; not a causal claim",
                "features": xgboost_importance,
            },
        },
        "metrics": {
            "validation": {
                "logistic_regression": logistic_validation_metrics,
                "pytorch_mlp_uncalibrated": calibration.uncalibrated_metrics,
                "pytorch_mlp_effective": pytorch_validation_metrics,
                "xgboost_uncalibrated": xgboost_calibration.uncalibrated_metrics,
                "xgboost_effective": xgboost_validation_metrics,
            },
            "test": {
                "selection_use": "none",
                "logistic_regression": logistic_test_metrics,
                "pytorch_mlp_uncalibrated": pytorch_test_uncalibrated_metrics,
                "pytorch_mlp_effective": pytorch_test_effective_metrics,
                "xgboost_uncalibrated": xgboost_test_uncalibrated_metrics,
                "xgboost_effective": xgboost_test_effective_metrics,
                "selected_model": selected_test_metrics,
            },
            "out_of_fold": oof_result.metrics,
        },
        "selection": {
            "model": selected_model,
            "source": "validation metrics only",
            "reason": selection_reason,
            "primary_metric": "validation log loss",
            "tie_breaker": "validation Brier score",
            "supporting_metric": "validation ROC-AUC",
            "test_metrics_used": False,
        },
        "out_of_fold": {
            "model": selected_model,
            "fold_count": OOF_FOLDS,
            "prediction_count": len(oof_result.predictions),
            "grouping": "match_id",
            "fixed_mlp_epochs": training_result.best_epoch,
            "fixed_xgboost_rounds": (
                xgboost_result.boosting_rounds if selected_model == "xgboost" else None
            ),
            "folds": oof_result.fold_summaries,
        },
        "production": production,
        "artifacts": {
            "evaluation_preprocessor": repository_relative_path(preprocessor_path),
            "evaluation_logistic_regression": repository_relative_path(logistic_path),
            "evaluation_pytorch_mlp": repository_relative_path(pytorch_path),
            "evaluation_mlp_history": repository_relative_path(history_path),
            "evaluation_xgboost": repository_relative_path(xgboost_path),
            "evaluation_xgboost_preprocessor": repository_relative_path(
                xgboost_preprocessor_path
            ),
            "test_calibration_plot": repository_relative_path(calibration_path),
            "test_predictions": repository_relative_path(test_predictions_path),
            "oof_predictions": repository_relative_path(oof_predictions_path),
            "production_preprocessor": production["preprocessor"],
            "production_model": production["model_path"],
        },
        "downstream": {
            "player_profiles_regenerated": player_profiles_regenerated,
            "player_similarities_regenerated": False,
            "downstream_pipeline_note": (
                "V3 player intelligence and similarity are separate, ordered pipeline stages."
            ),
            "postgresql_reload_required": player_profiles_regenerated,
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Artifacts:")
    for artifact_path in (
        preprocessor_path,
        logistic_path,
        pytorch_path,
        metadata_path,
        calibration_path,
        history_path,
        xgboost_path,
        xgboost_preprocessor_path,
        test_predictions_path,
        oof_predictions_path,
        Path(production["preprocessor"]),
        Path(production["model_path"]),
    ):
        print(f"  {artifact_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--models-directory", type=Path, default=DEFAULT_MODELS_DIRECTORY)
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=DEFAULT_TEST_PREDICTIONS_PATH,
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=DEFAULT_OOF_PREDICTIONS_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_save(
        args.input,
        args.models_directory,
        args.test_predictions,
        args.oof_predictions,
    )


if __name__ == "__main__":
    main()
