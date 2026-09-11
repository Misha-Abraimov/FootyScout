"""Train, evaluate, OOF-score, and package the FootyScout V2.1 xG model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.linear_model import LogisticRegression

from analytics.player_shooting_profiles import run as build_profiles
from analytics.shot_features import TARGET_COLUMN, XG_MODEL_FEATURE_COLUMNS
from analytics.xg_model import (
    RANDOM_SEED,
    XGBOOST_CANDIDATES,
    CalibrationResult,
    XGBoostConfig,
    assess_calibration,
    build_xg_preprocessor,
    build_xgboost,
    config_dict,
    evaluate_probabilities,
    fit_fixed_model,
    grouped_folds,
    probabilities_from_margins,
    raw_margins,
    split_by_match,
)

ROOT = Path(__file__).resolve().parents[2]
SHOTS_PATH = ROOT / "data" / "processed" / "shots.parquet"
OOF_PATH = ROOT / "data" / "processed" / "shot_oof_predictions.parquet"
PRODUCT_PATH = ROOT / "data" / "processed" / "product_shot_predictions.parquet"
MODEL_PATH = ROOT / "models" / "xg_model.json"
LOGISTIC_PATH = ROOT / "models" / "xg_model.joblib"
PREPROCESSOR_PATH = ROOT / "models" / "xg_preprocessor.joblib"
METADATA_PATH = ROOT / "models" / "xg_model_metadata.json"


def _partition_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "match_count": int(frame["match_id"].nunique()),
        "shot_count": len(frame),
        "goal_count": int(frame[TARGET_COLUMN].sum()),
        "goal_rate": float(frame[TARGET_COLUMN].mean()),
        "competition_seasons": [
            {
                "competition_id": int(competition_id),
                "season_id": int(season_id),
                "shot_count": len(group),
            }
            for (competition_id, season_id), group in frame.groupby(
                ["competition_id", "season_id"], sort=True
            )
        ],
    }


def _calibration_dict(result: CalibrationResult) -> dict[str, Any]:
    return {
        "method": "temperature_scaling_on_raw_margin",
        "fit_split": "validation",
        "temperature": result.temperature,
        "retained": result.retained,
        "retention_rule": (
            "validation log loss improves >= 0.001; Brier improves >= 0.0002; "
            "ECE worsens by no more than 0.002"
        ),
        "reason": result.reason,
        "uncalibrated_validation_metrics": result.uncalibrated_metrics,
        "calibrated_validation_metrics": result.calibrated_metrics,
        "uncalibrated_validation_curve": result.uncalibrated_curve,
        "calibrated_validation_curve": result.calibrated_curve,
    }


def _predict_effective(
    model: LogisticRegression | Any,
    matrix: np.ndarray,
    calibration: CalibrationResult,
) -> np.ndarray:
    temperature = calibration.temperature if calibration.retained else 1.0
    return probabilities_from_margins(raw_margins(model, matrix), temperature)


def _select_family(metrics: dict[str, dict[str, float]]) -> tuple[str, str]:
    ordered = sorted(metrics, key=lambda name: (metrics[name]["log_loss"], metrics[name]["brier_score"], name))
    selected, runner_up = ordered[0], ordered[1]
    difference = metrics[runner_up]["log_loss"] - metrics[selected]["log_loss"]
    return selected, f"lowest validation log loss, {difference:.6f} below {runner_up}"


def _fit_oof(
    data: pd.DataFrame,
    model_name: str,
    config: XGBoostConfig,
    rounds: int,
    retain_calibration: bool,
) -> pd.DataFrame:
    output = data.copy()
    output["expected_goal"] = np.nan
    output["fold"] = 0
    for fold_number, (train_indices, heldout_indices) in enumerate(grouped_folds(data), start=1):
        fold_train = data.iloc[train_indices]
        heldout = data.iloc[heldout_indices]
        if set(fold_train["match_id"]).intersection(heldout["match_id"]):
            raise AssertionError(f"Fold {fold_number} has match leakage")
        temperature = 1.0
        if retain_calibration:
            inner = split_by_match(fold_train, random_seed=RANDOM_SEED + fold_number)
            model_train = pd.concat([inner.train, inner.test], ignore_index=True)
            calibration_frame = inner.validation
        else:
            model_train = fold_train
            calibration_frame = None
        preprocessor = build_xg_preprocessor()
        train_matrix = preprocessor.fit_transform(model_train[XG_MODEL_FEATURE_COLUMNS])
        heldout_matrix = preprocessor.transform(heldout[XG_MODEL_FEATURE_COLUMNS])
        model = fit_fixed_model(
            model_name,
            train_matrix,
            model_train[TARGET_COLUMN].to_numpy(dtype=int),
            config,
            rounds,
        )
        if calibration_frame is not None:
            calibration_matrix = preprocessor.transform(
                calibration_frame[XG_MODEL_FEATURE_COLUMNS]
            )
            fold_calibration = assess_calibration(
                raw_margins(model, calibration_matrix),
                calibration_frame[TARGET_COLUMN].to_numpy(dtype=int),
            )
            temperature = fold_calibration.temperature
        probabilities = probabilities_from_margins(raw_margins(model, heldout_matrix), temperature)
        output.loc[output.index[heldout_indices], "expected_goal"] = probabilities
        output.loc[output.index[heldout_indices], "fold"] = fold_number
    if output["expected_goal"].isna().any() or not output["expected_goal"].between(0, 1).all():
        raise AssertionError("OOF probabilities must be complete and within [0, 1]")
    if output["shot_id"].duplicated().any() or len(output) != len(data):
        raise AssertionError("OOF output must contain exactly one row per eligible shot")
    output["fold"] = output["fold"].astype(int)
    return output


def run(shots_path: Path = SHOTS_PATH) -> dict[str, Any]:
    shots = pd.read_parquet(shots_path)
    eligible = shots.loc[shots["model_eligible"]].reset_index(drop=True)
    splits = split_by_match(eligible)
    preprocessor = build_xg_preprocessor()
    train_matrix = preprocessor.fit_transform(splits.train[XG_MODEL_FEATURE_COLUMNS])
    validation_matrix = preprocessor.transform(splits.validation[XG_MODEL_FEATURE_COLUMNS])
    test_matrix = preprocessor.transform(splits.test[XG_MODEL_FEATURE_COLUMNS])
    y_train = splits.train[TARGET_COLUMN].to_numpy(dtype=int)
    y_validation = splits.validation[TARGET_COLUMN].to_numpy(dtype=int)
    y_test = splits.test[TARGET_COLUMN].to_numpy(dtype=int)

    logistic = LogisticRegression(max_iter=2_000, C=1.0, random_state=RANDOM_SEED).fit(
        train_matrix, y_train
    )
    validation_metrics: dict[str, dict[str, float]] = {
        "logistic_regression": evaluate_probabilities(
            y_validation, logistic.predict_proba(validation_matrix)[:, 1]
        )
    }
    candidates: list[dict[str, Any]] = []
    candidate_models: list[Any] = []
    for index, config in enumerate(XGBOOST_CANDIDATES, start=1):
        model = build_xgboost(config)
        model.fit(
            train_matrix,
            y_train,
            eval_set=[(validation_matrix, y_validation)],
            verbose=False,
        )
        rounds = int(model.best_iteration) + 1
        metrics = evaluate_probabilities(y_validation, model.predict_proba(validation_matrix)[:, 1])
        name = f"xgboost_candidate_{index}"
        validation_metrics[name] = metrics
        candidates.append(
            {"name": name, "parameters": config_dict(config), "boosting_rounds": rounds, "validation_metrics": metrics}
        )
        candidate_models.append(model)
    best_index = min(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["validation_metrics"]["log_loss"],
            candidates[index]["validation_metrics"]["brier_score"],
        ),
    )
    best_candidate = candidates[best_index]
    best_xgb = candidate_models[best_index]
    family_metrics = {
        "logistic_regression": validation_metrics["logistic_regression"],
        "xgboost": best_candidate["validation_metrics"],
    }
    selected, selection_reason = _select_family(family_metrics)
    selected_model = logistic if selected == "logistic_regression" else best_xgb
    selected_config = XGBOOST_CANDIDATES[best_index]
    selected_rounds = int(best_candidate["boosting_rounds"])
    calibration = assess_calibration(raw_margins(selected_model, validation_matrix), y_validation)
    validation_metrics["selected_effective"] = (
        calibration.calibrated_metrics if calibration.retained else calibration.uncalibrated_metrics
    )

    test_metrics = {
        "logistic_regression": evaluate_probabilities(
            y_test, logistic.predict_proba(test_matrix)[:, 1]
        ),
        "xgboost": evaluate_probabilities(
            y_test, best_xgb.predict_proba(test_matrix)[:, 1]
        ),
        "selected_effective": evaluate_probabilities(
            y_test, _predict_effective(selected_model, test_matrix, calibration)
        ),
    }

    oof = _fit_oof(
        eligible,
        selected,
        selected_config,
        selected_rounds,
        calibration.retained,
    )
    oof_metrics = evaluate_probabilities(
        oof[TARGET_COLUMN].to_numpy(dtype=int), oof["expected_goal"].to_numpy(dtype=float)
    )
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OOF_PATH, index=False)

    final_preprocessor = build_xg_preprocessor()
    full_matrix = final_preprocessor.fit_transform(eligible[XG_MODEL_FEATURE_COLUMNS])
    final_model = fit_fixed_model(
        selected,
        full_matrix,
        eligible[TARGET_COLUMN].to_numpy(dtype=int),
        selected_config,
        selected_rounds,
    )
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_preprocessor, PREPROCESSOR_PATH)
    if selected == "xgboost":
        final_model.save_model(MODEL_PATH)
    else:
        joblib.dump(final_model, LOGISTIC_PATH)

    product = shots.loc[shots["is_product_cohort"]].copy()
    product["expected_goal"] = np.nan
    product_eligible = product["model_eligible"]
    product_matrix = final_preprocessor.transform(
        product.loc[product_eligible, XG_MODEL_FEATURE_COLUMNS]
    )
    temperature = calibration.temperature if calibration.retained else 1.0
    product.loc[product_eligible, "expected_goal"] = probabilities_from_margins(
        raw_margins(final_model, product_matrix), temperature
    )
    product.to_parquet(PRODUCT_PATH, index=False)
    profiles = build_profiles(PRODUCT_PATH)

    feature_importance: list[dict[str, float | str]] = []
    if selected == "xgboost":
        booster_scores = final_model.get_booster().get_score(importance_type="gain")
        names = list(final_preprocessor.get_feature_names_out())
        rows = [
            (names[int(key[1:])], float(value))
            for key, value in booster_scores.items()
            if key.startswith("f") and int(key[1:]) < len(names)
        ]
        total_gain = sum(value for _, value in rows) or 1.0
        feature_importance = [
            {"feature": name, "gain": gain, "normalized_gain": gain / total_gain}
            for name, gain in sorted(rows, key=lambda row: row[1], reverse=True)
        ]

    corpus = [
        {
            "competition_id": int(competition_id),
            "season_id": int(season_id),
            "matches": int(group["match_id"].nunique()),
            "shots": len(group),
            "goals": int(group[TARGET_COLUMN].sum()),
        }
        for (competition_id, season_id), group in shots.groupby(
            ["competition_id", "season_id"], sort=True
        )
    ]
    metadata: dict[str, Any] = {
        "task_name": "Expected goals",
        "selected_model": selected,
        "selection": {
            "primary_metric": "validation_log_loss",
            "secondary_metrics": ["brier_score", "roc_auc", "expected_calibration_error", "accuracy"],
            "reason": selection_reason,
            "test_metrics_used": False,
        },
        "features": XG_MODEL_FEATURE_COLUMNS,
        "preprocessing": {
            "numeric": "median imputation then standard scaling",
            "boolean": "omitted StatsBomb flags normalized false; most-frequent fallback",
            "categorical": "Unknown fill and one-hot encoding with unseen categories ignored",
            "evaluation_fit_split": "training matches only",
            "oof_fit_policy": "fit independently inside each outer training fold",
            "encoded_feature_count": int(full_matrix.shape[1]),
        },
        "penalty_policy": "penalties and period-5 shootout shots are preserved but excluded from modeling and player aggregates",
        "dataset": {
            "matches": int(shots["match_id"].nunique()),
            "shots": len(shots),
            "goals": int(shots[TARGET_COLUMN].sum()),
            "penalties": int(shots["penalty"].sum()),
            "penalty_shootout_shots": int(shots["penalty_shootout"].sum()),
            "eligible_non_penalty_shots": len(eligible),
            "eligible_non_penalty_goals": int(eligible[TARGET_COLUMN].sum()),
            "corpus": corpus,
        },
        "split": {
            "method": "match-grouped deterministic approximately 80/10/10 split",
            "random_seed": RANDOM_SEED,
            "train": _partition_summary(splits.train),
            "validation": _partition_summary(splits.validation),
            "test": _partition_summary(splits.test),
            "train_match_ids": list(splits.train_match_ids),
            "validation_match_ids": list(splits.validation_match_ids),
            "test_match_ids": list(splits.test_match_ids),
        },
        "benchmarks": {
            "logistic_regression": {"parameters": {"C": 1.0, "max_iter": 2000}},
            "xgboost_candidates": candidates,
        },
        "selected_parameters": (
            config_dict(selected_config)
            if selected == "xgboost"
            else {"C": 1.0, "max_iter": 2000, "random_state": RANDOM_SEED}
        ),
        "selected_boosting_rounds": selected_rounds if selected == "xgboost" else None,
        "metrics": {
            "validation": validation_metrics,
            "untouched_test": test_metrics,
            "out_of_fold": oof_metrics,
        },
        "calibration": _calibration_dict(calibration),
        "oof": {
            "fold_count": 5,
            "prediction_count": len(oof),
            "unique_shots": int(oof["shot_id"].nunique()),
            "missing_predictions": int(oof["expected_goal"].isna().sum()),
            "duplicate_shots": int(oof["shot_id"].duplicated().sum()),
            "probability_min": float(oof["expected_goal"].min()),
            "probability_max": float(oof["expected_goal"].max()),
            "group_integrity": True,
        },
        "feature_importance": {
            "type": "gain",
            "interpretation": "model feature importance, not causal effects",
            "features": feature_importance,
        },
        "production": {
            "training_rows": len(eligible),
            "model_format": "XGBoost native JSON" if selected == "xgboost" else "joblib",
            "temperature": temperature,
            "temperature_source": "fixed validation decision",
            "evaluation_use": "none; full-data artifact is for future inference only",
            "xgboost_version": xgboost.__version__,
        },
        "product_cohort": {
            "shots_preserved": len(product),
            "shots_scored": int(product["expected_goal"].notna().sum()),
            "players_profiled": len(profiles),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"OOF predictions: {OOF_PATH}")
    print(f"Product shots: {PRODUCT_PATH}")
    print(f"Preprocessor: {PREPROCESSOR_PATH}")
    print(f"Model: {MODEL_PATH if selected == 'xgboost' else LOGISTIC_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=Path, default=SHOTS_PATH)
    args = parser.parse_args()
    run(args.shots.resolve())


if __name__ == "__main__":
    main()
