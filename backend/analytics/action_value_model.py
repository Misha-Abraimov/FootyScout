"""Grouped, leakage-safe models for continuous attacking possession value."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from analytics.action_value_features import (
    ACTION_VALUE_BOOLEAN_FEATURE_COLUMNS,
    ACTION_VALUE_CATEGORICAL_FEATURE_COLUMNS,
    ACTION_VALUE_FEATURE_COLUMNS,
    ACTION_VALUE_NUMERIC_FEATURE_COLUMNS,
    validate_feature_contract,
)

RANDOM_SEED = 42


def boolean_matrix(values: Any) -> np.ndarray:
    frame = pd.DataFrame(values).apply(pd.to_numeric, errors="coerce")
    return frame.to_numpy(dtype=np.float64, na_value=np.nan)


@dataclass(frozen=True)
class RegressorConfig:
    objective: str
    max_depth: int
    learning_rate: float
    n_estimators: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    tweedie_variance_power: float | None = None
    early_stopping_rounds: int = 30


REGRESSOR_CANDIDATES = (
    RegressorConfig("reg:squarederror", 2, 0.04, 500, 20.0, 0.90, 0.90, 0.0, 4.0),
    RegressorConfig("reg:pseudohubererror", 3, 0.035, 600, 20.0, 0.90, 0.85, 0.05, 5.0),
    RegressorConfig("reg:tweedie", 3, 0.035, 600, 20.0, 0.90, 0.85, 0.05, 5.0, 1.3),
)


def build_preprocessor() -> ColumnTransformer:
    validate_feature_contract()
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    boolean = Pipeline(
        [
            (
                "to_float",
                FunctionTransformer(boolean_matrix, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, ACTION_VALUE_NUMERIC_FEATURE_COLUMNS),
            ("boolean", boolean, ACTION_VALUE_BOOLEAN_FEATURE_COLUMNS),
            ("categorical", categorical, ACTION_VALUE_CATEGORICAL_FEATURE_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def config_dict(config: RegressorConfig) -> dict[str, Any]:
    return asdict(config)


def build_regressor(config: RegressorConfig, rounds: int | None = None) -> XGBRegressor:
    parameters: dict[str, Any] = {
        "objective": config.objective,
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": 4,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "n_estimators": rounds or config.n_estimators,
        "min_child_weight": config.min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
    }
    if config.tweedie_variance_power is not None:
        parameters["tweedie_variance_power"] = config.tweedie_variance_power
    if rounds is None:
        parameters["early_stopping_rounds"] = config.early_stopping_rounds
    return XGBRegressor(**parameters)


def nonnegative(values: np.ndarray) -> np.ndarray:
    """State value is non-negative; floor regression noise at zero without an upper cap."""
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("State-value predictions must be finite")
    return np.maximum(result, 0.0)


def evaluate_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(target, dtype=float)
    predicted = nonnegative(prediction)
    positive = actual > 0
    correlation = spearmanr(actual, predicted).statistic
    metrics: dict[str, Any] = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "positive_target_mae": float(mean_absolute_error(actual[positive], predicted[positive])),
        "positive_target_rmse": float(mean_squared_error(actual[positive], predicted[positive]) ** 0.5),
        "prediction_min": float(predicted.min()),
        "prediction_max": float(predicted.max()),
    }
    buckets = [
        ("zero", actual == 0),
        ("0_to_0.05", (actual > 0) & (actual <= 0.05)),
        ("0.05_to_0.10", (actual > 0.05) & (actual <= 0.10)),
        ("0.10_to_0.20", (actual > 0.10) & (actual <= 0.20)),
        ("above_0.20", actual > 0.20),
    ]
    metrics["target_buckets"] = [
        {
            "bucket": name,
            "count": int(mask.sum()),
            "mae": float(mean_absolute_error(actual[mask], predicted[mask])),
            "rmse": float(mean_squared_error(actual[mask], predicted[mask]) ** 0.5),
            "mean_actual": float(actual[mask].mean()),
            "mean_predicted": float(predicted[mask].mean()),
        }
        for name, mask in buckets
        if mask.any()
    ]
    return metrics


def fit_hurdle(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray | None = None,
    validation_y: np.ndarray | None = None,
    *,
    classifier_rounds: int | None = None,
    regressor_rounds: int | None = None,
) -> tuple[XGBClassifier, XGBRegressor, int, int]:
    classifier_parameters: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": 4,
        "max_depth": 3,
        "learning_rate": 0.04,
        "n_estimators": classifier_rounds or 450,
        "min_child_weight": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 5.0,
    }
    regressor_parameters: dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": 4,
        "max_depth": 2,
        "learning_rate": 0.04,
        "n_estimators": regressor_rounds or 450,
        "min_child_weight": 12,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 5.0,
    }
    fit_classifier: dict[str, Any] = {"verbose": False}
    fit_regressor: dict[str, Any] = {"verbose": False}
    if classifier_rounds is None and validation_x is not None and validation_y is not None:
        classifier_parameters["early_stopping_rounds"] = 30
        fit_classifier["eval_set"] = [(validation_x, validation_y > 0)]
        regressor_parameters["early_stopping_rounds"] = 30
        positive_validation = validation_y > 0
        fit_regressor["eval_set"] = [
            (validation_x[positive_validation], validation_y[positive_validation])
        ]
    classifier = XGBClassifier(**classifier_parameters).fit(
        train_x, train_y > 0, **fit_classifier
    )
    positive = train_y > 0
    regressor = XGBRegressor(**regressor_parameters).fit(
        train_x[positive], train_y[positive], **fit_regressor
    )
    classifier_used = classifier_rounds or int(classifier.best_iteration) + 1
    regressor_used = regressor_rounds or int(regressor.best_iteration) + 1
    return classifier, regressor, classifier_used, regressor_used


def predict_hurdle(
    classifier: XGBClassifier, regressor: XGBRegressor, features: np.ndarray
) -> np.ndarray:
    probability = classifier.predict_proba(features)[:, 1]
    positive_value = nonnegative(regressor.predict(features))
    return probability * positive_value


def grouped_folds(states: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = states["match_id"].to_numpy()
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    folds = list(splitter.split(states, groups=groups))
    heldout: list[int] = []
    for train, test in folds:
        if set(groups[train]).intersection(groups[test]):
            raise AssertionError("OOF match IDs overlap")
        heldout.extend(test.tolist())
    if sorted(heldout) != list(range(len(states))):
        raise AssertionError("Each state must be held out exactly once")
    return folds


def feature_frame(states: pd.DataFrame) -> pd.DataFrame:
    return states[ACTION_VALUE_FEATURE_COLUMNS]
