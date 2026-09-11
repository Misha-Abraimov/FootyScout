"""Leakage-safe grouped modeling utilities for FootyScout expected goals."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from analytics.shot_features import (
    TARGET_COLUMN,
    XG_BOOLEAN_FEATURE_COLUMNS,
    XG_CATEGORICAL_FEATURE_COLUMNS,
    XG_NUMERIC_FEATURE_COLUMNS,
    validate_xg_model_contract,
)

RANDOM_SEED = 42


@dataclass(frozen=True)
class XGBoostConfig:
    max_depth: int
    learning_rate: float
    n_estimators: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    early_stopping_rounds: int = 40


XGBOOST_CANDIDATES = (
    XGBoostConfig(2, 0.04, 900, 8.0, 0.85, 0.85, 0.0, 3.0),
    XGBoostConfig(3, 0.035, 1100, 10.0, 0.90, 0.90, 0.05, 4.0),
    XGBoostConfig(4, 0.025, 1400, 12.0, 0.90, 0.85, 0.10, 6.0),
)


@dataclass(frozen=True)
class MatchSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_match_ids: tuple[int, ...]
    validation_match_ids: tuple[int, ...]
    test_match_ids: tuple[int, ...]


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    retained: bool
    reason: str
    uncalibrated_metrics: dict[str, float]
    calibrated_metrics: dict[str, float]
    uncalibrated_curve: list[dict[str, float | int]]
    calibrated_curve: list[dict[str, float | int]]


def boolean_matrix(values: Any) -> np.ndarray:
    frame = pd.DataFrame(values).apply(pd.to_numeric, errors="coerce")
    return frame.to_numpy(dtype=np.float64, na_value=np.nan)


def build_xg_preprocessor() -> ColumnTransformer:
    """Build numeric, boolean, and categorical branches with unseen-category safety."""
    validate_xg_model_contract()
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
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
            ("numeric", numeric, XG_NUMERIC_FEATURE_COLUMNS),
            ("boolean", boolean, XG_BOOLEAN_FEATURE_COLUMNS),
            ("categorical", categorical, XG_CATEGORICAL_FEATURE_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def split_by_match(data: pd.DataFrame, random_seed: int = RANDOM_SEED) -> MatchSplits:
    """Deterministically partition whole matches approximately 80/10/10."""
    match_ids = sorted(int(value) for value in data["match_id"].dropna().unique())
    if len(match_ids) < 10:
        raise ValueError("At least 10 matches are required for grouped xG splitting")
    train_ids, remainder = train_test_split(
        match_ids, test_size=0.2, random_state=random_seed, shuffle=True
    )
    validation_ids, test_ids = train_test_split(
        remainder, test_size=0.5, random_state=random_seed, shuffle=True
    )
    split = MatchSplits(
        train=data[data["match_id"].isin(train_ids)].copy(),
        validation=data[data["match_id"].isin(validation_ids)].copy(),
        test=data[data["match_id"].isin(test_ids)].copy(),
        train_match_ids=tuple(sorted(train_ids)),
        validation_match_ids=tuple(sorted(validation_ids)),
        test_match_ids=tuple(sorted(test_ids)),
    )
    assert_disjoint_splits(split)
    if len(split.train) + len(split.validation) + len(split.test) != len(data):
        raise AssertionError("Every eligible shot must occur in exactly one fixed split")
    for name, frame in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if frame[TARGET_COLUMN].nunique() != 2:
            raise ValueError(f"{name} split must contain goals and non-goals")
    return split


def assert_disjoint_splits(split: MatchSplits) -> None:
    groups = [set(split.train_match_ids), set(split.validation_match_ids), set(split.test_match_ids)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise AssertionError("Train, validation, and test match IDs must be disjoint")


def grouped_folds(data: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = data["match_id"].to_numpy()
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    folds = list(splitter.split(data, groups=groups))
    heldout: list[int] = []
    for train_indices, heldout_indices in folds:
        if set(groups[train_indices]).intersection(groups[heldout_indices]):
            raise AssertionError("OOF train and held-out match IDs overlap")
        heldout.extend(int(index) for index in heldout_indices)
    if sorted(heldout) != list(range(len(data))):
        raise AssertionError("OOF folds must hold out every shot exactly once")
    return folds


def expected_calibration_error(
    target: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    return float(
        sum(
            row["weight"] * abs(row["observed_rate"] - row["mean_probability"])
            for row in calibration_curve(target, probabilities, bins)
        )
    )


def calibration_curve(
    target: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> list[dict[str, float | int]]:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.digitize(probabilities, boundaries[1:-1], right=True)
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            rows.append(
                {
                    "bin": index + 1,
                    "count": int(mask.sum()),
                    "weight": float(mask.mean()),
                    "mean_probability": float(probabilities[mask].mean()),
                    "observed_rate": float(target[mask].mean()),
                }
            )
    return rows


def evaluate_probabilities(target: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    binary = np.asarray(target, dtype=int)
    return {
        "roc_auc": float(roc_auc_score(binary, clipped)),
        "log_loss": float(log_loss(binary, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(binary, clipped)),
        "accuracy": float(accuracy_score(binary, clipped >= 0.5)),
        "expected_calibration_error": expected_calibration_error(binary, clipped),
    }


def probabilities_from_margins(margins: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.clip(np.asarray(margins, dtype=float) / temperature, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def fit_temperature(margins: np.ndarray, target: np.ndarray) -> float:
    """Minimize validation log loss over one positive scaling parameter."""
    result = minimize_scalar(
        lambda log_t: log_loss(
            target,
            probabilities_from_margins(margins, math.exp(float(log_t))),
            labels=[0, 1],
        ),
        bounds=(math.log(0.2), math.log(5.0)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError("Temperature scaling optimization failed")
    return float(math.exp(float(result.x)))


def assess_calibration(margins: np.ndarray, target: np.ndarray) -> CalibrationResult:
    """Retain scaling only for material validation improvement without ECE harm."""
    raw_probabilities = probabilities_from_margins(margins)
    temperature = fit_temperature(margins, target)
    scaled_probabilities = probabilities_from_margins(margins, temperature)
    raw = evaluate_probabilities(target, raw_probabilities)
    scaled = evaluate_probabilities(target, scaled_probabilities)
    retained = (
        scaled["log_loss"] <= raw["log_loss"] - 0.001
        and scaled["brier_score"] <= raw["brier_score"] - 0.0002
        and scaled["expected_calibration_error"] <= raw["expected_calibration_error"] + 0.002
    )
    reason = (
        "retained: validation log loss improved by at least 0.001 and Brier by at "
        "least 0.0002 without materially worsening ECE"
        if retained
        else "not retained: the conservative material-improvement rule was not satisfied"
    )
    return CalibrationResult(
        temperature=temperature,
        retained=retained,
        reason=reason,
        uncalibrated_metrics=raw,
        calibrated_metrics=scaled,
        uncalibrated_curve=calibration_curve(target, raw_probabilities),
        calibrated_curve=calibration_curve(target, scaled_probabilities),
    )


def build_xgboost(config: XGBoostConfig, *, rounds: int | None = None) -> XGBClassifier:
    parameters: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "n_estimators": rounds or config.n_estimators,
        "min_child_weight": config.min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "random_state": RANDOM_SEED,
        "n_jobs": 4,
    }
    if rounds is None:
        parameters["early_stopping_rounds"] = config.early_stopping_rounds
    return XGBClassifier(**parameters)


def fit_fixed_model(
    model_name: str,
    features: np.ndarray,
    target: np.ndarray,
    config: XGBoostConfig,
    rounds: int,
) -> LogisticRegression | XGBClassifier:
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2_000, C=1.0, random_state=RANDOM_SEED).fit(
            features, target
        )
    return build_xgboost(config, rounds=rounds).fit(features, target, verbose=False)


def raw_margins(model: LogisticRegression | XGBClassifier, features: np.ndarray) -> np.ndarray:
    if isinstance(model, LogisticRegression):
        return np.asarray(model.decision_function(features), dtype=float)
    return np.asarray(model.predict(features, output_margin=True), dtype=float)


def config_dict(config: XGBoostConfig) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        **asdict(config),
    }
