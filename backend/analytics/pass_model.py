"""Reusable modeling components for expected pass completion."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from analytics.pass_features import (
    KNOWN_LEAKAGE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

RANDOM_SEED = 42
BOOLEAN_FEATURE_COLUMNS = ["under_pressure", "progressive"]
CATEGORICAL_FEATURE_COLUMNS = [
    "pass_height",
    "body_part",
    "pass_type",
    "start_zone",
    "end_zone",
]
MODEL_NUMERIC_COLUMNS = list(NUMERIC_FEATURE_COLUMNS)
MODEL_CATEGORICAL_COLUMNS = list(CATEGORICAL_FEATURE_COLUMNS)
MODEL_BOOLEAN_COLUMNS = list(BOOLEAN_FEATURE_COLUMNS)

MODEL_ARCHITECTURE = [
    "Linear(input_dim, 64)",
    "ReLU",
    "Dropout(0.2)",
    "Linear(64, 32)",
    "ReLU",
    "Dropout(0.2)",
    "Linear(32, 1)",
]

XGBOOST_VERSION = xgboost.__version__


@dataclass(frozen=True)
class XGBoostConfig:
    max_depth: int
    learning_rate: float
    n_estimators: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float
    early_stopping_rounds: int = 50
    random_state: int = RANDOM_SEED
    n_jobs: int = 4


XGBOOST_CANDIDATES = (
    XGBoostConfig(3, 0.05, 1200, 5.0, 0.85, 0.85, 2.0, 0.05),
    XGBoostConfig(4, 0.05, 1200, 5.0, 0.85, 0.85, 3.0, 0.10),
    XGBoostConfig(5, 0.03, 1600, 8.0, 0.90, 0.90, 4.0, 0.10),
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
class PreprocessedSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_target: np.ndarray
    validation_target: np.ndarray
    test_target: np.ndarray


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 7
    min_delta: float = 1e-4
    random_seed: int = RANDOM_SEED


@dataclass(frozen=True)
class TrainingResult:
    model: PassCompletionMLP
    epochs_trained: int
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    history: list[dict[str, float | int]]


@dataclass(frozen=True)
class GroupedFold:
    """Row positions and match IDs for one grouped outer fold."""

    fold: int
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    train_match_ids: tuple[int, ...]
    heldout_match_ids: tuple[int, ...]


@dataclass(frozen=True)
class TemperatureScalingResult:
    """Validation-only temperature-scaling decision and diagnostics."""

    temperature: float
    retained: bool
    reason: str
    uncalibrated_metrics: dict[str, float]
    calibrated_metrics: dict[str, float]


@dataclass(frozen=True)
class XGBoostCandidateResult:
    model: XGBClassifier
    config: XGBoostConfig
    best_iteration: int
    boosting_rounds: int
    metrics: dict[str, float]


def set_deterministic_seeds(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_model_contract() -> None:
    """Assert that only approved pre-outcome fields enter preprocessing."""
    configured = MODEL_NUMERIC_COLUMNS + MODEL_BOOLEAN_COLUMNS + MODEL_CATEGORICAL_COLUMNS
    if set(configured) != set(MODEL_FEATURE_COLUMNS):
        missing = set(MODEL_FEATURE_COLUMNS).difference(configured)
        extra = set(configured).difference(MODEL_FEATURE_COLUMNS)
        raise ValueError(f"Model feature configuration mismatch; missing={missing}, extra={extra}")
    leaked = KNOWN_LEAKAGE_COLUMNS.intersection(configured)
    if leaked:
        raise ValueError(f"Leakage columns configured as model inputs: {sorted(leaked)}")


def boolean_to_float_matrix(values: Any) -> np.ndarray:
    """Represent booleans as 0/1 floats while preserving missing values as NaN."""
    frame = pd.DataFrame(values).apply(pd.to_numeric, errors="coerce")
    return frame.to_numpy(dtype=np.float64, na_value=np.nan)


def assert_disjoint_match_ids(splits: MatchSplits) -> None:
    """Raise when a match appears in more than one data split."""
    train_ids = set(splits.train_match_ids)
    validation_ids = set(splits.validation_match_ids)
    test_ids = set(splits.test_match_ids)
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Train, validation, and test match IDs must not overlap")


def split_by_match(data: pd.DataFrame, random_seed: int = RANDOM_SEED) -> MatchSplits:
    """Deterministically split whole matches approximately 80/10/10."""
    if "match_id" not in data.columns:
        raise ValueError("Feature data must contain match_id for grouped splitting")

    match_ids = sorted(int(value) for value in data["match_id"].dropna().unique())
    if len(match_ids) < 10:
        raise ValueError("At least 10 matches are required for an approximately 80/10/10 split")

    train_ids, holdout_ids = train_test_split(
        match_ids,
        test_size=0.2,
        random_state=random_seed,
        shuffle=True,
    )
    validation_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=0.5,
        random_state=random_seed,
        shuffle=True,
    )

    train_id_set = set(train_ids)
    validation_id_set = set(validation_ids)
    test_id_set = set(test_ids)
    splits = MatchSplits(
        train=data[data["match_id"].isin(train_id_set)].copy(),
        validation=data[data["match_id"].isin(validation_id_set)].copy(),
        test=data[data["match_id"].isin(test_id_set)].copy(),
        train_match_ids=tuple(sorted(train_id_set)),
        validation_match_ids=tuple(sorted(validation_id_set)),
        test_match_ids=tuple(sorted(test_id_set)),
    )
    assert_disjoint_match_ids(splits)
    if sum(len(frame) for frame in (splits.train, splits.validation, splits.test)) != len(data):
        raise AssertionError("Every pass must belong to exactly one match split")
    return splits


def grouped_match_folds(
    data: pd.DataFrame,
    n_splits: int = 5,
    random_seed: int = RANDOM_SEED,
) -> list[GroupedFold]:
    """Return deterministic shuffled folds that keep every match intact."""
    if "match_id" not in data.columns:
        raise ValueError("Feature data must contain match_id for grouped folds")
    if data["match_id"].isna().any():
        raise ValueError("match_id cannot be missing for grouped folds")
    match_count = int(data["match_id"].nunique())
    if not 2 <= n_splits <= match_count:
        raise ValueError("n_splits must be between 2 and the number of matches")

    groups = data["match_id"].to_numpy()
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds: list[GroupedFold] = []
    all_heldout_indices: list[int] = []
    for fold_number, (train_indices, heldout_indices) in enumerate(
        splitter.split(data, groups=groups),
        start=1,
    ):
        train_ids = tuple(sorted(int(value) for value in np.unique(groups[train_indices])))
        heldout_ids = tuple(sorted(int(value) for value in np.unique(groups[heldout_indices])))
        if set(train_ids).intersection(heldout_ids):
            raise AssertionError(f"Fold {fold_number} has overlapping train/held-out matches")
        folds.append(
            GroupedFold(
                fold=fold_number,
                train_indices=train_indices,
                heldout_indices=heldout_indices,
                train_match_ids=train_ids,
                heldout_match_ids=heldout_ids,
            )
        )
        all_heldout_indices.extend(int(index) for index in heldout_indices)

    if sorted(all_heldout_indices) != list(range(len(data))):
        raise AssertionError("Grouped folds must hold out every input row exactly once")
    return folds


def build_preprocessor() -> ColumnTransformer:
    """Create numeric, categorical, and boolean preprocessing branches."""
    validate_model_contract()
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    boolean_pipeline = Pipeline(
        [
            (
                "to_float",
                FunctionTransformer(boolean_to_float_matrix, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, MODEL_NUMERIC_COLUMNS),
            ("boolean", boolean_pipeline, MODEL_BOOLEAN_COLUMNS),
            ("categorical", categorical_pipeline, MODEL_CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    splits: MatchSplits,
) -> PreprocessedSplits:
    """Fit on training passes only, then transform validation and test passes."""
    validate_model_contract()
    train_matrix = preprocessor.fit_transform(splits.train[MODEL_FEATURE_COLUMNS])
    validation_matrix = preprocessor.transform(splits.validation[MODEL_FEATURE_COLUMNS])
    test_matrix = preprocessor.transform(splits.test[MODEL_FEATURE_COLUMNS])

    return PreprocessedSplits(
        train=np.asarray(train_matrix, dtype=np.float32),
        validation=np.asarray(validation_matrix, dtype=np.float32),
        test=np.asarray(test_matrix, dtype=np.float32),
        train_target=splits.train[TARGET_COLUMN].to_numpy(dtype=np.float32),
        validation_target=splits.validation[TARGET_COLUMN].to_numpy(dtype=np.float32),
        test_target=splits.test[TARGET_COLUMN].to_numpy(dtype=np.float32),
    )


def train_logistic_regression(
    matrix: PreprocessedSplits,
    random_seed: int = RANDOM_SEED,
) -> LogisticRegression:
    """Fit the interpretable Logistic Regression baseline."""
    model = LogisticRegression(max_iter=2_000, solver="lbfgs", random_state=random_seed)
    model.fit(matrix.train, matrix.train_target)
    return model


def fit_logistic_regression(
    features: np.ndarray,
    target: np.ndarray,
    random_seed: int = RANDOM_SEED,
) -> LogisticRegression:
    """Fit Logistic Regression from an already-fitted feature matrix."""
    model = LogisticRegression(max_iter=2_000, solver="lbfgs", random_state=random_seed)
    model.fit(features, target)
    return model


def build_xgboost_model(
    config: XGBoostConfig,
    *,
    boosting_rounds: int | None = None,
    random_seed: int | None = None,
    early_stopping: bool = True,
) -> XGBClassifier:
    """Construct the deterministic, regularized sklearn-compatible classifier."""
    parameters: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "n_estimators": boosting_rounds or config.n_estimators,
        "min_child_weight": config.min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_lambda": config.reg_lambda,
        "reg_alpha": config.reg_alpha,
        "random_state": config.random_state if random_seed is None else random_seed,
        "n_jobs": config.n_jobs,
    }
    if early_stopping:
        parameters["early_stopping_rounds"] = config.early_stopping_rounds
    return XGBClassifier(**parameters)


def train_xgboost_candidate(
    matrix: PreprocessedSplits,
    config: XGBoostConfig,
) -> XGBoostCandidateResult:
    """Fit one candidate with validation-only early stopping and evaluate it."""
    model = build_xgboost_model(config)
    model.fit(
        matrix.train,
        matrix.train_target,
        eval_set=[(matrix.validation, matrix.validation_target)],
        verbose=False,
    )
    best_iteration = int(model.best_iteration)
    probabilities = model.predict_proba(matrix.validation)[:, 1]
    return XGBoostCandidateResult(
        model=model,
        config=config,
        best_iteration=best_iteration,
        boosting_rounds=best_iteration + 1,
        metrics=evaluate_probabilities(matrix.validation_target, probabilities),
    )


def fit_xgboost_fixed_rounds(
    features: np.ndarray,
    target: np.ndarray,
    config: XGBoostConfig,
    boosting_rounds: int,
    random_seed: int = RANDOM_SEED,
) -> XGBClassifier:
    """Fit frozen XGBoost parameters without consulting held-out observations."""
    model = build_xgboost_model(
        config,
        boosting_rounds=boosting_rounds,
        random_seed=random_seed,
        early_stopping=False,
    )
    model.fit(features, target, verbose=False)
    return model


def predict_xgboost_margins(model: XGBClassifier, features: np.ndarray) -> np.ndarray:
    """Return raw margins suitable for temperature scaling before sigmoid."""
    return np.asarray(model.predict(features, output_margin=True), dtype=np.float32)


def predict_xgboost_probabilities(
    model: XGBClassifier,
    features: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    return probabilities_from_logits(predict_xgboost_margins(model, features), temperature)


class PassCompletionMLP(nn.Module):
    """Compact feed-forward network that returns one logit per pass."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _tensor_loader(
    features: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32, copy=False)),
        torch.from_numpy(target.astype(np.float32, copy=False)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def train_mlp(
    matrix: PreprocessedSplits,
    config: TrainingConfig | None = None,
    device: torch.device | None = None,
) -> TrainingResult:
    """Train with BCE logits loss, AdamW, mini-batches, and early stopping."""
    config = config or TrainingConfig()
    set_deterministic_seeds(config.random_seed)
    selected_device = device or torch.device("cpu")
    model = PassCompletionMLP(matrix.train.shape[1]).to(selected_device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _tensor_loader(
        matrix.train,
        matrix.train_target,
        config.batch_size,
        config.random_seed,
    )
    validation_features = torch.from_numpy(matrix.validation).to(selected_device)
    validation_target = torch.from_numpy(matrix.validation_target).to(selected_device)

    history: list[dict[str, float | int]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_features, batch_target in train_loader:
            batch_features = batch_features.to(selected_device)
            batch_target = batch_target.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features).squeeze(1)
            loss = criterion(logits, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_target)
            total_rows += len(batch_target)

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_features).squeeze(1)
            validation_loss = float(criterion(validation_logits, validation_target).item())
        training_loss = total_loss / total_rows
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:02d}/{config.max_epochs}: "
            f"train_loss={training_loss:.6f} val_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss - config.min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(torch.device("cpu"))
    model.eval()
    return TrainingResult(
        model=model,
        epochs_trained=len(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=len(history) < config.max_epochs,
        history=history,
    )


def train_mlp_fixed_epochs(
    features: np.ndarray,
    target: np.ndarray,
    epochs: int,
    config: TrainingConfig | None = None,
    random_seed: int | None = None,
    progress_label: str | None = None,
) -> TrainingResult:
    """Train an MLP for a preselected epoch count without using held-out data."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    config = config or TrainingConfig()
    seed = config.random_seed if random_seed is None else random_seed
    set_deterministic_seeds(seed)
    model = PassCompletionMLP(features.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = _tensor_loader(features, target, config.batch_size, seed)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_features, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features).squeeze(1)
            loss = criterion(logits, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_target)
            total_rows += len(batch_target)
        training_loss = total_loss / total_rows
        history.append({"epoch": epoch, "training_loss": training_loss})
        if progress_label:
            print(f"{progress_label} epoch {epoch:02d}/{epochs}: train_loss={training_loss:.6f}")

    model.eval()
    return TrainingResult(
        model=model,
        epochs_trained=epochs,
        best_epoch=epochs,
        best_validation_loss=math.nan,
        stopped_early=False,
        history=history,
    )


def predict_mlp_logits(model: PassCompletionMLP, features: np.ndarray) -> np.ndarray:
    """Return raw MLP logits without applying a sigmoid."""
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(features.astype(np.float32, copy=False))
        logits = model(inputs).squeeze(1)
    return logits.cpu().numpy()


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling and sigmoid to logits."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    scaled = torch.from_numpy(np.asarray(logits, dtype=np.float32)) / temperature
    return torch.sigmoid(scaled).numpy()


def predict_mlp_probabilities(
    model: PassCompletionMLP,
    features: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert model logits into optionally temperature-scaled probabilities."""
    return probabilities_from_logits(predict_mlp_logits(model, features), temperature)


def expected_calibration_error(
    target: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Calculate weighted absolute calibration error over equal-width bins."""
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.digitize(probabilities, boundaries[1:-1], right=True)
    error = 0.0
    for bin_index in range(bins):
        mask = assignments == bin_index
        if mask.any():
            weight = float(mask.mean())
            error += weight * abs(float(target[mask].mean() - probabilities[mask].mean()))
    return error


def evaluate_probabilities(target: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Evaluate discrimination, probability quality, calibration, and accuracy."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    binary_target = np.asarray(target, dtype=int)
    predictions = (clipped >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(binary_target, clipped)),
        "log_loss": float(log_loss(binary_target, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(binary_target, clipped)),
        "accuracy": float(accuracy_score(binary_target, predictions)),
        "expected_calibration_error": expected_calibration_error(binary_target, clipped),
    }


def fit_temperature(logits: np.ndarray, target: np.ndarray) -> float:
    """Fit one positive temperature by minimizing validation BCE only."""
    logit_tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    target_tensor = torch.from_numpy(np.asarray(target, dtype=np.float32))
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float32))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature)
        loss = criterion(logit_tensor / temperature, target_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature.detach()).clamp(0.05, 20.0).item())
    if not math.isfinite(temperature):
        raise RuntimeError("Temperature optimization produced a non-finite value")
    return temperature


def evaluate_temperature_scaling(
    validation_logits: np.ndarray,
    validation_target: np.ndarray,
    tolerance: float = 1e-6,
) -> TemperatureScalingResult:
    """Fit on validation and retain scaling only if all probability metrics improve."""
    uncalibrated = evaluate_probabilities(
        validation_target,
        probabilities_from_logits(validation_logits),
    )
    temperature = fit_temperature(validation_logits, validation_target)
    calibrated = evaluate_probabilities(
        validation_target,
        probabilities_from_logits(validation_logits, temperature),
    )
    improved_log_loss = calibrated["log_loss"] < uncalibrated["log_loss"] - tolerance
    improved_brier = calibrated["brier_score"] < uncalibrated["brier_score"] - tolerance
    ece_not_worse = (
        calibrated["expected_calibration_error"]
        <= uncalibrated["expected_calibration_error"] + tolerance
    )
    retained = improved_log_loss and improved_brier and ece_not_worse
    if retained:
        reason = "validation log loss and Brier score improved without worsening ECE"
    else:
        reason = (
            "not retained because validation log loss, Brier score, and ECE did not "
            "jointly satisfy the conservative improvement rule"
        )
    return TemperatureScalingResult(
        temperature=temperature,
        retained=retained,
        reason=reason,
        uncalibrated_metrics=uncalibrated,
        calibrated_metrics=calibrated,
    )


def select_model(
    logistic_metrics: dict[str, float],
    pytorch_metrics: dict[str, float],
    xgboost_metrics: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Select candidate families using validation log loss, then Brier score."""
    tolerance = 1e-6
    candidates = {
        "logistic_regression": logistic_metrics,
        "pytorch_mlp": pytorch_metrics,
    }
    if xgboost_metrics is not None:
        candidates["xgboost"] = xgboost_metrics
    ordered = sorted(
        candidates,
        key=lambda name: (candidates[name]["log_loss"], candidates[name]["brier_score"], name),
    )
    selected = ordered[0]
    runner_up = ordered[1]
    difference = candidates[runner_up]["log_loss"] - candidates[selected]["log_loss"]
    if difference > tolerance:
        reason = f"lower validation log loss by {difference:.6f} versus {runner_up}"
    else:
        selected = min(ordered[:2], key=lambda name: (candidates[name]["brier_score"], name))
        reason = "lower validation Brier score after effectively tied validation log loss"
    return selected, reason


def xgboost_config_dict(config: XGBoostConfig) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        **asdict(config),
    }


def save_xgboost_model(model: XGBClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def load_xgboost_model(path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model


def xgboost_gain_importance(
    model: XGBClassifier,
    feature_names: list[str],
) -> list[dict[str, float | str]]:
    """Return normalized gain importance; association is not causation."""
    raw = model.get_booster().get_score(importance_type="gain")
    mapped: list[tuple[str, float]] = []
    for key, value in raw.items():
        index = int(key[1:]) if key.startswith("f") and key[1:].isdigit() else None
        name = feature_names[index] if index is not None and index < len(feature_names) else key
        mapped.append((name, float(value)))
    total = sum(value for _, value in mapped)
    return [
        {"feature": name, "gain": value, "normalized_gain": value / total if total else 0.0}
        for name, value in sorted(mapped, key=lambda item: (-item[1], item[0]))
    ]


def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path) -> ColumnTransformer:
    return joblib.load(path)


def save_logistic_model(model: LogisticRegression, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_logistic_model(path: Path) -> LogisticRegression:
    return joblib.load(path)


def save_torch_model(
    model: PassCompletionMLP,
    path: Path,
    temperature: float = 1.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": model.input_dim,
            "architecture": MODEL_ARCHITECTURE,
            "temperature": temperature,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_torch_model(path: Path) -> PassCompletionMLP:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = PassCompletionMLP(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_torch_temperature(path: Path) -> float:
    """Load the saved calibration temperature, defaulting legacy artifacts to one."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return float(checkpoint.get("temperature", 1.0))


def save_training_history(result: TrainingResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epochs_trained": result.epochs_trained,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "stopped_early": result.stopped_early,
        "history": result.history,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def training_config_dict(config: TrainingConfig) -> dict[str, Any]:
    return asdict(config)
