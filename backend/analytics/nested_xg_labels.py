"""Strictly nested FootyScout xG labels for outer action-value evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.shot_features import TARGET_COLUMN as SHOT_TARGET_COLUMN
from analytics.shot_features import XG_MODEL_FEATURE_COLUMNS
from analytics.xg_model import (
    XGBoostConfig,
    build_xg_preprocessor,
    fit_fixed_model,
    grouped_folds,
    probabilities_from_margins,
    raw_margins,
)

STATE_TARGET_COLUMN = "future_oof_xg_same_possession"


@dataclass(frozen=True)
class FrozenXGSpec:
    model_name: str
    config: XGBoostConfig
    rounds: int
    retained_calibration: bool
    temperature: float


def load_frozen_xg_spec(metadata_path: Path) -> FrozenXGSpec:
    """Load the frozen V2.1 choice without selecting or tuning anything."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parameters = metadata["selected_parameters"]
    config = XGBoostConfig(
        max_depth=int(parameters["max_depth"]),
        learning_rate=float(parameters["learning_rate"]),
        n_estimators=int(parameters["n_estimators"]),
        min_child_weight=float(parameters["min_child_weight"]),
        subsample=float(parameters["subsample"]),
        colsample_bytree=float(parameters["colsample_bytree"]),
        reg_alpha=float(parameters["reg_alpha"]),
        reg_lambda=float(parameters["reg_lambda"]),
        early_stopping_rounds=int(parameters["early_stopping_rounds"]),
    )
    return FrozenXGSpec(
        model_name=str(metadata["selected_model"]),
        config=config,
        rounds=int(metadata["selected_boosting_rounds"]),
        retained_calibration=bool(metadata["calibration"]["retained"]),
        temperature=float(metadata["calibration"]["temperature"]),
    )


def _match_hash(match_ids: set[int]) -> str:
    payload = ",".join(str(value) for value in sorted(match_ids))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _predict(
    train: pd.DataFrame,
    prediction: pd.DataFrame,
    spec: FrozenXGSpec,
) -> np.ndarray:
    if set(train["match_id"]).intersection(prediction["match_id"]):
        raise AssertionError("Nested xG train and prediction matches overlap")
    preprocessor = build_xg_preprocessor()
    train_matrix = preprocessor.fit_transform(train[XG_MODEL_FEATURE_COLUMNS])
    prediction_matrix = preprocessor.transform(prediction[XG_MODEL_FEATURE_COLUMNS])
    model = fit_fixed_model(
        spec.model_name,
        train_matrix,
        train[SHOT_TARGET_COLUMN].to_numpy(dtype=int),
        spec.config,
        spec.rounds,
    )
    temperature = spec.temperature if spec.retained_calibration else 1.0
    return probabilities_from_margins(
        raw_margins(model, prediction_matrix),
        temperature,
    )


def nested_xg_predictions(
    shots: pd.DataFrame,
    outer_train_match_ids: set[int],
    outer_heldout_match_ids: set[int],
    spec: FrozenXGSpec,
    *,
    outer_name: str,
    cache_path: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Cross-fit outer-train xG and predict outer holdout from outer train only."""
    if outer_train_match_ids & outer_heldout_match_ids:
        raise AssertionError("Outer train and held-out match IDs overlap")
    eligible = shots.loc[shots["model_eligible"]].reset_index(drop=True)
    expected_ids = outer_train_match_ids | outer_heldout_match_ids
    available_ids = set(eligible["match_id"].astype(int))
    if not expected_ids.issubset(available_ids):
        raise ValueError("Nested xG split contains matches outside the shot corpus")

    manifest_path = cache_path.with_suffix(".json")
    expected_manifest = {
        "outer_name": outer_name,
        "outer_train_match_hash": _match_hash(outer_train_match_ids),
        "outer_heldout_match_hash": _match_hash(outer_heldout_match_ids),
        "outer_train_matches": len(outer_train_match_ids),
        "outer_heldout_matches": len(outer_heldout_match_ids),
        "xg_model": spec.model_name,
        "xg_rounds": spec.rounds,
        "calibration_retained": spec.retained_calibration,
        "temperature_used": spec.temperature if spec.retained_calibration else 1.0,
        "heldout_outcomes_used_for_training": False,
    }
    if not force and cache_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected_manifest.items()):
            cached = pd.read_parquet(cache_path)
            expected_shots = eligible["match_id"].isin(expected_ids).sum()
            if len(cached) == expected_shots and not cached["shot_id"].duplicated().any():
                return cached

    outer_train = eligible.loc[eligible["match_id"].isin(outer_train_match_ids)].copy()
    outer_heldout = eligible.loc[eligible["match_id"].isin(outer_heldout_match_ids)].copy()
    train_output = outer_train[["shot_id", "match_id"]].copy()
    train_output["expected_goal"] = np.nan
    train_output["outer_role"] = "train_inner_oof"
    train_output["inner_fold"] = 0
    for inner_fold, (train_indices, heldout_indices) in enumerate(
        grouped_folds(outer_train), start=1
    ):
        inner_train = outer_train.iloc[train_indices]
        inner_heldout = outer_train.iloc[heldout_indices]
        if set(inner_train["match_id"]).intersection(inner_heldout["match_id"]):
            raise AssertionError("Inner xG match leakage")
        prediction = _predict(inner_train, inner_heldout, spec)
        train_output.loc[train_output.index[heldout_indices], "expected_goal"] = prediction
        train_output.loc[train_output.index[heldout_indices], "inner_fold"] = inner_fold

    heldout_output = outer_heldout[["shot_id", "match_id"]].copy()
    heldout_output["expected_goal"] = _predict(outer_train, outer_heldout, spec)
    heldout_output["outer_role"] = "outer_heldout"
    heldout_output["inner_fold"] = 0
    output = pd.concat([train_output, heldout_output], ignore_index=True)
    if output["shot_id"].duplicated().any() or output["expected_goal"].isna().any():
        raise AssertionError("Nested xG output must contain one prediction per shot")
    if not output["expected_goal"].between(0, 1).all():
        raise AssertionError("Nested xG probabilities must remain within [0, 1]")
    output["inner_fold"] = output["inner_fold"].astype(int)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    output.to_parquet(temporary, index=False)
    temporary.replace(cache_path)
    manifest = {
        **expected_manifest,
        "outer_train_shots": len(train_output),
        "outer_heldout_shots": len(heldout_output),
        "prediction_count": len(output),
        "inner_fold_count": 5,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def apply_nested_state_targets(
    states: pd.DataFrame,
    shot_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild future-possession xG targets from one outer split's nested labels."""
    if shot_predictions["shot_id"].duplicated().any():
        raise ValueError("Nested shot predictions must be unique")
    result = states.copy().sort_values(
        ["match_id", "possession_id", "event_index"], kind="stable"
    )
    contributing = shot_predictions.loc[
        shot_predictions["shot_id"].isin(set(result["event_id"]))
    ]
    shot_values = contributing.set_index("shot_id")["expected_goal"]
    mapped = result["event_id"].map(shot_values)
    if mapped.notna().sum() != len(contributing):
        raise ValueError("Every nested xG shot must map to exactly one eligible state")
    result["nested_oof_shot_xg"] = mapped.fillna(0.0).astype(float)
    result[STATE_TARGET_COLUMN] = result.groupby(
        ["match_id", "possession_id"], sort=False
    )["nested_oof_shot_xg"].transform(lambda values: values.iloc[::-1].cumsum().iloc[::-1])
    return result.sort_index()
