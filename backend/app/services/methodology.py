"""Deployment-safe model metadata services shared by REST and AI tools."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.runtime_metadata import (
    ACTION_VALUE_MODEL_METADATA_PATH,
    PASS_MODEL_METADATA_PATH,
    XG_MODEL_METADATA_PATH,
)
from app.schemas import (
    ActionValueModelInfoResponse,
    DatasetSummary,
    ModelInfoResponse,
    ModelMethodology,
    ModelMetrics,
    ModelSelectionSummary,
    PreprocessingSummary,
    ProductionModelSummary,
    SplitSummary,
    TemperatureScalingSummary,
    XGBoostSummary,
    XGModelInfoResponse,
)


class MethodologyUnavailableError(RuntimeError):
    """The configured metadata file could not be read."""


class MethodologyMalformedError(RuntimeError):
    """The configured metadata file did not match its governed contract."""


def _metric_map(raw_metrics: dict[str, Any]) -> dict[str, ModelMetrics]:
    return {
        name: ModelMetrics.model_validate(values)
        for name, values in raw_metrics.items()
        if isinstance(values, dict)
    }


def build_pass_model_info(raw: dict[str, Any]) -> ModelInfoResponse:
    preprocessing = raw["preprocessing"]
    metrics = raw["metrics"]
    mlp_calibration = raw["temperature_scaling"]
    production = raw["production"]
    oof = raw["out_of_fold"]
    xgboost = raw["xgboost"]
    selected_model = raw["selection"]["model"]
    calibration = xgboost["calibration"] if selected_model == "xgboost" else mlp_calibration
    architecture = raw["pytorch"]["architecture"]
    if selected_model == "xgboost":
        architecture = [
            f"XGBClassifier ({xgboost['boosting_rounds']} boosting rounds)",
            f"max_depth={xgboost['selected_parameters']['max_depth']}",
            f"learning_rate={xgboost['selected_parameters']['learning_rate']}",
            "objective=binary:logistic · tree_method=hist",
        ]
    return ModelInfoResponse(
        task_name="Expected pass completion",
        task_description=(
            "Supervised binary classification of whether a pass is completed, "
            "returning an expected-completion probability from pre-outcome features."
        ),
        selected_model=selected_model,
        models_evaluated=raw["models_evaluated"],
        architecture=architecture,
        feature_columns=raw["feature_columns"],
        preprocessing=PreprocessingSummary(
            evaluation_fit_split=preprocessing["evaluation_fit_split"],
            oof_fit_policy=preprocessing["oof_fit_policy"],
            numeric=preprocessing["numeric"],
            boolean=preprocessing["boolean"],
            categorical=preprocessing["categorical"],
            encoded_feature_count=raw["evaluation_encoded_feature_count"],
        ),
        dataset=DatasetSummary.model_validate(raw["dataset"]),
        split_methodology=SplitSummary.model_validate(raw["split"]),
        selection=ModelSelectionSummary.model_validate(raw["selection"]),
        validation_metrics=_metric_map(metrics["validation"]),
        untouched_test_metrics=_metric_map(metrics["test"]),
        out_of_fold_metrics=ModelMetrics.model_validate(metrics["out_of_fold"]),
        calibration=TemperatureScalingSummary(
            method=calibration.get("method", "temperature_scaling"),
            fit_split=calibration["fit_split"],
            temperature=calibration["temperature"],
            retained=calibration["retained"],
            reason=calibration["reason"],
            uncalibrated_validation_metrics=calibration["uncalibrated_validation_metrics"],
            calibrated_validation_metrics=calibration["calibrated_validation_metrics"],
        ),
        xgboost=XGBoostSummary(
            version=xgboost["version"],
            random_seed=xgboost["random_seed"],
            selection_metric=xgboost["selection_metric"],
            selected_parameters=xgboost["selected_parameters"],
            best_iteration=xgboost["best_iteration"],
            boosting_rounds=xgboost["boosting_rounds"],
            calibration=TemperatureScalingSummary(
                method=xgboost["calibration"]["method"],
                fit_split=xgboost["calibration"]["fit_split"],
                temperature=xgboost["calibration"]["temperature"],
                retained=xgboost["calibration"]["retained"],
                reason=xgboost["calibration"]["reason"],
                uncalibrated_validation_metrics=xgboost["calibration"][
                    "uncalibrated_validation_metrics"
                ],
                calibrated_validation_metrics=xgboost["calibration"][
                    "calibrated_validation_metrics"
                ],
            ),
            feature_importance_type=xgboost["feature_importance"]["importance_type"],
            feature_importance_interpretation=xgboost["feature_importance"]["interpretation"],
            feature_importance=xgboost["feature_importance"]["features"][:10],
        ),
        oof_fold_count=oof["fold_count"],
        production=ProductionModelSummary(
            model=production["model"],
            training_rows=production["training_rows"],
            encoded_feature_count=production["encoded_feature_count"],
            epochs=production.get("epochs"),
            boosting_rounds=production.get("boosting_rounds"),
            temperature=production["temperature"],
            temperature_source=production["temperature_source"],
            evaluation_use=production["evaluation_use"],
        ),
        methodology=ModelMethodology(
            model_selection="Model selection was performed using validation data.",
            test_set="The test data was not used for model selection.",
            player_profiles="Player profiles use grouped out-of-fold predictions.",
            production_model=(
                "The production full-data model is for future inference and is not "
                "used for evaluation claims."
            ),
        ),
    )


def build_xg_model_info(raw: dict[str, Any]) -> XGModelInfoResponse:
    split = raw["split"]
    safe_split = {
        "method": split["method"],
        "random_seed": split["random_seed"],
        "train": split["train"],
        "validation": split["validation"],
        "test": split["test"],
    }
    return XGModelInfoResponse.model_validate(
        {
            "task_name": raw["task_name"],
            "selected_model": raw["selected_model"],
            "feature_columns": raw["features"],
            "dataset": raw["dataset"],
            "preprocessing": raw["preprocessing"],
            "split_methodology": safe_split,
            "selection": raw["selection"],
            "validation_metrics": raw["metrics"]["validation"],
            "untouched_test_metrics": raw["metrics"]["untouched_test"],
            "out_of_fold_metrics": raw["metrics"]["out_of_fold"],
            "calibration": raw["calibration"],
            "selected_parameters": raw["selected_parameters"],
            "selected_boosting_rounds": raw["selected_boosting_rounds"],
            "oof": raw["oof"],
            "penalty_policy": raw["penalty_policy"],
            "feature_importance": raw["feature_importance"]["features"][:10],
        }
    )


def build_action_value_info(raw: dict[str, Any]) -> ActionValueModelInfoResponse:
    split = raw["split"]
    safe_split = {
        "method": split["method"],
        "train_matches": len(split["train_match_ids"]),
        "validation_matches": len(split["validation_match_ids"]),
        "test_matches": len(split["test_match_ids"]),
        "train_states": split["train_states"],
        "validation_states": split["validation_states"],
        "test_states": split["test_states"],
    }
    return ActionValueModelInfoResponse.model_validate(
        {
            "task_name": raw["task_name"],
            "target": raw["target"],
            "target_interpretation": raw["target_interpretation"],
            "state_convention": raw["state_convention"],
            "selected_horizon": raw["selected_horizon"],
            "feature_columns": raw["features"],
            "leakage_protection": raw["leakage_protection"],
            "preprocessing": raw["preprocessing"],
            "split_methodology": safe_split,
            "nested_cross_fitting": raw["nested_cross_fitting"],
            "training_corpus": raw["training_corpus"],
            "target_distribution": raw["target_distribution"],
            "baseline": raw["baseline"],
            "candidates": raw["candidates"],
            "hurdle": raw["hurdle"],
            "selection": raw["selection"],
            "validation_metrics": raw["validation_metrics"],
            "untouched_test_metrics": raw["untouched_test_metrics"],
            "out_of_fold_metrics": raw["out_of_fold_metrics"],
            "oof": raw["oof"],
            "transition_rules": raw["transition_rules"],
            "limitations": raw["limitations"],
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MethodologyUnavailableError(str(path)) from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise MethodologyMalformedError(str(path)) from exc
    if not isinstance(raw, dict):
        raise MethodologyMalformedError(str(path))
    return raw


def get_pass_model_info(
    path: Path = PASS_MODEL_METADATA_PATH,
) -> ModelInfoResponse:
    try:
        return build_pass_model_info(_load_json(path))
    except MethodologyUnavailableError:
        raise
    except (AttributeError, KeyError, TypeError, ValidationError) as exc:
        raise MethodologyMalformedError(str(path)) from exc


def get_xg_model_info(path: Path = XG_MODEL_METADATA_PATH) -> XGModelInfoResponse:
    try:
        return build_xg_model_info(_load_json(path))
    except MethodologyUnavailableError:
        raise
    except (KeyError, TypeError, ValidationError) as exc:
        raise MethodologyMalformedError(str(path)) from exc


def get_action_value_model_info(
    path: Path = ACTION_VALUE_MODEL_METADATA_PATH,
) -> ActionValueModelInfoResponse:
    try:
        return build_action_value_info(_load_json(path))
    except MethodologyUnavailableError:
        raise
    except (KeyError, TypeError, ValidationError) as exc:
        raise MethodologyMalformedError(str(path)) from exc
