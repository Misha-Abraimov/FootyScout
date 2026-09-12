"""Curated public model-methodology information."""

import json
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from app.runtime_metadata import PASS_MODEL_METADATA_PATH
from app.schemas import (
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
)

router = APIRouter(prefix="/api/model", tags=["model"])
MODEL_METADATA_PATH = PASS_MODEL_METADATA_PATH


def _metric_map(raw_metrics: dict[str, Any]) -> dict[str, ModelMetrics]:
    return {
        name: ModelMetrics.model_validate(values)
        for name, values in raw_metrics.items()
        if isinstance(values, dict)
    }


def build_model_info(raw: dict[str, Any]) -> ModelInfoResponse:
    """Select only documented, frontend-safe fields from training metadata."""
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
            uncalibrated_validation_metrics=calibration[
                "uncalibrated_validation_metrics"
            ],
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


@router.get("", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    try:
        with MODEL_METADATA_PATH.open(encoding="utf-8") as metadata_file:
            raw = json.load(metadata_file)
        if not isinstance(raw, dict):
            raise TypeError("metadata root must be an object")
        return build_model_info(raw)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is currently unavailable.",
        ) from exc
    except (
        AttributeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        UnicodeError,
        ValidationError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model metadata is malformed.",
        ) from exc
