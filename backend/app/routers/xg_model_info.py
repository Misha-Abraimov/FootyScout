"""Public, path-safe methodology metadata for the expected-goals model."""

import json
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from app.runtime_metadata import XG_MODEL_METADATA_PATH
from app.schemas import XGModelInfoResponse

router = APIRouter(prefix="/api/models/xg", tags=["models"])
METADATA_PATH = XG_MODEL_METADATA_PATH


def build_xg_model_info(raw: dict[str, Any]) -> XGModelInfoResponse:
    """Expose analytical metadata while excluding artifact paths and match IDs."""
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


@router.get("", response_model=XGModelInfoResponse)
def get_xg_model_info() -> XGModelInfoResponse:
    try:
        raw = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("metadata root must be an object")
        return build_xg_model_info(raw)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="xG model metadata is currently unavailable.",
        ) from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="xG model metadata is malformed.",
        ) from exc
