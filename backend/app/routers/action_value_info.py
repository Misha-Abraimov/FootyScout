"""Public, path-safe methodology metadata for attacking action value."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from app.schemas import ActionValueModelInfoResponse

router = APIRouter(prefix="/api/models/action-value", tags=["models"])
METADATA_PATH = Path(__file__).resolve().parents[3] / "models" / "action_value_model_metadata.json"


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


@router.get("", response_model=ActionValueModelInfoResponse)
def get_action_value_model_info() -> ActionValueModelInfoResponse:
    try:
        raw = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("metadata root must be an object")
        return build_action_value_info(raw)
    except OSError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Action-value metadata unavailable") from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Action-value metadata malformed") from exc
