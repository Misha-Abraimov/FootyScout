"""Public, path-safe methodology metadata for attacking action value."""

from fastapi import APIRouter, HTTPException, status

from app.runtime_metadata import ACTION_VALUE_MODEL_METADATA_PATH
from app.schemas import ActionValueModelInfoResponse
from app.services.methodology import (
    MethodologyMalformedError,
    MethodologyUnavailableError,
)
from app.services.methodology import (
    get_action_value_model_info as get_action_value_model_info_service,
)

router = APIRouter(prefix="/api/models/action-value", tags=["models"])
METADATA_PATH = ACTION_VALUE_MODEL_METADATA_PATH


@router.get("", response_model=ActionValueModelInfoResponse)
def get_action_value_model_info() -> ActionValueModelInfoResponse:
    try:
        return get_action_value_model_info_service(METADATA_PATH)
    except MethodologyUnavailableError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Action-value metadata unavailable",
        ) from exc
    except MethodologyMalformedError as exc:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Action-value metadata malformed",
        ) from exc
