"""Public, path-safe methodology metadata for the expected-goals model."""

from fastapi import APIRouter, HTTPException, status

from app.runtime_metadata import XG_MODEL_METADATA_PATH
from app.schemas import XGModelInfoResponse
from app.services.methodology import (
    MethodologyMalformedError,
    MethodologyUnavailableError,
)
from app.services.methodology import get_xg_model_info as get_xg_model_info_service

router = APIRouter(prefix="/api/models/xg", tags=["models"])
METADATA_PATH = XG_MODEL_METADATA_PATH


@router.get("", response_model=XGModelInfoResponse)
def get_xg_model_info() -> XGModelInfoResponse:
    try:
        return get_xg_model_info_service(METADATA_PATH)
    except MethodologyUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="xG model metadata is currently unavailable.",
        ) from exc
    except MethodologyMalformedError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="xG model metadata is malformed.",
        ) from exc
