"""Curated public model-methodology information."""

from fastapi import APIRouter, HTTPException, status

from app.runtime_metadata import PASS_MODEL_METADATA_PATH
from app.schemas import ModelInfoResponse
from app.services.methodology import (
    MethodologyMalformedError,
    MethodologyUnavailableError,
    build_pass_model_info,
    get_pass_model_info,
)

router = APIRouter(prefix="/api/model", tags=["model"])
MODEL_METADATA_PATH = PASS_MODEL_METADATA_PATH
build_model_info = build_pass_model_info


@router.get("", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    try:
        return get_pass_model_info(MODEL_METADATA_PATH)
    except MethodologyUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is currently unavailable.",
        ) from exc
    except MethodologyMalformedError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model metadata is malformed.",
        ) from exc
