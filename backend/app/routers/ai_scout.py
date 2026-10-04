"""Public product endpoint for one grounded AI Scout question."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas import AIScoutRequest, AIScoutResponse
from app.services.ai_scout import (
    AIScoutRunner,
    get_ai_scout_runner,
    public_ai_scout_response,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["AI Scout"])


@router.post("/api/ai-scout", response_model=AIScoutResponse)
def ask_ai_scout(
    request: AIScoutRequest,
    session: Annotated[Session, Depends(get_db)],
    runner: Annotated[AIScoutRunner, Depends(get_ai_scout_runner)],
) -> AIScoutResponse:
    """Run the synchronous bounded workflow; application statuses remain HTTP 200."""
    try:
        result = runner.run(session, request.question)
    except Exception as exc:
        logger.exception("AI Scout workflow failed", exc_info=exc)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "AI Scout could not complete this request.",
        ) from exc
    return public_ai_scout_response(result)
