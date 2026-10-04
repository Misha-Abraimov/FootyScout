"""Two-player profile comparison endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas import ComparisonResponse
from app.services.comparisons import compare_players as compare_players_service

router = APIRouter(prefix="/api/compare", tags=["comparison"])


def parse_player_ids(raw_player_ids: str) -> list[int]:
    parts = [part.strip() for part in raw_player_ids.split(",")]
    if len(parts) != 2 or any(not part for part in parts):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="player_ids must contain exactly two comma-separated player IDs.",
        )
    try:
        player_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="player_ids must contain integers only.",
        ) from exc
    if any(player_id <= 0 for player_id in player_ids):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="player IDs must be positive integers.",
        )
    if player_ids[0] == player_ids[1]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="player_ids must contain two unique player IDs.",
        )
    return player_ids


@router.get("", response_model=ComparisonResponse)
def compare_players(
    session: Annotated[Session, Depends(get_db)],
    player_ids: Annotated[
        str,
        Query(description="Exactly two comma-separated StatsBomb player IDs."),
    ],
) -> ComparisonResponse:
    requested_ids = parse_player_ids(player_ids)
    return compare_players_service(session, requested_ids)
