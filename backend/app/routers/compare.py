"""Two-player profile comparison endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.intelligence import get_intelligence_response
from app.models import Player, PlayerAttackingProfile, PlayerProfile
from app.presenters import player_profile
from app.schemas import AttackingProfileResponse, ComparisonMetadata, ComparisonResponse

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
    rows = session.execute(
        select(Player, PlayerProfile)
        .join(PlayerProfile, PlayerProfile.player_id == Player.player_id)
        .where(Player.player_id.in_(requested_ids))
    ).all()
    by_player_id = {player.player_id: (player, profile) for player, profile in rows}
    missing_ids = [player_id for player_id in requested_ids if player_id not in by_player_id]
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown player IDs: {missing_ids}.",
        )

    profiles = [
        player_profile(*by_player_id[player_id]) for player_id in requested_ids
    ]
    attacking = []
    for player_id in requested_ids:
        value = session.get(PlayerAttackingProfile, player_id)
        attacking.append(AttackingProfileResponse.model_validate(value) if value else None)
    intelligence = [
        get_intelligence_response(session, by_player_id[player_id][0])
        for player_id in requested_ids
    ]
    return ComparisonResponse(
        comparison=ComparisonMetadata(
            player_ids=requested_ids,
            same_position_group=(
                profiles[0].position_group == profiles[1].position_group
            ),
        ),
        players=profiles,
        attacking=attacking,
        intelligence=intelligence,
    )
