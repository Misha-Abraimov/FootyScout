"""Read-only two-player comparison service."""

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.intelligence import get_intelligence_response
from app.models import Player, PlayerAttackingProfile, PlayerProfile
from app.presenters import player_profile
from app.schemas import AttackingProfileResponse, ComparisonMetadata, ComparisonResponse


def compare_players(session: Session, player_ids: list[int]) -> ComparisonResponse:
    rows = session.execute(
        select(Player, PlayerProfile)
        .join(PlayerProfile, PlayerProfile.player_id == Player.player_id)
        .where(Player.player_id.in_(player_ids))
    ).all()
    by_player_id = {player.player_id: (player, profile) for player, profile in rows}
    missing_ids = [player_id for player_id in player_ids if player_id not in by_player_id]
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown player IDs: {missing_ids}.",
        )

    profiles = [player_profile(*by_player_id[player_id]) for player_id in player_ids]
    attacking = []
    for player_id in player_ids:
        value = session.get(PlayerAttackingProfile, player_id)
        attacking.append(AttackingProfileResponse.model_validate(value) if value else None)
    intelligence = [
        get_intelligence_response(session, by_player_id[player_id][0]) for player_id in player_ids
    ]
    return ComparisonResponse(
        comparison=ComparisonMetadata(
            player_ids=player_ids,
            same_position_group=(profiles[0].position_group == profiles[1].position_group),
        ),
        players=profiles,
        attacking=attacking,
        intelligence=intelligence,
    )
