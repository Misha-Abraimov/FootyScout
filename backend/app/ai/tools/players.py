"""Deterministic player and leaderboard tool adapters."""

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.ai.schemas import (
    ComparePlayersInput,
    LeaderboardInput,
    PlayerDossierInput,
    PlayerDossierResponse,
    PlayerDossierSection,
    SearchPlayersInput,
    SimilarPlayersInput,
)
from app.presenters import player_identity
from app.schemas import (
    ComparisonResponse,
    LeaderboardResponse,
    PlayerListResponse,
    SimilarPlayersResponse,
)
from app.services import comparisons, leaderboards, players


def search_players(
    session: Session,
    request: SearchPlayersInput,
) -> PlayerListResponse:
    return players.search_players(
        session,
        search=request.query,
        team=request.team,
        position_group=request.position_group,
        min_pass_attempts=request.min_pass_attempts,
        sort_by=request.sort_by,
        sort_order=request.sort_order,
        limit=request.limit,
        offset=0,
    )


def get_player_dossier(
    session: Session,
    request: PlayerDossierInput,
) -> PlayerDossierResponse:
    player, profile_record = players.get_player_profile_record(
        session,
        request.player_id,
    )
    passing = None
    shooting = None
    attacking = None
    intelligence = None
    unavailable: dict[PlayerDossierSection, str] = {}

    if PlayerDossierSection.PASSING in request.sections:
        from app.presenters import player_profile

        passing = player_profile(player, profile_record)
    for section, loader in (
        (PlayerDossierSection.SHOOTING, players.get_player_shooting),
        (PlayerDossierSection.ATTACKING_IMPACT, players.get_player_attacking),
        (PlayerDossierSection.INTELLIGENCE, players.get_player_intelligence),
    ):
        if section not in request.sections:
            continue
        try:
            value = loader(session, request.player_id)
        except HTTPException as exc:
            unavailable[section] = str(exc.detail)
            continue
        if section is PlayerDossierSection.SHOOTING:
            shooting = value
        elif section is PlayerDossierSection.ATTACKING_IMPACT:
            attacking = value
        else:
            intelligence = value

    return PlayerDossierResponse(
        player=player_identity(player),
        requested_sections=request.sections,
        passing=passing,
        shooting=shooting,
        attacking_impact=attacking,
        intelligence=intelligence,
        unavailable_sections=unavailable,
    )


def compare_players(
    session: Session,
    request: ComparePlayersInput,
) -> ComparisonResponse:
    return comparisons.compare_players(session, list(request.player_ids))


def get_similar_players(
    session: Session,
    request: SimilarPlayersInput,
) -> SimilarPlayersResponse:
    return players.get_similar_players(
        session,
        request.player_id,
        limit=request.limit,
    )


def get_leaderboard(
    session: Session,
    request: LeaderboardInput,
) -> LeaderboardResponse:
    return leaderboards.get_leaderboard(
        session,
        metric=request.metric,
        position_group=request.position_group,
        team=request.team,
        limit=request.limit,
    )
