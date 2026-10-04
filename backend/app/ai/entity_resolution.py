"""Deterministic player entity resolution using the shared search service."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.policy import PLAYER_SEARCH_MAX_RESULTS
from app.ai.schemas import (
    EntityRef,
    EntityResolutionStatus,
    PlayerResolution,
    SearchPlayersInput,
    TeamRef,
    TeamResolution,
    TeamResolutionCandidate,
)
from app.models import Player, TeamStyleProfile
from app.presenters import player_identity
from app.services import players as player_service
from app.services.players import find_players_by_normalized_name
from app.services.teams import find_loaded_teams_by_name, find_team_profiles_by_name


def resolve_player(
    session: Session,
    reference: EntityRef,
    *,
    search_request: SearchPlayersInput | None = None,
) -> PlayerResolution:
    """Resolve an ID directly or a name through normalized deterministic matching."""
    if reference.player_id is not None:
        player = session.get(Player, reference.player_id)
        if player is None:
            return PlayerResolution(
                status=EntityResolutionStatus.NOT_FOUND,
                query=reference.player_id,
            )
        return PlayerResolution(
            status=EntityResolutionStatus.RESOLVED,
            query=reference.player_id,
            player_id=player.player_id,
            player=player_identity(player),
        )

    assert reference.player_name is not None
    if search_request is not None:
        response = player_service.search_players(
            session,
            search=search_request.query,
            team=search_request.team,
            position_group=search_request.position_group,
            min_pass_attempts=search_request.min_pass_attempts,
            sort_by=search_request.sort_by,
            sort_order=search_request.sort_order,
            limit=search_request.limit,
            offset=0,
        )
        candidates = [
            player_identity(player)
            for item in response.items
            if (player := session.get(Player, item.player_id)) is not None
        ]
        if response.total == 0:
            return PlayerResolution(
                status=EntityResolutionStatus.NOT_FOUND,
                query=reference.player_name,
            )
        if response.total == 1 and len(candidates) == 1:
            match = candidates[0]
            return PlayerResolution(
                status=EntityResolutionStatus.RESOLVED,
                query=reference.player_name,
                player_id=match.player_id,
                player=match,
                candidates=candidates,
            )
        return PlayerResolution(
            status=EntityResolutionStatus.AMBIGUOUS,
            query=reference.player_name,
            candidates=candidates,
        )

    matches = find_players_by_normalized_name(
        session,
        reference.player_name,
        limit=PLAYER_SEARCH_MAX_RESULTS,
    )
    candidates = [player_identity(match) for match in matches]
    if not matches:
        return PlayerResolution(
            status=EntityResolutionStatus.NOT_FOUND,
            query=reference.player_name,
        )
    if len(matches) == 1:
        match = matches[0]
        return PlayerResolution(
            status=EntityResolutionStatus.RESOLVED,
            query=reference.player_name,
            player_id=match.player_id,
            player=player_identity(match),
            candidates=candidates,
        )
    return PlayerResolution(
        status=EntityResolutionStatus.AMBIGUOUS,
        query=reference.player_name,
        candidates=candidates,
    )


def resolve_team(session: Session, reference: TeamRef) -> TeamResolution:
    """Resolve loaded teams separately from qualified production analytics support."""
    if reference.team_id is not None:
        profile = session.get(TeamStyleProfile, reference.team_id)
        player = session.scalars(
            select(Player)
            .where(Player.team_id == reference.team_id)
            .order_by(Player.player_id.asc())
            .limit(1)
        ).first()
        if profile is None and player is None:
            return TeamResolution(
                status=EntityResolutionStatus.NOT_FOUND,
                query=reference.team_id,
            )
        candidate = TeamResolutionCandidate(
            team_id=profile.team_id if profile is not None else player.team_id,
            team_name=profile.team_name if profile is not None else player.team_name,
            analytics_supported=profile is not None,
        )
        return TeamResolution(
            status=EntityResolutionStatus.RESOLVED,
            query=reference.team_id,
            team_id=candidate.team_id,
            team=candidate,
            candidates=[candidate],
            analytics_supported=candidate.analytics_supported,
        )

    assert reference.team_name is not None
    profile_matches = find_team_profiles_by_name(session, reference.team_name)
    loaded_matches = find_loaded_teams_by_name(session, reference.team_name)
    candidates_by_id = {
        team_id: TeamResolutionCandidate(
            team_id=team_id,
            team_name=team_name,
            analytics_supported=session.get(TeamStyleProfile, team_id) is not None,
        )
        for team_id, team_name in loaded_matches
    }
    for profile in profile_matches:
        candidates_by_id[profile.team_id] = TeamResolutionCandidate(
            team_id=profile.team_id,
            team_name=profile.team_name,
            analytics_supported=True,
        )
    if not candidates_by_id:
        profiles = session.scalars(
            select(TeamStyleProfile)
            .where(TeamStyleProfile.team_name.icontains(reference.team_name, autoescape=True))
            .order_by(TeamStyleProfile.team_name.asc(), TeamStyleProfile.team_id.asc())
        ).all()
        for profile in profiles:
            candidates_by_id[profile.team_id] = TeamResolutionCandidate(
                team_id=profile.team_id,
                team_name=profile.team_name,
                analytics_supported=True,
            )
    candidates = sorted(
        candidates_by_id.values(), key=lambda candidate: (candidate.team_name, candidate.team_id)
    )
    if not candidates:
        return TeamResolution(
            status=EntityResolutionStatus.NOT_FOUND,
            query=reference.team_name,
        )
    if len(candidates) > 1:
        return TeamResolution(
            status=EntityResolutionStatus.AMBIGUOUS,
            query=reference.team_name,
            candidates=candidates,
        )
    return TeamResolution(
        status=EntityResolutionStatus.RESOLVED,
        query=reference.team_name,
        team_id=candidates[0].team_id,
        team=candidates[0],
        candidates=candidates,
        analytics_supported=candidates[0].analytics_supported,
    )
