"""Small adapters from database records to explicit API response schemas."""

from app.models import Player, PlayerProfile
from app.schemas import PlayerIdentity, PlayerProfileResponse, PlayerSummary


def player_identity(player: Player) -> PlayerIdentity:
    return PlayerIdentity(
        player_id=player.player_id,
        player_name=player.player_name,
        team_id=player.team_id,
        team_name=player.team_name,
        position=player.position,
        position_group=player.position_group,
    )


def player_summary(player: Player, profile: PlayerProfile) -> PlayerSummary:
    return PlayerSummary(
        **player_identity(player).model_dump(),
        matches_observed=player.matches_observed,
        pass_attempts=player.pass_attempts,
        actual_completion_rate=profile.actual_completion_rate,
        expected_completion_rate=profile.expected_completion_rate,
        completion_above_expected_pp=profile.completion_above_expected_pp,
        progressive_pass_rate=profile.progressive_pass_rate,
        pressure_pass_rate=profile.pressure_pass_rate,
        final_third_entries_per_100_passes=profile.final_third_entries_per_100_passes,
        overall_reliable=player.overall_reliable,
    )


def player_profile(player: Player, profile: PlayerProfile) -> PlayerProfileResponse:
    profile_metrics = {
        column.name: getattr(profile, column.name)
        for column in PlayerProfile.__table__.columns
        if column.name != "player_id"
    }
    return PlayerProfileResponse(
        **player_identity(player).model_dump(),
        matches_observed=player.matches_observed,
        pass_attempts=player.pass_attempts,
        overall_reliable=player.overall_reliable,
        **profile_metrics,
    )
