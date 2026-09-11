"""Reliability-aware player leaderboard endpoint."""

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.models import Player, PlayerAttackingProfile, PlayerProfile
from app.presenters import player_identity
from app.schemas import (
    LeaderboardEntry,
    LeaderboardMetric,
    LeaderboardResponse,
    PositionGroup,
)

router = APIRouter(prefix="/api/leaderboard", tags=["leaderboard"])
MINIMUM_PASS_ATTEMPTS = 100


@dataclass(frozen=True)
class MetricDefinition:
    value_column: Any
    reliability_column: Any
    attempts_column: Any


METRIC_DEFINITIONS = {
    LeaderboardMetric.COMPLETION_ABOVE_EXPECTED_PP: MetricDefinition(
        PlayerProfile.completion_above_expected_pp,
        Player.overall_reliable,
        Player.pass_attempts,
    ),
    LeaderboardMetric.PRESSURE_ABOVE_EXPECTED_PP: MetricDefinition(
        PlayerProfile.pressure_above_expected_pp,
        PlayerProfile.pressure_reliable,
        PlayerProfile.pressure_attempts,
    ),
    LeaderboardMetric.PROGRESSIVE_ABOVE_EXPECTED_PP: MetricDefinition(
        PlayerProfile.progressive_above_expected_pp,
        PlayerProfile.progressive_reliable,
        PlayerProfile.progressive_attempts,
    ),
    LeaderboardMetric.LONG_PASS_ABOVE_EXPECTED_PP: MetricDefinition(
        PlayerProfile.long_pass_above_expected_pp,
        PlayerProfile.long_pass_reliable,
        PlayerProfile.long_pass_attempts,
    ),
    LeaderboardMetric.FINAL_THIRD_ENTRIES_PER_100_PASSES: MetricDefinition(
        PlayerProfile.final_third_entries_per_100_passes,
        Player.overall_reliable,
        Player.pass_attempts,
    ),
    LeaderboardMetric.EXPECTED_COMPLETION_RATE: MetricDefinition(
        PlayerProfile.expected_completion_rate,
        Player.overall_reliable,
        Player.pass_attempts,
    ),
    LeaderboardMetric.PROGRESSIVE_PASS_RATE: MetricDefinition(
        PlayerProfile.progressive_pass_rate,
        Player.overall_reliable,
        Player.pass_attempts,
    ),
    LeaderboardMetric.ATTACKING_VALUE_PER_100_ACTIONS: MetricDefinition(
        PlayerAttackingProfile.attacking_value_per_100_actions,
        PlayerAttackingProfile.attacking_value_reliable,
        PlayerAttackingProfile.actions,
    ),
    LeaderboardMetric.PASS_VALUE_PER_100_PASSES: MetricDefinition(
        PlayerAttackingProfile.pass_value_per_100_passes,
        PlayerAttackingProfile.pass_value_reliable,
        PlayerAttackingProfile.passes,
    ),
    LeaderboardMetric.CARRY_VALUE_PER_100_CARRIES: MetricDefinition(
        PlayerAttackingProfile.carry_value_per_100_carries,
        PlayerAttackingProfile.carry_value_reliable,
        PlayerAttackingProfile.carries,
    ),
    LeaderboardMetric.PROGRESSIVE_VALUE_PER_100_ACTIONS: MetricDefinition(
        PlayerAttackingProfile.progressive_value_per_100_actions,
        PlayerAttackingProfile.attacking_value_reliable,
        PlayerAttackingProfile.actions,
    ),
    LeaderboardMetric.PRESSURE_VALUE_PER_100_ACTIONS: MetricDefinition(
        PlayerAttackingProfile.pressure_value_per_100_actions,
        PlayerAttackingProfile.attacking_value_reliable,
        PlayerAttackingProfile.actions,
    ),
}


@router.get("", response_model=LeaderboardResponse)
def get_leaderboard(
    session: Annotated[Session, Depends(get_db)],
    metric: LeaderboardMetric = LeaderboardMetric.COMPLETION_ABOVE_EXPECTED_PP,
    position_group: PositionGroup | None = None,
    team: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> LeaderboardResponse:
    definition = METRIC_DEFINITIONS[metric]
    attacking_metric = metric in {
        LeaderboardMetric.ATTACKING_VALUE_PER_100_ACTIONS,
        LeaderboardMetric.PASS_VALUE_PER_100_PASSES,
        LeaderboardMetric.CARRY_VALUE_PER_100_CARRIES,
        LeaderboardMetric.PROGRESSIVE_VALUE_PER_100_ACTIONS,
        LeaderboardMetric.PRESSURE_VALUE_PER_100_ACTIONS,
    }
    filters: list[ColumnElement[bool]] = [
        definition.reliability_column.is_(True),
        definition.value_column.is_not(None),
    ]
    if not attacking_metric:
        filters.append(Player.pass_attempts >= MINIMUM_PASS_ATTEMPTS)
    if position_group:
        filters.append(Player.position_group == position_group.value)
    if team:
        filters.append(func.lower(Player.team_name) == team.casefold())

    total = session.scalar(
        select(func.count())
        .select_from(Player)
        .join(PlayerProfile, PlayerProfile.player_id == Player.player_id)
        .outerjoin(PlayerAttackingProfile, PlayerAttackingProfile.player_id == Player.player_id)
        .where(*filters)
    ) or 0
    rows = session.execute(
        select(
            Player,
            definition.value_column.label("metric_value"),
            definition.attempts_column.label("relevant_attempts"),
        )
        .join(PlayerProfile, PlayerProfile.player_id == Player.player_id)
        .outerjoin(PlayerAttackingProfile, PlayerAttackingProfile.player_id == Player.player_id)
        .where(*filters)
        .order_by(definition.value_column.desc(), Player.player_id.asc())
        .limit(limit)
    ).all()
    items = [
        LeaderboardEntry(
            **player_identity(player).model_dump(),
            rank=rank,
            pass_attempts=player.pass_attempts,
            relevant_attempts=relevant_attempts,
            metric=metric,
            metric_value=metric_value,
            overall_reliable=player.overall_reliable,
        )
        for rank, (player, metric_value, relevant_attempts) in enumerate(rows, start=1)
    ]
    return LeaderboardResponse(
        metric=metric,
        total=total,
        limit=limit,
        minimum_pass_attempts=0 if attacking_metric else MINIMUM_PASS_ATTEMPTS,
        items=items,
    )
