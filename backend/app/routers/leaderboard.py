"""Reliability-aware player leaderboard endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas import (
    LeaderboardMetric,
    LeaderboardResponse,
    PositionGroup,
)
from app.services.leaderboards import get_leaderboard as get_leaderboard_service

router = APIRouter(prefix="/api/leaderboard", tags=["leaderboard"])


@router.get("", response_model=LeaderboardResponse)
def get_leaderboard(
    session: Annotated[Session, Depends(get_db)],
    metric: LeaderboardMetric = LeaderboardMetric.COMPLETION_ABOVE_EXPECTED_PP,
    position_group: PositionGroup | None = None,
    team: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> LeaderboardResponse:
    return get_leaderboard_service(
        session,
        metric=metric,
        position_group=position_group,
        team=team,
        limit=limit,
    )
