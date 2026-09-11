"""Lightweight PostgreSQL-derived filter metadata."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.models import Player
from app.schemas import MetaResponse

router = APIRouter(prefix="/api/meta", tags=["metadata"])
POSITION_GROUP_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


@router.get("", response_model=MetaResponse)
def get_meta(session: Annotated[Session, Depends(get_db)]) -> MetaResponse:
    teams = list(
        session.scalars(select(Player.team_name).distinct().order_by(Player.team_name))
    )
    position_groups = list(
        session.scalars(select(Player.position_group).distinct())
    )
    position_groups.sort(key=lambda group: POSITION_GROUP_ORDER.get(group, 99))
    return MetaResponse(
        teams=teams,
        position_groups=position_groups,
        player_count=session.scalar(select(func.count()).select_from(Player)) or 0,
        reliable_player_count=session.scalar(
            select(func.count())
            .select_from(Player)
            .where(Player.overall_reliable.is_(True))
        )
        or 0,
    )
