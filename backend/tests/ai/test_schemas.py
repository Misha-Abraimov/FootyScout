"""Strict AI Scout intent and tool input contracts."""

import pytest
from pydantic import TypeAdapter, ValidationError

from app.ai.schemas import (
    ComparePlayersInput,
    EntityRef,
    IntentKind,
    RoleRecommendationsInput,
    ScoutIntent,
    SearchPlayersInput,
    TeamRef,
)


def test_entity_reference_requires_exactly_one_identifier() -> None:
    assert EntityRef(player_id=3500).player_id == 3500
    assert EntityRef(player_name="Granit Xhaka").player_name == "Granit Xhaka"
    with pytest.raises(ValidationError):
        EntityRef()
    with pytest.raises(ValidationError):
        EntityRef(player_id=3500, player_name="Granit Xhaka")
    assert TeamRef(team_name="Bayer Leverkusen").team_name == "Bayer Leverkusen"
    with pytest.raises(ValidationError):
        TeamRef(team_id=904, team_name="Bayer Leverkusen")


def test_intent_union_is_discriminated_and_strict() -> None:
    adapter = TypeAdapter(ScoutIntent)
    parsed = adapter.validate_python(
        {"kind": "similar_players", "player": {"player_id": 3500}, "limit": 6}
    )
    assert parsed.kind is IntentKind.SIMILAR_PLAYERS
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "unknown"})
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "methodology",
                "topic": "xpass",
                "unexpected": "not allowed",
            }
        )
    team_intent = adapter.validate_python(
        {
            "kind": "team_analysis",
            "team": {"team_name": "Bayer Leverkusen"},
            "position_group": "MID",
        }
    )
    assert team_intent.team.team_name == "Bayer Leverkusen"


def test_tool_inputs_reject_extras_invalid_limits_and_duplicate_ids() -> None:
    with pytest.raises(ValidationError):
        SearchPlayersInput(query="Xhaka", limit=21)
    with pytest.raises(ValidationError):
        SearchPlayersInput(query="Xhaka", sql="DROP TABLE players")
    with pytest.raises(ValidationError):
        ComparePlayersInput(player_ids=(1, 1))
    with pytest.raises(ValidationError):
        RoleRecommendationsInput(team_id=904, position_group="GK", limit=5)
