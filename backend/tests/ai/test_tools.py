"""Tool behavior, evidence contracts, and REST/source-of-truth parity."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.ai.schemas import (
    ProductionStatus,
    SourceCategory,
    ToolExecutionStatus,
    ToolName,
)
from app.ai.tools.registry import execute_tool, get_tool_definition


def _success(
    session: Session,
    name: ToolName,
    arguments: dict[str, object],
) -> dict[str, object]:
    execution = execute_tool(session, name, arguments)
    assert execution.status is ToolExecutionStatus.SUCCESS
    assert execution.result is not None
    return execution.result


def test_player_search_parity(
    ai_client: TestClient,
    ai_session: Session,
) -> None:
    api = ai_client.get(
        "/api/players",
        params={"search": "Alice", "limit": 10},
    ).json()
    tool = _success(
        ai_session,
        ToolName.SEARCH_PLAYERS,
        {"query": "Alice", "limit": 10},
    )
    assert tool == api


def test_player_lookup_mechanics_remain_internal_to_tool_evidence() -> None:
    search = get_tool_definition(ToolName.SEARCH_PLAYERS)
    dossier = get_tool_definition(ToolName.GET_PLAYER_DOSSIER)

    assert search.limitations == ()
    assert any("deterministic normalized matching" in note for note in search.internal_notes)
    assert all("deterministic" not in item for item in search.limitations)
    assert dossier.limitations == (
        "A profile describes observed event data, not future performance.",
    )
    assert dossier.internal_notes == ("The dossier does not return raw event rows.",)
    assert all("raw event rows" not in item for item in dossier.limitations)


def test_player_profile_and_dossier_parity(
    ai_client: TestClient,
    ai_session: Session,
) -> None:
    api = ai_client.get("/api/players/1").json()
    result = _success(
        ai_session,
        ToolName.GET_PLAYER_DOSSIER,
        {"player_id": 1, "sections": ["passing"]},
    )
    assert result["passing"] == api
    assert result["player"]["player_id"] == api["player_id"]


def test_similar_players_parity_and_evidence(
    ai_client: TestClient,
    ai_session: Session,
) -> None:
    api = ai_client.get("/api/players/1/similar", params={"limit": 5}).json()
    execution = execute_tool(
        ai_session,
        ToolName.GET_SIMILAR_PLAYERS,
        {"player_id": 1, "limit": 5},
    )
    assert execution.status is ToolExecutionStatus.SUCCESS
    assert execution.result == api
    assert execution.arguments == {"player_id": 1, "limit": 5}
    assert execution.entity_ids == [1, 4]
    assert execution.source_category is SourceCategory.POSTGRESQL
    assert any("not player quality" in warning for warning in execution.warnings)


def test_leaderboard_and_comparison_parity(
    ai_client: TestClient,
    ai_session: Session,
) -> None:
    api_board = ai_client.get(
        "/api/leaderboard",
        params={"metric": "completion_above_expected_pp", "limit": 3},
    ).json()
    tool_board = _success(
        ai_session,
        ToolName.GET_LEADERBOARD,
        {"metric": "completion_above_expected_pp", "limit": 3},
    )
    assert tool_board == api_board

    api_comparison = ai_client.get(
        "/api/compare",
        params={"player_ids": "2,1"},
    ).json()
    tool_comparison = _success(
        ai_session,
        ToolName.COMPARE_PLAYERS,
        {"player_ids": [2, 1]},
    )
    assert tool_comparison == api_comparison


def test_role_fit_and_recommendation_parity_preserve_distance_direction(
    ai_client: TestClient,
    ai_session: Session,
) -> None:
    api_fit = ai_client.get(
        "/api/players/4/role-fit",
        params={"target_team_id": 904},
    ).json()
    tool_fit = _success(
        ai_session,
        ToolName.GET_ROLE_FIT,
        {"player_id": 4, "target_team_id": 904},
    )
    assert tool_fit == api_fit
    assert tool_fit["role_distance"] == pytest.approx(0.39)

    api_recommendations = ai_client.get(
        "/api/teams/904/roles/MID/recommendations",
        params={"limit": 5},
    ).json()
    tool_recommendations = _success(
        ai_session,
        ToolName.GET_ROLE_RECOMMENDATIONS,
        {"team_id": 904, "position_group": "MID", "limit": 5},
    )
    assert tool_recommendations == api_recommendations
    distances = [item["role_distance"] for item in tool_recommendations["items"]]
    assert distances == sorted(distances)
    assert all(
        item["player"]["team_name"] != "Bayer Leverkusen" for item in tool_recommendations["items"]
    )


def test_team_intelligence_parity_and_unsupported_states(
    ai_client: TestClient,
    ai_session: Session,
) -> None:
    api = ai_client.get("/api/teams/904/intelligence").json()
    tool = _success(
        ai_session,
        ToolName.GET_TEAM_INTELLIGENCE,
        {"team_id": 904},
    )
    assert tool["intelligence"] == api
    unsupported = execute_tool(
        ai_session,
        ToolName.GET_TEAM_INTELLIGENCE,
        {"team_id": 169},
    )
    assert unsupported.status is ToolExecutionStatus.NOT_FOUND
    goalkeeper = execute_tool(
        ai_session,
        ToolName.GET_ROLE_RECOMMENDATIONS,
        {"team_id": 904, "position_group": "GK", "limit": 5},
    )
    assert goalkeeper.status is ToolExecutionStatus.INVALID_ARGUMENTS


def test_tool_hard_limits_invalid_enums_extras_and_unknown_names(
    ai_session: Session,
) -> None:
    for name, arguments in (
        (ToolName.SEARCH_PLAYERS, {"query": "Alice", "limit": 21}),
        (ToolName.GET_SIMILAR_PLAYERS, {"player_id": 1, "limit": 11}),
        (ToolName.GET_LEADERBOARD, {"metric": "player_name", "limit": 10}),
        (
            ToolName.GET_ROLE_RECOMMENDATIONS,
            {"team_id": 904, "position_group": "MID", "limit": 21},
        ),
        (ToolName.SEARCH_PLAYERS, {"query": "Alice", "unexpected": True}),
    ):
        result = execute_tool(ai_session, name, arguments)
        assert result.status is ToolExecutionStatus.INVALID_ARGUMENTS
    with pytest.raises(ValueError, match="Unknown AI Scout tool"):
        get_tool_definition("run_sql")


def test_methodology_is_deterministic_and_labels_production_status(
    ai_session: Session,
) -> None:
    possession = execute_tool(
        ai_session,
        ToolName.GET_METHODOLOGY,
        {"topic": "possession_value"},
    )
    assert possession.status is ToolExecutionStatus.SUCCESS
    assert possession.production_status is ProductionStatus.PRODUCTION
    assert possession.result is not None
    assert possession.result["current_production_model"] == "xgboost"
    assert "pytorch_causal_transformer" in possession.result["experimental_models"]
    transformer_source = next(
        source for source in possession.result["sources"] if "transformer" in source["source_id"]
    )
    assert transformer_source["status"] == "experimental"

    xpass = _success(
        ai_session,
        ToolName.GET_METHODOLOGY,
        {"topic": "xpass"},
    )
    assert xpass["current_production_model"] == "xgboost"


@pytest.mark.parametrize(
    "topic",
    [
        "xpass",
        "xg",
        "possession_value",
        "attacking_impact",
        "player_profiles",
        "percentiles",
        "archetypes",
        "similarity",
        "team_intelligence",
        "role_fit",
        "role_recommendations",
    ],
)
def test_every_guarded_methodology_topic_executes_without_entities(
    ai_session: Session,
    topic: str,
) -> None:
    result = _success(
        ai_session,
        ToolName.GET_METHODOLOGY,
        {"topic": topic},
    )
    assert result["topic"] == topic
    assert result["summary"]
    assert result["sources"]
