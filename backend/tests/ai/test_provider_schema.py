"""OpenAI Structured Outputs compatibility and internal-plan conversion."""

from __future__ import annotations

from typing import Any

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.ai.plan_normalizer import prepare_plan
from app.ai.provider_schemas import LLMScoutPlan
from app.ai.schemas import PlanPreparationStatus, ScoutPlan, ToolName

UNSUPPORTED_STRUCTURED_OUTPUT_KEYWORDS = frozenset(
    {
        "oneOf",
        "discriminator",
        "prefixItems",
        "allOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "patternProperties",
    }
)


def _provider_schema() -> dict[str, Any]:
    return to_strict_json_schema(LLMScoutPlan)


def _walk(value: object, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    nodes: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        nodes.append((path, value))
        for key, child in value.items():
            nodes.extend(_walk(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            nodes.extend(_walk(child, f"{path}[{index}]"))
    return nodes


def test_provider_schema_uses_supported_structured_outputs_subset() -> None:
    schema = _provider_schema()
    assert schema == _provider_schema()
    assert schema["type"] == "object"
    for path, node in _walk(schema):
        forbidden = UNSUPPORTED_STRUCTURED_OUTPUT_KEYWORDS.intersection(node)
        assert not forbidden, f"Unsupported schema keys at {path}: {sorted(forbidden)}"
        if node.get("type") == "object" or "properties" in node:
            assert node.get("additionalProperties") is False, path
            assert set(node.get("properties", {})) == set(node.get("required", [])), path
    assert "anyOf" in schema["properties"]["intent"]
    assert "anyOf" in schema["properties"]["calls"]["items"]


@pytest.mark.parametrize(
    ("provider_payload", "expected_payload"),
    [
        (
            {
                "decision": "ready",
                "intent": {"kind": "player_search", "query": "Alice", "limit": 5},
                "calls": [
                    {
                        "name": "search_players",
                        "arguments": {"query": "Alice", "limit": 5},
                    }
                ],
            },
            None,
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "player_profile",
                    "player": {"player_name": "Alice"},
                    "sections": ["passing"],
                },
                "calls": [
                    {
                        "name": "get_player_dossier",
                        "arguments": {
                            "player": {"player_name": "Alice"},
                            "sections": ["passing"],
                        },
                    }
                ],
            },
            None,
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "player_comparison",
                    "player_a": {"player_id": 1},
                    "player_b": {"player_id": 2},
                },
                "calls": [
                    {
                        "name": "compare_players",
                        "arguments": {
                            "player_a": {"player_id": 1},
                            "player_b": {"player_id": 2},
                        },
                    }
                ],
            },
            {
                "decision": "ready",
                "intent": {
                    "kind": "player_comparison",
                    "players": [{"player_id": 1}, {"player_id": 2}],
                },
                "calls": [
                    {
                        "name": "compare_players",
                        "arguments": {"players": [{"player_id": 1}, {"player_id": 2}]},
                    }
                ],
            },
        ),
        (
            {
                "decision": "ready",
                "intent": {"kind": "similar_players", "player": {"player_id": 1}},
                "calls": [
                    {
                        "name": "get_similar_players",
                        "arguments": {"player": {"player_id": 1}, "limit": 6},
                    }
                ],
            },
            None,
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "leaderboard",
                    "metric": "completion_above_expected_pp",
                    "position_group": "MID",
                    "limit": 10,
                },
                "calls": [
                    {
                        "name": "get_leaderboard",
                        "arguments": {
                            "metric": "completion_above_expected_pp",
                            "position_group": "MID",
                            "limit": 10,
                        },
                    }
                ],
            },
            None,
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "team_analysis",
                    "team": {"team_id": 904},
                    "position_group": "MID",
                },
                "calls": [
                    {
                        "name": "get_team_intelligence",
                        "arguments": {
                            "team": {"team_id": 904},
                            "position_group": "MID",
                        },
                    }
                ],
            },
            None,
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "role_fit",
                    "player": {"player_id": 4},
                    "target_team": {"team_id": 904},
                },
                "calls": [
                    {
                        "name": "get_role_fit",
                        "arguments": {
                            "player": {"player_id": 4},
                            "target_team": {"team_id": 904},
                        },
                    }
                ],
            },
            None,
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "role_recommendations",
                    "team": {"team_id": 904},
                    "position_group": "MID",
                    "limit": 5,
                },
                "calls": [
                    {
                        "name": "get_role_recommendations",
                        "arguments": {
                            "team": {"team_id": 904},
                            "position_group": "MID",
                            "limit": 5,
                        },
                    }
                ],
            },
            None,
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "methodology",
                    "topic": "xpass",
                    "question": "How is xPass calculated?",
                },
                "calls": [{"name": "get_methodology", "arguments": {"topic": "xpass"}}],
            },
            None,
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "player_profile",
                    "player": {"player_name": "Bob"},
                    "sections": ["passing"],
                },
                "calls": [
                    {
                        "name": "search_players",
                        "arguments": {"query": "Bob", "limit": 10},
                    },
                    {
                        "name": "get_player_dossier",
                        "arguments": {
                            "player": {"player_name": "Bob"},
                            "sections": ["passing"],
                        },
                    },
                ],
            },
            None,
        ),
    ],
)
def test_provider_plan_converts_to_strict_internal_scout_plan(
    provider_payload: dict[str, Any],
    expected_payload: dict[str, Any] | None,
) -> None:
    provider_plan = LLMScoutPlan.model_validate(provider_payload)
    internal = provider_plan.to_scout_plan()
    expected = ScoutPlan.model_validate(expected_payload or provider_payload)
    assert internal == expected


def test_provider_schema_rejects_malformed_irrelevant_and_excess_calls() -> None:
    with pytest.raises(ValidationError):
        LLMScoutPlan.model_validate({"calls": []})
    with pytest.raises(ValidationError):
        LLMScoutPlan.model_validate(
            {
                "decision": "ready",
                "intent": {"kind": "player_search", "query": "Alice"},
                "calls": [
                    {
                        "name": "search_players",
                        "arguments": {"query": "Alice", "sql": "SELECT 1"},
                    }
                ],
            }
        )
    call = {"name": "search_players", "arguments": {"query": "Alice"}}
    with pytest.raises(ValidationError):
        LLMScoutPlan.model_validate(
            {
                "decision": "ready",
                "intent": {"kind": "player_search", "query": "Alice"},
                "calls": [call] * 7,
            }
        )


def test_internal_plan_rejects_duplicate_comparison_players() -> None:
    provider_plan = LLMScoutPlan.model_validate(
        {
            "decision": "ready",
            "intent": {
                "kind": "player_comparison",
                "player_a": {"player_id": 1},
                "player_b": {"player_id": 1},
            },
            "calls": [
                {
                    "name": "compare_players",
                    "arguments": {
                        "player_a": {"player_id": 1},
                        "player_b": {"player_id": 1},
                    },
                }
            ],
        }
    )
    with pytest.raises(ValidationError):
        provider_plan.to_scout_plan()


def test_internal_policy_rejects_invalid_tool_intent_combination(
    ai_session: Session,
) -> None:
    provider_plan = LLMScoutPlan.model_validate(
        {
            "decision": "ready",
            "intent": {"kind": "player_search", "query": "Alice"},
            "calls": [{"name": "get_methodology", "arguments": {"topic": "xpass"}}],
        }
    )
    internal = provider_plan.to_scout_plan()
    prepared = prepare_plan(ai_session, internal)
    assert prepared.status is PlanPreparationStatus.INVALID
    assert "not valid for intent" in (prepared.error_message or "")


def test_provider_schema_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        LLMScoutPlan.model_validate(
            {
                "decision": "ready",
                "intent": {"kind": "player_search", "query": "Alice"},
                "calls": [{"name": "run_sql", "arguments": {}}],
            }
        )


def test_comparison_provider_shape_uses_named_players_not_fixed_tuple() -> None:
    schema = _provider_schema()
    comparison = schema["$defs"]["LLMPlannedComparePlayersInput"]
    assert set(comparison["properties"]) == {"player_a", "player_b"}
    assert "prefixItems" not in comparison
    assert ToolName.COMPARE_PLAYERS.value == "compare_players"
