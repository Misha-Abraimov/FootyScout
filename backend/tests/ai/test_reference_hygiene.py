"""Conservative punctuation normalization for entity and search references."""

import pytest
from sqlalchemy.orm import Session

from app.ai.plan_normalizer import prepare_plan, sanitize_reference_text
from app.ai.schemas import ScoutPlan


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("xHaKa.", "xHaKa"),
        ("Smith?", "Smith"),
        ("O'Connor", "O'Connor"),
        ("Jean-Pierre", "Jean-Pierre"),
        ("São Paulo", "São Paulo"),
        ("U.S.", "U.S."),
    ],
)
def test_sanitize_reference_text(raw: str, expected: str) -> None:
    assert sanitize_reference_text(raw) == expected


def test_player_reference_is_sanitized_before_resolution(ai_session: Session) -> None:
    plan = ScoutPlan.model_validate(
        {
            "decision": "ready",
            "intent": {
                "kind": "player_profile",
                "player": {"player_name": "Alice Playmaker."},
                "sections": ["passing"],
            },
            "calls": [
                {
                    "name": "get_player_dossier",
                    "arguments": {
                        "player": {"player_name": "Alice Playmaker."},
                        "sections": ["passing"],
                    },
                }
            ],
        }
    )
    prepared = prepare_plan(ai_session, plan)
    assert prepared.normalized_plan is not None
    assert prepared.normalized_plan.calls[0].arguments["player_id"] == 1
    assert prepared.player_resolutions[0].query == "Alice Playmaker"


def test_search_reference_is_sanitized_in_normalized_arguments(ai_session: Session) -> None:
    plan = ScoutPlan.model_validate(
        {
            "decision": "ready",
            "intent": {"kind": "player_search", "query": "Alice?"},
            "calls": [
                {
                    "name": "search_players",
                    "arguments": {"query": "Alice?", "limit": 5},
                }
            ],
        }
    )
    prepared = prepare_plan(ai_session, plan)
    assert prepared.normalized_plan is not None
    assert prepared.normalized_plan.calls[0].arguments["query"] == "Alice"
