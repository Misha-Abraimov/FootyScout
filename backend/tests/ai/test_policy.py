"""Governed AI Scout limits, caveats, registry, and import boundaries."""

import ast
import json
from collections import Counter
from pathlib import Path

from app.ai.policy import (
    LEADERBOARD_MAX_RESULTS,
    MAX_TOOL_CALLS,
    PLAYER_SEARCH_MAX_RESULTS,
    RECOMMENDATIONS_MAX_RESULTS,
    ROLE_FIT_LIMITATIONS,
    SIMILAR_PLAYERS_MAX_RESULTS,
    SIMILARITY_LIMITATIONS,
)
from app.ai.schemas import ToolName
from app.ai.tools.registry import TOOL_REGISTRY


def test_governed_limits_and_domain_caveats_are_explicit() -> None:
    assert PLAYER_SEARCH_MAX_RESULTS == 20
    assert SIMILAR_PLAYERS_MAX_RESULTS == 10
    assert LEADERBOARD_MAX_RESULTS == 25
    assert RECOMMENDATIONS_MAX_RESULTS == 20
    assert MAX_TOOL_CALLS == 6
    assert any("not player quality" in item for item in SIMILARITY_LIMITATIONS)
    assert ROLE_FIT_LIMITATIONS == (
        "Role Fit supports eligible outfield positions only.",
        "Role Fit does not predict transfer success or future performance.",
        "Role Fit is not lineup selection or a tactical guarantee.",
    )


def test_registry_is_fixed_and_contains_exactly_nine_tools() -> None:
    assert set(TOOL_REGISTRY) == set(ToolName) - {ToolName.SEARCH_WEB}
    assert len(TOOL_REGISTRY) == 9
    assert "not predict" in TOOL_REGISTRY[ToolName.GET_ROLE_FIT].description
    assert "Do not use" in TOOL_REGISTRY[ToolName.GET_SIMILAR_PLAYERS].description


def test_ai_package_never_imports_training_analytics() -> None:
    ai_root = Path(__file__).resolve().parents[2] / "app" / "ai"
    violations: list[str] = []
    for path in ai_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(
                name == "analytics" or name.startswith("analytics.") or ".analytics." in f".{name}."
                for name in names
            ):
                violations.append(f"{path.name}:{node.lineno}")
    assert violations == []


def test_golden_dataset_is_valid_unique_and_intentionally_distributed() -> None:
    path = Path(__file__).resolve().parents[2] / "evals" / "ai_scout_golden.jsonl"
    cases = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(cases) == 85
    assert len({case["id"] for case in cases}) == len(cases)
    assert Counter(case["category"] for case in cases) == {
        "entity_resolution": 12,
        "profile_comparison": 15,
        "similarity_archetype": 11,
        "team_role": 17,
        "methodology": 15,
        "unsupported_adversarial": 8,
        "current_world": 5,
        "mixed": 2,
    }
