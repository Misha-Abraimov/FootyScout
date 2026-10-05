"""Contracts that keep the monorepo deployment self-contained."""

import json
from pathlib import Path

import pytest

from app.runtime_metadata import (
    ACTION_VALUE_MODEL_METADATA_PATH,
    PASS_MODEL_METADATA_PATH,
    PLAYER_ARCHETYPE_METADATA_PATH,
    XG_MODEL_METADATA_PATH,
)
from app.team_feature_contract import (
    DESCRIPTIVE_FEATURES as API_DESCRIPTIVE_FEATURES,
)
from app.team_feature_contract import FIT_FEATURES as API_FIT_FEATURES

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_vercel_services_and_routes_are_declared() -> None:
    config = json.loads((REPOSITORY_ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert config["services"] == {
        "frontend": {"root": "frontend/", "framework": "nextjs"},
        "backend": {
            "root": "backend/",
            "framework": "fastapi",
            "entrypoint": "app.main:app",
        },
    }
    assert config["rewrites"] == [
        {"source": "/api/(.*)", "destination": {"service": "backend"}},
        {"source": "/health", "destination": {"service": "backend"}},
        {"source": "/docs", "destination": {"service": "backend"}},
        {"source": "/openapi.json", "destination": {"service": "backend"}},
        {"source": "/(.*)", "destination": {"service": "frontend"}},
    ]


@pytest.mark.parametrize(
    ("packaged_path", "source_name"),
    [
        (PASS_MODEL_METADATA_PATH, "pass_model_metadata.json"),
        (XG_MODEL_METADATA_PATH, "xg_model_metadata.json"),
        (ACTION_VALUE_MODEL_METADATA_PATH, "action_value_model_metadata.json"),
        (PLAYER_ARCHETYPE_METADATA_PATH, "player_archetype_metadata.json"),
    ],
)
def test_runtime_metadata_is_packaged_and_content_equivalent(
    packaged_path: Path,
    source_name: str,
) -> None:
    packaged = json.loads(packaged_path.read_text(encoding="utf-8"))
    assert isinstance(packaged, dict)

    ignored_source = REPOSITORY_ROOT / "models" / source_name
    if ignored_source.exists():
        source = json.loads(ignored_source.read_text(encoding="utf-8"))
        assert packaged == source


def test_runtime_team_feature_contract_matches_frozen_analytics_contract() -> None:
    from analytics.team_role_research import DESCRIPTIVE_FEATURES, FIT_FEATURES

    assert API_FIT_FEATURES == FIT_FEATURES
    assert API_DESCRIPTIVE_FEATURES == DESCRIPTIVE_FEATURES
