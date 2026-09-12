"""Deployment-safe paths for frozen, API-facing model metadata snapshots."""

from pathlib import Path

RUNTIME_METADATA_DIR = Path(__file__).resolve().parent / "runtime_metadata"

PASS_MODEL_METADATA_PATH = RUNTIME_METADATA_DIR / "pass_model_metadata.json"
XG_MODEL_METADATA_PATH = RUNTIME_METADATA_DIR / "xg_model_metadata.json"
ACTION_VALUE_MODEL_METADATA_PATH = RUNTIME_METADATA_DIR / "action_value_model_metadata.json"
PLAYER_ARCHETYPE_METADATA_PATH = RUNTIME_METADATA_DIR / "player_archetype_metadata.json"
