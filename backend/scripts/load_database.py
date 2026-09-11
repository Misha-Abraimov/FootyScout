"""Replace the PostgreSQL analytics snapshot from generated Parquet files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session, sessionmaker

from analytics.player_similarity import broad_position_group
from app.database import SessionLocal
from app.models import (
    AttackingAction,
    Pass,
    Player,
    PlayerArchetype,
    PlayerAttackingProfile,
    PlayerIntelligenceProfile,
    PlayerPercentile,
    PlayerProfile,
    PlayerRoleFit,
    PlayerShootingProfile,
    PlayerSimilarity,
    Shot,
    TeamRoleProfile,
    TeamStyleProfile,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "player_profiles.parquet"
)
DEFAULT_SIMILARITIES_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "player_similarities.parquet"
)
DEFAULT_OOF_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "pass_oof_predictions.parquet"
)
DEFAULT_FEATURES_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "pass_features.parquet"
)
DEFAULT_SHOTS_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "product_shot_predictions.parquet"
)
DEFAULT_SHOOTING_PROFILES_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "player_shooting_profiles.parquet"
)
DEFAULT_ACTIONS_PATH = REPOSITORY_ROOT / "data" / "processed" / "attacking_actions.parquet"
DEFAULT_ATTACKING_PROFILES_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "player_attacking_profiles.parquet"
)
DEFAULT_INTELLIGENCE_PROFILES_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "player_intelligence_profiles.parquet"
)
DEFAULT_PERCENTILES_PATH = REPOSITORY_ROOT / "data" / "processed" / "player_percentiles.parquet"
DEFAULT_ARCHETYPES_PATH = REPOSITORY_ROOT / "data" / "processed" / "player_archetypes.parquet"
DEFAULT_TEAM_STYLE_PATH = REPOSITORY_ROOT / "data" / "processed" / "team_style_profiles.parquet"
DEFAULT_TEAM_ROLES_PATH = REPOSITORY_ROOT / "data" / "processed" / "team_role_profiles.parquet"
DEFAULT_ROLE_FITS_PATH = REPOSITORY_ROOT / "data" / "processed" / "player_role_fits.parquet"
BULK_BATCH_SIZE = 5_000

PLAYER_SOURCE_COLUMNS = [
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "position",
    "matches_observed",
    "pass_attempts",
    "overall_reliable",
]
PROFILE_DATABASE_COLUMNS = [column.name for column in PlayerProfile.__table__.columns]
SIMILARITY_DATABASE_COLUMNS = [column.name for column in PlayerSimilarity.__table__.columns]
PASS_DATABASE_COLUMNS = [column.name for column in Pass.__table__.columns]
SHOT_DATABASE_COLUMNS = [column.name for column in Shot.__table__.columns]
SHOOTING_PROFILE_DATABASE_COLUMNS = [
    column.name for column in PlayerShootingProfile.__table__.columns
]
ACTION_DATABASE_COLUMNS = [column.name for column in AttackingAction.__table__.columns]
ATTACKING_PROFILE_DATABASE_COLUMNS = [
    column.name for column in PlayerAttackingProfile.__table__.columns
]
INTELLIGENCE_PROFILE_DATABASE_COLUMNS = [
    column.name for column in PlayerIntelligenceProfile.__table__.columns
]
PERCENTILE_DATABASE_COLUMNS = [column.name for column in PlayerPercentile.__table__.columns]
ARCHETYPE_DATABASE_COLUMNS = [column.name for column in PlayerArchetype.__table__.columns]
TEAM_STYLE_DATABASE_COLUMNS = [column.name for column in TeamStyleProfile.__table__.columns]
TEAM_ROLE_DATABASE_COLUMNS = [column.name for column in TeamRoleProfile.__table__.columns]
ROLE_FIT_DATABASE_COLUMNS = [column.name for column in PlayerRoleFit.__table__.columns]

OOF_REQUIRED_COLUMNS = {
    "pass_index",
    "match_id",
    "player_id",
    "team_id",
    "position",
    "completed",
    "expected_completion",
    "fold",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "pass_length",
    "forward_distance",
    "under_pressure",
    "progressive",
}
FEATURE_REQUIRED_COLUMNS = set(PASS_DATABASE_COLUMNS).difference(
    {"pass_index", "expected_completion", "fold"}
)


@dataclass(frozen=True)
class DatabaseFrames:
    players: pd.DataFrame
    player_profiles: pd.DataFrame
    passes: pd.DataFrame
    player_similarities: pd.DataFrame
    shots: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_shooting_profiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    attacking_actions: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_attacking_profiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_intelligence_profiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_percentiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_archetypes: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_style_profiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_role_profiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_role_fits: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def counts(self) -> dict[str, int]:
        counts = {
            "players": len(self.players),
            "player_profiles": len(self.player_profiles),
            "passes": len(self.passes),
            "player_similarities": len(self.player_similarities),
        }
        if not self.shots.empty or not self.player_shooting_profiles.empty:
            counts["shots"] = len(self.shots)
            counts["player_shooting_profiles"] = len(self.player_shooting_profiles)
        if not self.attacking_actions.empty or not self.player_attacking_profiles.empty:
            counts["attacking_actions"] = len(self.attacking_actions)
            counts["player_attacking_profiles"] = len(self.player_attacking_profiles)
        if not self.player_intelligence_profiles.empty or not self.player_percentiles.empty:
            counts["player_intelligence_profiles"] = len(self.player_intelligence_profiles)
            counts["player_percentiles"] = len(self.player_percentiles)
        if not self.player_archetypes.empty:
            counts["player_archetypes"] = len(self.player_archetypes)
        if not self.team_style_profiles.empty:
            counts["team_style_profiles"] = len(self.team_style_profiles)
        if not self.team_role_profiles.empty:
            counts["team_role_profiles"] = len(self.team_role_profiles)
        if not self.player_role_fits.empty:
            counts["player_role_fits"] = len(self.player_role_fits)
        return counts


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _validate_pass_alignment(features: pd.DataFrame, oof: pd.DataFrame) -> None:
    """Prove reconstructed feature pass indices align with the OOF snapshot."""
    if len(features) != len(oof):
        raise ValueError("Pass feature and OOF files must have identical row counts")
    if oof["pass_index"].isna().any() or oof["pass_index"].duplicated().any():
        raise ValueError("OOF pass_index values must be complete and unique")
    expected_indices = set(range(len(features)))
    if set(oof["pass_index"].astype(int)) != expected_indices:
        raise ValueError("OOF pass_index values do not match the feature row positions")

    ordered_oof = oof.sort_values("pass_index").reset_index(drop=True)
    exact_columns = [
        "match_id",
        "player_id",
        "team_id",
        "position",
        "completed",
        "under_pressure",
        "progressive",
    ]
    numeric_columns = [
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "pass_length",
        "forward_distance",
    ]
    for column in exact_columns:
        left = features[column].astype("string").fillna("<NULL>")
        right = ordered_oof[column].astype("string").fillna("<NULL>")
        if not left.equals(right):
            raise ValueError(f"Pass alignment failed for {column}")
    for column in numeric_columns:
        left = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(ordered_oof[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(left, right, equal_nan=True):
            raise ValueError(f"Pass alignment failed for {column}")


def transform_source_frames(
    profiles: pd.DataFrame,
    similarities: pd.DataFrame,
    oof: pd.DataFrame,
    features: pd.DataFrame,
    shots: pd.DataFrame | None = None,
    shooting_profiles: pd.DataFrame | None = None,
    attacking_actions: pd.DataFrame | None = None,
    attacking_profiles: pd.DataFrame | None = None,
    intelligence_profiles: pd.DataFrame | None = None,
    percentiles: pd.DataFrame | None = None,
    archetypes: pd.DataFrame | None = None,
    team_style_profiles: pd.DataFrame | None = None,
    team_role_profiles: pd.DataFrame | None = None,
    player_role_fits: pd.DataFrame | None = None,
) -> DatabaseFrames:
    """Validate analytics frames and shape them for ORM bulk insertion."""
    _require_columns(profiles, set(PLAYER_SOURCE_COLUMNS + PROFILE_DATABASE_COLUMNS), "profiles")
    _require_columns(similarities, set(SIMILARITY_DATABASE_COLUMNS), "similarities")
    _require_columns(oof, OOF_REQUIRED_COLUMNS, "OOF predictions")
    _require_columns(features, FEATURE_REQUIRED_COLUMNS, "pass features")

    if profiles["player_id"].isna().any() or profiles["player_id"].duplicated().any():
        raise ValueError("Profiles must contain exactly one row per player_id")
    if similarities.duplicated(["player_id", "similar_player_id"]).any():
        raise ValueError("Similarity player pairs must be unique")
    if similarities.duplicated(["player_id", "rank"]).any():
        raise ValueError("Similarity ranks must be unique within each player")
    if similarities["player_id"].eq(similarities["similar_player_id"]).any():
        raise ValueError("Similarity rows cannot recommend a player to themself")
    if not similarities["same_position_group"].all() or not similarities[
        "position_group"
    ].eq(similarities["similar_position_group"]).all():
        raise ValueError("V3.3B similarities must remain within one broad position")
    if similarities["position_group"].eq("GK").any():
        raise ValueError("V3.3B similarity excludes goalkeepers")
    if not similarities["methodology_version"].eq("V3.3B").all():
        raise ValueError("Similarity snapshot must use methodology V3.3B")
    expected_pair_support = similarities[
        ["query_matches_observed", "candidate_matches_observed"]
    ].min(axis=1)
    if not similarities["pair_support_matches"].eq(expected_pair_support).all():
        raise ValueError("Similarity pair support must use the weaker observed-match count")
    if oof["expected_completion"].isna().any() or not oof[
        "expected_completion"
    ].between(0.0, 1.0).all():
        raise ValueError("OOF expected_completion must remain within [0, 1]")

    players = profiles[PLAYER_SOURCE_COLUMNS].copy()
    players.insert(5, "position_group", players["position"].map(broad_position_group))
    players = players[[column.name for column in Player.__table__.columns]]
    valid_player_ids = set(players["player_id"].astype(int))

    profile_rows = profiles[PROFILE_DATABASE_COLUMNS].copy()
    similarity_rows = similarities[SIMILARITY_DATABASE_COLUMNS].copy()
    similarity_ids = set(similarity_rows["player_id"].astype(int)) | set(
        similarity_rows["similar_player_id"].astype(int)
    )
    if not similarity_ids.issubset(valid_player_ids):
        raise ValueError("Every similarity must reference a valid profile player")
    player_groups = players.set_index("player_id")["position_group"]
    query_groups = similarity_rows["player_id"].map(player_groups)
    candidate_groups = similarity_rows["similar_player_id"].map(player_groups)
    if not query_groups.eq(similarity_rows["position_group"]).all() or not candidate_groups.eq(
        similarity_rows["similar_position_group"]
    ).all():
        raise ValueError("Similarity position groups must match referenced players")

    pass_player_ids = set(oof["player_id"].dropna().astype(int))
    if not pass_player_ids.issubset(valid_player_ids):
        raise ValueError("Every non-null pass player_id must reference a valid profile player")
    _validate_pass_alignment(features.reset_index(drop=True), oof)
    indexed_features = features.reset_index(drop=True).copy()
    indexed_features.insert(0, "pass_index", np.arange(len(indexed_features), dtype=np.int64))
    passes = indexed_features.merge(
        oof[["pass_index", "expected_completion", "fold"]],
        on="pass_index",
        how="left",
        validate="one_to_one",
    )[PASS_DATABASE_COLUMNS]
    if passes["pass_index"].duplicated().any():
        raise ValueError("Pass rows must be unique by pass_index")
    shot_rows = pd.DataFrame(columns=SHOT_DATABASE_COLUMNS)
    shooting_profile_rows = pd.DataFrame(columns=SHOOTING_PROFILE_DATABASE_COLUMNS)
    if (shots is None) != (shooting_profiles is None):
        raise ValueError("Shots and shooting profiles must be supplied together")
    if shots is not None and shooting_profiles is not None:
        shot_source_columns = {
            "shot_id",
            "match_id",
            "player_id",
            "team_id",
            "period",
            "minute",
            "second",
            "shot_x",
            "shot_y",
            "distance_to_goal",
            "angle_to_goal",
            "goal",
            "expected_goal",
            "body_part",
            "shot_type",
            "technique",
            "play_pattern",
            "under_pressure",
            "first_time",
            "one_on_one",
            "open_goal",
            "penalty",
            "penalty_shootout",
            "model_eligible",
        }
        _require_columns(shots, shot_source_columns, "product shots")
        _require_columns(
            shooting_profiles,
            set(SHOOTING_PROFILE_DATABASE_COLUMNS),
            "shooting profiles",
        )
        if shots["shot_id"].isna().any() or shots["shot_id"].duplicated().any():
            raise ValueError("Product shots must have unique non-null shot_id values")
        eligible = shots["model_eligible"].astype(bool)
        if shots.loc[eligible, "expected_goal"].isna().any() or not shots.loc[
            eligible, "expected_goal"
        ].between(0, 1).all():
            raise ValueError("Eligible product shots require expected_goal within [0, 1]")
        shot_rows = shots.rename(
            columns={
                "shot_x": "start_x",
                "shot_y": "start_y",
                "distance_to_goal": "distance",
                "angle_to_goal": "angle",
            }
        )[SHOT_DATABASE_COLUMNS].copy()
        invalid_players = ~shot_rows["player_id"].isin(valid_player_ids)
        shot_rows.loc[invalid_players, "player_id"] = pd.NA
        shooting_profile_rows = shooting_profiles.loc[
            shooting_profiles["player_id"].isin(valid_player_ids),
            SHOOTING_PROFILE_DATABASE_COLUMNS,
        ].copy()
        if shooting_profile_rows["player_id"].duplicated().any():
            raise ValueError("Shooting profiles must contain one row per player")

    action_rows = pd.DataFrame(columns=ACTION_DATABASE_COLUMNS)
    attacking_profile_rows = pd.DataFrame(columns=ATTACKING_PROFILE_DATABASE_COLUMNS)
    if (attacking_actions is None) != (attacking_profiles is None):
        raise ValueError("Attacking actions and profiles must be supplied together")
    if attacking_actions is not None and attacking_profiles is not None:
        _require_columns(attacking_actions, set(ACTION_DATABASE_COLUMNS), "attacking actions")
        _require_columns(
            attacking_profiles,
            set(ATTACKING_PROFILE_DATABASE_COLUMNS),
            "attacking profiles",
        )
        if attacking_actions["action_id"].isna().any() or attacking_actions[
            "action_id"
        ].duplicated().any():
            raise ValueError("Attacking actions require unique non-null action IDs")
        if attacking_actions[["state_value_before", "state_value_after"]].lt(0).any().any():
            raise ValueError("Attacking state values must be non-negative")
        action_rows = attacking_actions[ACTION_DATABASE_COLUMNS].copy()
        action_rows.loc[~action_rows["player_id"].isin(valid_player_ids), "player_id"] = pd.NA
        attacking_profile_rows = attacking_profiles.loc[
            attacking_profiles["player_id"].isin(valid_player_ids),
            ATTACKING_PROFILE_DATABASE_COLUMNS,
        ].copy()
        if attacking_profile_rows["player_id"].duplicated().any():
            raise ValueError("Attacking profiles must contain one row per player")

    intelligence_rows = pd.DataFrame(columns=INTELLIGENCE_PROFILE_DATABASE_COLUMNS)
    percentile_rows = pd.DataFrame(columns=PERCENTILE_DATABASE_COLUMNS)
    if (intelligence_profiles is None) != (percentiles is None):
        raise ValueError("Intelligence profiles and percentiles must be supplied together")
    if intelligence_profiles is not None and percentiles is not None:
        _require_columns(
            intelligence_profiles,
            set(INTELLIGENCE_PROFILE_DATABASE_COLUMNS),
            "intelligence profiles",
        )
        _require_columns(percentiles, set(PERCENTILE_DATABASE_COLUMNS), "player percentiles")
        intelligence_rows = intelligence_profiles[
            INTELLIGENCE_PROFILE_DATABASE_COLUMNS
        ].copy()
        percentile_rows = percentiles[PERCENTILE_DATABASE_COLUMNS].copy()
        if intelligence_rows["player_id"].duplicated().any():
            raise ValueError("Intelligence profiles must contain one row per player")
        if set(intelligence_rows["player_id"].astype(int)) != valid_player_ids:
            raise ValueError("Intelligence profiles must cover every product player exactly once")
        if percentile_rows.duplicated(["player_id", "metric_name"]).any():
            raise ValueError("Player percentile metrics must be unique")
        if not set(percentile_rows["player_id"].astype(int)).issubset(valid_player_ids):
            raise ValueError("Every percentile must reference a valid product player")
        percentile_values = percentile_rows["percentile"].dropna()
        if not percentile_values.between(0, 100).all():
            raise ValueError("Player percentile values must remain within [0, 100]")

    archetype_rows = pd.DataFrame(columns=ARCHETYPE_DATABASE_COLUMNS)
    if archetypes is not None:
        _require_columns(archetypes, set(ARCHETYPE_DATABASE_COLUMNS), "player archetypes")
        archetype_rows = archetypes[ARCHETYPE_DATABASE_COLUMNS].copy()
        if archetype_rows["player_id"].isna().any() or archetype_rows[
            "player_id"
        ].duplicated().any():
            raise ValueError("Player archetypes require unique, non-null player IDs")
        if not set(archetype_rows["player_id"].astype(int)).issubset(valid_player_ids):
            raise ValueError("Every archetype must reference a valid product player")
        if not archetype_rows["eligible"].eq(True).all():
            raise ValueError("Only eligible production archetype rows may be persisted")
        if not archetype_rows["position_group"].isin(["DEF", "MID", "FWD"]).all():
            raise ValueError("Production archetypes are restricted to outfield players")
        if not archetype_rows["archetype_id"].isin(
            ["direct_progressor", "safe_circulator"]
        ).all():
            raise ValueError("Unknown semantic archetype ID")
        if not archetype_rows["separation_margin"].between(0, 1).all():
            raise ValueError("Archetype separation must remain within [0, 1]")
        if not archetype_rows["second_centroid_distance"].ge(
            archetype_rows["centroid_distance"]
        ).all():
            raise ValueError("Assigned centroid must be the nearest centroid")

    team_style_rows = pd.DataFrame(columns=TEAM_STYLE_DATABASE_COLUMNS)
    team_role_rows = pd.DataFrame(columns=TEAM_ROLE_DATABASE_COLUMNS)
    role_fit_rows = pd.DataFrame(columns=ROLE_FIT_DATABASE_COLUMNS)
    supplied_v4 = [team_style_profiles, team_role_profiles, player_role_fits]
    if any(frame is not None for frame in supplied_v4) and not all(
        frame is not None for frame in supplied_v4
    ):
        raise ValueError("All three V4 production frames must be supplied together")
    if all(frame is not None for frame in supplied_v4):
        assert team_style_profiles is not None
        assert team_role_profiles is not None
        assert player_role_fits is not None
        _require_columns(team_style_profiles, set(TEAM_STYLE_DATABASE_COLUMNS), "team styles")
        _require_columns(team_role_profiles, set(TEAM_ROLE_DATABASE_COLUMNS), "team roles")
        _require_columns(player_role_fits, set(ROLE_FIT_DATABASE_COLUMNS), "player role fits")
        team_style_rows = team_style_profiles[TEAM_STYLE_DATABASE_COLUMNS].copy()
        team_role_rows = team_role_profiles[TEAM_ROLE_DATABASE_COLUMNS].copy()
        role_fit_rows = player_role_fits[ROLE_FIT_DATABASE_COLUMNS].copy()
        if team_style_rows["team_id"].duplicated().any():
            raise ValueError("Team style profiles must be unique by team_id")
        if team_role_rows.duplicated(["team_id", "position_group"]).any():
            raise ValueError("Team role profiles must be unique by team and position")
        if role_fit_rows.duplicated(["target_team_id", "player_id"]).any():
            raise ValueError("Player Role Fits must be unique by target team and player")
        if not set(role_fit_rows["player_id"].astype(int)).issubset(valid_player_ids):
            raise ValueError("Every Role Fit must reference a valid player")
        external = role_fit_rows.loc[~role_fit_rows["is_target_team_player"].astype(bool)]
        if external.duplicated(
            ["target_team_id", "position_group", "recommendation_rank"]
        ).any():
            raise ValueError("External recommendation ranks must be unique")
        if not external["sample_support"].eq("limited").all():
            raise ValueError("Current external Role Fit rows must visibly retain limited support")

    return DatabaseFrames(
        players=players,
        player_profiles=profile_rows,
        passes=passes,
        player_similarities=similarity_rows,
        shots=shot_rows,
        player_shooting_profiles=shooting_profile_rows,
        attacking_actions=action_rows,
        player_attacking_profiles=attacking_profile_rows,
        player_intelligence_profiles=intelligence_rows,
        player_percentiles=percentile_rows,
        player_archetypes=archetype_rows,
        team_style_profiles=team_style_rows,
        team_role_profiles=team_role_rows,
        player_role_fits=role_fit_rows,
    )


def load_source_files(
    profiles_path: Path,
    similarities_path: Path,
    oof_path: Path,
    features_path: Path,
    shots_path: Path | None = None,
    shooting_profiles_path: Path | None = None,
    actions_path: Path | None = None,
    attacking_profiles_path: Path | None = None,
    intelligence_profiles_path: Path | None = None,
    percentiles_path: Path | None = None,
    archetypes_path: Path | None = None,
    team_style_path: Path | None = None,
    team_roles_path: Path | None = None,
    role_fits_path: Path | None = None,
) -> DatabaseFrames:
    """Read and validate all inputs before opening a database transaction."""
    paths = {
        "player profiles": profiles_path,
        "player similarities": similarities_path,
        "OOF predictions": oof_path,
        "pass features": features_path,
    }
    missing = [f"{label}: {path}" for label, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    if (shots_path is None) != (shooting_profiles_path is None):
        raise ValueError("Both xG source paths must be supplied together")
    if shots_path is not None and shooting_profiles_path is not None:
        xg_paths = {
            "product shots": shots_path,
            "shooting profiles": shooting_profiles_path,
        }
        xg_missing = [f"{label}: {path}" for label, path in xg_paths.items() if not path.is_file()]
        if xg_missing:
            raise FileNotFoundError("Missing required xG input files:\n" + "\n".join(xg_missing))
    if (actions_path is None) != (attacking_profiles_path is None):
        raise ValueError("Both action-value source paths must be supplied together")
    if actions_path is not None and attacking_profiles_path is not None:
        value_paths = {
            "attacking actions": actions_path,
            "attacking profiles": attacking_profiles_path,
        }
        value_missing = [
            f"{label}: {path}" for label, path in value_paths.items() if not path.is_file()
        ]
        if value_missing:
            raise FileNotFoundError(
                "Missing required action-value input files:\n" + "\n".join(value_missing)
            )
    if (intelligence_profiles_path is None) != (percentiles_path is None):
        raise ValueError("Both V3.1 source paths must be supplied together")
    if intelligence_profiles_path is not None and percentiles_path is not None:
        intelligence_paths = {
            "player intelligence profiles": intelligence_profiles_path,
            "player percentiles": percentiles_path,
        }
        intelligence_missing = [
            f"{label}: {path}"
            for label, path in intelligence_paths.items()
            if not path.is_file()
        ]
        if intelligence_missing:
            raise FileNotFoundError(
                "Missing required V3.1 input files:\n" + "\n".join(intelligence_missing)
            )
    if archetypes_path is not None and not archetypes_path.is_file():
        raise FileNotFoundError(f"Missing player archetypes: {archetypes_path}")
    v4_paths = [team_style_path, team_roles_path, role_fits_path]
    if any(path is not None for path in v4_paths) and not all(
        path is not None for path in v4_paths
    ):
        raise ValueError("All three V4 production paths must be supplied together")
    for path in v4_paths:
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Missing V4 production artifact: {path}")
    return transform_source_frames(
        pd.read_parquet(profiles_path),
        pd.read_parquet(similarities_path),
        pd.read_parquet(oof_path),
        pd.read_parquet(features_path),
        pd.read_parquet(shots_path) if shots_path is not None else None,
        pd.read_parquet(shooting_profiles_path)
        if shooting_profiles_path is not None
        else None,
        pd.read_parquet(actions_path) if actions_path is not None else None,
        pd.read_parquet(attacking_profiles_path)
        if attacking_profiles_path is not None
        else None,
        pd.read_parquet(intelligence_profiles_path)
        if intelligence_profiles_path is not None
        else None,
        pd.read_parquet(percentiles_path) if percentiles_path is not None else None,
        pd.read_parquet(archetypes_path) if archetypes_path is not None else None,
        pd.read_parquet(team_style_path) if team_style_path is not None else None,
        pd.read_parquet(team_roles_path) if team_roles_path is not None else None,
        pd.read_parquet(role_fits_path) if role_fits_path is not None else None,
    )


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas missing/scalar values to database-safe Python values."""
    records: list[dict[str, Any]] = []
    for raw_record in frame.astype(object).where(frame.notna(), None).to_dict("records"):
        records.append(
            {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in raw_record.items()
            }
        )
    return records


def _bulk_insert(session: Session, model: type[Any], frame: pd.DataFrame) -> None:
    records = dataframe_records(frame)
    for start in range(0, len(records), BULK_BATCH_SIZE):
        session.execute(
            insert(model),
            records[start : start + BULK_BATCH_SIZE],
            execution_options={"render_nulls": True},
        )


def replace_database_snapshot(
    frames: DatabaseFrames,
    session_factory: sessionmaker[Session] = SessionLocal,
) -> dict[str, int]:
    """Atomically delete and reload the deterministic analytics snapshot."""
    expected_counts = frames.counts
    with session_factory.begin() as session:
        session.execute(delete(PlayerRoleFit))
        session.execute(delete(TeamRoleProfile))
        session.execute(delete(TeamStyleProfile))
        session.execute(delete(PlayerArchetype))
        session.execute(delete(PlayerPercentile))
        session.execute(delete(PlayerIntelligenceProfile))
        session.execute(delete(PlayerAttackingProfile))
        session.execute(delete(AttackingAction))
        session.execute(delete(PlayerShootingProfile))
        session.execute(delete(Shot))
        session.execute(delete(PlayerSimilarity))
        session.execute(delete(Pass))
        session.execute(delete(PlayerProfile))
        session.execute(delete(Player))

        _bulk_insert(session, Player, frames.players)
        _bulk_insert(session, PlayerProfile, frames.player_profiles)
        _bulk_insert(session, Pass, frames.passes)
        _bulk_insert(session, PlayerSimilarity, frames.player_similarities)
        if not frames.shots.empty:
            _bulk_insert(session, Shot, frames.shots)
        if not frames.player_shooting_profiles.empty:
            _bulk_insert(session, PlayerShootingProfile, frames.player_shooting_profiles)
        if not frames.attacking_actions.empty:
            _bulk_insert(session, AttackingAction, frames.attacking_actions)
        if not frames.player_attacking_profiles.empty:
            _bulk_insert(session, PlayerAttackingProfile, frames.player_attacking_profiles)
        if not frames.player_intelligence_profiles.empty:
            _bulk_insert(
                session, PlayerIntelligenceProfile, frames.player_intelligence_profiles
            )
        if not frames.player_percentiles.empty:
            _bulk_insert(session, PlayerPercentile, frames.player_percentiles)
        if not frames.player_archetypes.empty:
            _bulk_insert(session, PlayerArchetype, frames.player_archetypes)
        if not frames.team_style_profiles.empty:
            _bulk_insert(session, TeamStyleProfile, frames.team_style_profiles)
        if not frames.team_role_profiles.empty:
            _bulk_insert(session, TeamRoleProfile, frames.team_role_profiles)
        if not frames.player_role_fits.empty:
            _bulk_insert(session, PlayerRoleFit, frames.player_role_fits)

        actual_counts = {
            "players": session.scalar(select(func.count()).select_from(Player)) or 0,
            "player_profiles": session.scalar(select(func.count()).select_from(PlayerProfile))
            or 0,
            "passes": session.scalar(select(func.count()).select_from(Pass)) or 0,
            "player_similarities": session.scalar(
                select(func.count()).select_from(PlayerSimilarity)
            )
            or 0,
        }
        if "shots" in expected_counts:
            actual_counts["shots"] = session.scalar(select(func.count()).select_from(Shot)) or 0
            actual_counts["player_shooting_profiles"] = (
                session.scalar(select(func.count()).select_from(PlayerShootingProfile)) or 0
            )
        if "attacking_actions" in expected_counts:
            actual_counts["attacking_actions"] = (
                session.scalar(select(func.count()).select_from(AttackingAction)) or 0
            )
            actual_counts["player_attacking_profiles"] = (
                session.scalar(select(func.count()).select_from(PlayerAttackingProfile)) or 0
            )
        if "player_intelligence_profiles" in expected_counts:
            actual_counts["player_intelligence_profiles"] = (
                session.scalar(select(func.count()).select_from(PlayerIntelligenceProfile)) or 0
            )
            actual_counts["player_percentiles"] = (
                session.scalar(select(func.count()).select_from(PlayerPercentile)) or 0
            )
        if "player_archetypes" in expected_counts:
            actual_counts["player_archetypes"] = (
                session.scalar(select(func.count()).select_from(PlayerArchetype)) or 0
            )
        if "team_style_profiles" in expected_counts:
            actual_counts["team_style_profiles"] = (
                session.scalar(select(func.count()).select_from(TeamStyleProfile)) or 0
            )
        if "team_role_profiles" in expected_counts:
            actual_counts["team_role_profiles"] = (
                session.scalar(select(func.count()).select_from(TeamRoleProfile)) or 0
            )
        if "player_role_fits" in expected_counts:
            actual_counts["player_role_fits"] = (
                session.scalar(select(func.count()).select_from(PlayerRoleFit)) or 0
            )
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Database counts do not match source counts: "
                f"expected={expected_counts}, actual={actual_counts}"
            )
    return actual_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--similarities", type=Path, default=DEFAULT_SIMILARITIES_PATH)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--shots", type=Path, default=DEFAULT_SHOTS_PATH)
    parser.add_argument(
        "--shooting-profiles", type=Path, default=DEFAULT_SHOOTING_PROFILES_PATH
    )
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS_PATH)
    parser.add_argument(
        "--attacking-profiles", type=Path, default=DEFAULT_ATTACKING_PROFILES_PATH
    )
    parser.add_argument(
        "--intelligence-profiles", type=Path, default=DEFAULT_INTELLIGENCE_PROFILES_PATH
    )
    parser.add_argument("--percentiles", type=Path, default=DEFAULT_PERCENTILES_PATH)
    parser.add_argument("--archetypes", type=Path, default=DEFAULT_ARCHETYPES_PATH)
    parser.add_argument("--team-style", type=Path, default=DEFAULT_TEAM_STYLE_PATH)
    parser.add_argument("--team-roles", type=Path, default=DEFAULT_TEAM_ROLES_PATH)
    parser.add_argument("--role-fits", type=Path, default=DEFAULT_ROLE_FITS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = load_source_files(
        args.profiles,
        args.similarities,
        args.oof,
        args.features,
        args.shots,
        args.shooting_profiles,
        args.actions,
        args.attacking_profiles,
        args.intelligence_profiles,
        args.percentiles,
        args.archetypes,
        args.team_style,
        args.team_roles,
        args.role_fits,
    )
    print("Validated source rows before database mutation:")
    for table, count in frames.counts.items():
        print(f"  {table} = {count:,}")
    counts = replace_database_snapshot(frames)
    print("PostgreSQL snapshot load complete:")
    for table, count in counts.items():
        print(f"  {table} = {count:,}")


if __name__ == "__main__":
    main()
