"""Build production V4 team intelligence, role profiles, and Role Fit artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analytics.artifact_paths import repository_relative_path
from analytics.player_similarity import (
    HIGHER_SUPPORT_MIN_MATCHES,
    SIMILARITY_FEATURE_COLUMNS,
    prepare_similarity_pool,
)
from analytics.team_role_research import (
    DESCRIPTIVE_FEATURES,
    FIT_FEATURES,
    ROLE_GROUPS,
    _aggregate_profiles,
    enrich_actions,
    normalize_profile,
    validate_fit_contract,
    vector_distance,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

DEFAULT_TEAM_RESEARCH = PROCESSED / "leverkusen_team_style_research_v40.parquet"
DEFAULT_ROLE_RESEARCH = PROCESSED / "leverkusen_role_profiles_v40.parquet"
DEFAULT_FIT_RESEARCH = PROCESSED / "player_role_fit_research_v40.parquet"
DEFAULT_RESEARCH_METADATA = MODELS / "v40_team_role_research.json"
DEFAULT_PLAYERS = PROCESSED / "player_similarity_features_v3.parquet"
DEFAULT_PASSES = PROCESSED / "pass_oof_predictions.parquet"
DEFAULT_ACTIONS = PROCESSED / "attacking_actions.parquet"
DEFAULT_SHOTS = PROCESSED / "product_shot_predictions.parquet"
DEFAULT_ARCHETYPES = PROCESSED / "player_archetypes.parquet"

DEFAULT_TEAM_OUTPUT = PROCESSED / "team_style_profiles.parquet"
DEFAULT_ROLE_OUTPUT = PROCESSED / "team_role_profiles.parquet"
DEFAULT_FIT_OUTPUT = PROCESSED / "player_role_fits.parquet"
DEFAULT_METADATA_OUTPUT = MODELS / "team_intelligence_metadata.json"

TEAM_VERSION = "V4.1"
ROLE_VERSION = "V4.2"
FIT_VERSION = "V4.3"
RECOMMENDATION_VERSION = "V4.4"
TARGET_TEAM_ID = 904
TARGET_TEAM_NAME = "Bayer Leverkusen"

TEAM_COLUMNS = [
    "team_id", "team_name", "methodology_version", "sample_scope",
    "matches_observed", "contributors", "passes", "carries", "actions", "shots",
    *DESCRIPTIVE_FEATURES,
]
ROLE_COLUMNS = [
    "team_id", "team_name", "position_group", "methodology_version",
    "aggregation_method", "matches_observed", "contributor_count", "contributors",
    "passes", "carries", "actions", "shots", "support_level", "support_message",
    *FIT_FEATURES, *[f"{feature}_z" for feature in FIT_FEATURES],
]
FIT_COLUMNS = [
    "target_team_id", "target_team_name", "player_id", "player_name", "player_team_name",
    "position", "position_group", "is_target_team_player", "calculation_scope",
    "role_distance", "recommendation_rank", "closest_feature_1", "closest_feature_2",
    "closest_feature_3", "largest_difference", "feature_gaps", "distance_contributions",
    "player_matches_observed", "player_pass_attempts", "player_carries", "sample_support",
    "sample_support_message", "role_matches_observed", "role_contributor_count",
    "role_actions", "role_support_message", "archetype_id", "archetype_name",
    "methodology_version",
]


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _python_value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _explain_fit(player_z: np.ndarray, role_z: np.ndarray) -> dict[str, Any]:
    gaps = np.abs(player_z - role_z)
    squared = np.square(gaps)
    total = float(squared.sum())
    order = np.argsort(gaps, kind="stable")
    return {
        "closest_feature_1": FIT_FEATURES[int(order[0])],
        "closest_feature_2": FIT_FEATURES[int(order[1])],
        "closest_feature_3": FIT_FEATURES[int(order[2])],
        "largest_difference": FIT_FEATURES[int(order[-1])],
        "feature_gaps": json.dumps(
            {feature: float(value) for feature, value in zip(FIT_FEATURES, gaps, strict=True)},
            sort_keys=True,
        ),
        "distance_contributions": json.dumps(
            {
                feature: float(value / total) if total else 0.0
                for feature, value in zip(FIT_FEATURES, squared, strict=True)
            },
            sort_keys=True,
        ),
    }


def _sample_support(matches: int) -> tuple[str, str]:
    if matches < HIGHER_SUPPORT_MIN_MATCHES:
        return "limited", f"Limited sample: observed across {matches} match{'es' if matches != 1 else ''}."
    return "higher", f"Higher sample support: observed across {matches} matches."


def _leave_self_out_support_message(
    position_group: str,
    matches_observed: int,
    contributor_count: int,
) -> str:
    if position_group == "FWD":
        return (
            "Limited contributor diversity: this leave-self-out role profile is based "
            f"on {contributor_count} other contributing forwards across "
            f"{matches_observed} observed matches."
        )
    return (
        f"Leave-self-out role observed across {matches_observed} matches and "
        f"{contributor_count} other {position_group} contributors."
    )


def build_team_profile(
    team_research: pd.DataFrame,
    target_team_id: int,
    target_team_name: str,
) -> pd.DataFrame:
    source = team_research.loc[team_research["analysis_type"].eq("full")]
    if len(source) != 1:
        raise ValueError("Production team profile requires exactly one full research row")
    row = source.iloc[0]
    record: dict[str, Any] = {
        "team_id": target_team_id,
        "team_name": target_team_name,
        "methodology_version": TEAM_VERSION,
        "sample_scope": (
            "Observed team style across the 34 Bundesliga matches available in the "
            "FootyScout product sample."
        ),
        "matches_observed": int(row["matches_observed"]),
        "contributors": int(row["action_players"]),
        "passes": int(row["pass_attempts"]),
        "carries": int(row["carries"]),
        "actions": int(row["attacking_actions"]),
        "shots": int(row["shots"]),
    }
    record.update({feature: float(row[feature]) for feature in DESCRIPTIVE_FEATURES})
    result = pd.DataFrame([record], columns=TEAM_COLUMNS)
    if result.iloc[0]["matches_observed"] != 34:
        raise AssertionError("Only the complete 34-match target profile may enter production")
    return result


def build_role_profiles(
    role_research: pd.DataFrame,
    research_metadata: dict[str, Any],
    target_team_id: int,
    target_team_name: str,
) -> pd.DataFrame:
    source = role_research.loc[role_research["analysis_type"].eq("full")].copy()
    if set(source["position_group"]) != set(ROLE_GROUPS) or len(source) != 3:
        raise ValueError("Production requires one full DEF, MID, and FWD role")
    contribution_rows = pd.DataFrame(research_metadata["role_contribution_rows"])
    records = []
    for row in source.itertuples(index=False):
        group = str(row.position_group)
        contributors = contribution_rows.loc[
            contribution_rows["position_group"].eq(group)
        ].sort_values(["actions", "player_id"], ascending=[False, True])
        limited = group == "FWD"
        record: dict[str, Any] = {
            "team_id": target_team_id,
            "team_name": target_team_name,
            "position_group": group,
            "methodology_version": ROLE_VERSION,
            "aggregation_method": "pooled_events_actions",
            "matches_observed": int(row.matches_observed),
            "contributor_count": int(row.action_players),
            "contributors": json.dumps(
                [
                    {
                        "player_id": int(player.player_id),
                        "player_name": str(player.player_name),
                        "actions": int(player.actions),
                        "action_share": float(player.action_share),
                    }
                    for player in contributors.itertuples(index=False)
                ],
                ensure_ascii=False,
            ),
            "passes": int(row.pass_attempts),
            "carries": int(row.carries),
            "actions": int(row.attacking_actions),
            "shots": int(row.shots),
            "support_level": "limited_contributor_diversity" if limited else "established",
            "support_message": (
                "Limited contributor diversity: this role profile is based on three "
                "contributing forwards across 33 observed matches and is more sensitive "
                "to individual-player composition than the DEF and MID profiles."
                if limited
                else f"Observed across {int(row.matches_observed)} matches and "
                f"{int(row.action_players)} contributing {group} players."
            ),
        }
        for feature in FIT_FEATURES:
            record[feature] = float(getattr(row, feature))
            record[f"{feature}_z"] = float(getattr(row, f"{feature}_z"))
        records.append(record)
    return pd.DataFrame(records, columns=ROLE_COLUMNS).sort_values("position_group")


def _target_player_profiles(
    passes: pd.DataFrame,
    actions: pd.DataFrame,
    shots: pd.DataFrame,
    players: pd.DataFrame,
    target_team_id: int,
) -> pd.DataFrame:
    profiles = _aggregate_profiles(
        passes.loc[passes["team_id"].eq(target_team_id)],
        actions.loc[actions["team_id"].eq(target_team_id)],
        ["player_id"],
        shots.loc[shots["team_id"].eq(target_team_id)],
    )
    metadata = players[
        ["player_id", "player_name", "team_name", "position", "position_group"]
    ].drop_duplicates("player_id")
    return profiles.merge(metadata, on="player_id", how="left", validate="one_to_one")


def build_player_role_fits(
    fit_research: pd.DataFrame,
    role_research: pd.DataFrame,
    role_profiles: pd.DataFrame,
    players: pd.DataFrame,
    target_player_profiles: pd.DataFrame,
    archetypes: pd.DataFrame,
    statistics: dict[str, dict[str, dict[str, float]]],
    target_team_id: int,
    target_team_name: str,
) -> pd.DataFrame:
    eligible = prepare_similarity_pool(players)
    eligible_ids = set(eligible["player_id"].astype(int))
    player_lookup = eligible.set_index("player_id")
    archetype_lookup = archetypes.set_index("player_id")
    role_lookup = role_profiles.set_index("position_group")
    records: list[dict[str, Any]] = []

    external = fit_research.loc[
        fit_research["analysis_type"].eq("external_fit")
        & fit_research["distance_method"].eq("rms")
    ]
    for row in external.itertuples(index=False):
        player_id = int(row.player_id)
        if player_id not in eligible_ids:
            raise AssertionError("Ineligible player entered external Role Fit")
        player = player_lookup.loc[player_id]
        role = role_lookup.loc[str(row.position_group)]
        support, support_message = _sample_support(int(player.matches_observed))
        archetype = archetype_lookup.loc[player_id] if player_id in archetype_lookup.index else None
        records.append(
            {
                "target_team_id": target_team_id,
                "target_team_name": target_team_name,
                "player_id": player_id,
                "player_name": str(player.player_name),
                "player_team_name": str(player.team_name),
                "position": str(player.position),
                "position_group": str(player.position_group),
                "is_target_team_player": False,
                "calculation_scope": "full_target_role",
                "role_distance": float(row.distance),
                "recommendation_rank": int(row.rank),
                "closest_feature_1": str(row.closest_feature_1),
                "closest_feature_2": str(row.closest_feature_2),
                "closest_feature_3": str(row.closest_feature_3),
                "largest_difference": str(row.largest_gap_feature),
                "feature_gaps": str(row.feature_gaps),
                "distance_contributions": str(row.distance_contributions),
                "player_matches_observed": int(player.matches_observed),
                "player_pass_attempts": int(player.pass_attempts),
                "player_carries": int(player.carries),
                "sample_support": support,
                "sample_support_message": support_message,
                "role_matches_observed": int(role.matches_observed),
                "role_contributor_count": int(role.contributor_count),
                "role_actions": int(role.actions),
                "role_support_message": str(role.support_message),
                "archetype_id": None if archetype is None else str(archetype.archetype_id),
                "archetype_name": None if archetype is None else str(archetype.archetype_name),
                "methodology_version": FIT_VERSION,
            }
        )

    lopo = role_research.loc[role_research["analysis_type"].eq("leave_one_player_out")]
    target_lookup = target_player_profiles.set_index("player_id")
    for role in lopo.itertuples(index=False):
        player_id = int(role.excluded_player_id)
        if player_id not in eligible_ids:
            continue
        player = target_lookup.loc[player_id]
        group = str(role.position_group)
        player_z = normalize_profile(player, group, statistics)
        role_z = np.array([float(getattr(role, f"{feature}_z")) for feature in FIT_FEATURES])
        explanation = _explain_fit(player_z, role_z)
        distance = vector_distance(player_z, role_z, "rms")
        support, support_message = _sample_support(int(player.matches_observed))
        archetype = archetype_lookup.loc[player_id] if player_id in archetype_lookup.index else None
        records.append(
            {
                "target_team_id": target_team_id,
                "target_team_name": target_team_name,
                "player_id": player_id,
                "player_name": str(player.player_name),
                "player_team_name": target_team_name,
                "position": str(player.position),
                "position_group": group,
                "is_target_team_player": True,
                "calculation_scope": "leave_self_out_target_role",
                "role_distance": distance,
                "recommendation_rank": None,
                **explanation,
                "player_matches_observed": int(player.matches_observed),
                "player_pass_attempts": int(player.pass_attempts),
                "player_carries": int(player.carries),
                "sample_support": support,
                "sample_support_message": support_message,
                "role_matches_observed": int(role.matches_observed),
                "role_contributor_count": int(role.action_players),
                "role_actions": int(role.attacking_actions),
                "role_support_message": _leave_self_out_support_message(
                    group,
                    int(role.matches_observed),
                    int(role.action_players),
                ),
                "archetype_id": None if archetype is None else str(archetype.archetype_id),
                "archetype_name": None if archetype is None else str(archetype.archetype_name),
                "methodology_version": FIT_VERSION,
            }
        )

    result = pd.DataFrame(records, columns=FIT_COLUMNS)
    result = result.sort_values(
        ["is_target_team_player", "position_group", "recommendation_rank", "player_id"],
        kind="stable",
    ).reset_index(drop=True)
    validate_player_role_fits(result, eligible, target_team_name)
    return result


def validate_player_role_fits(
    fits: pd.DataFrame, eligible: pd.DataFrame, target_team_name: str
) -> None:
    validate_fit_contract()
    if list(FIT_FEATURES) != SIMILARITY_FEATURE_COLUMNS:
        raise AssertionError("V4 Role Fit diverged from V3.3B")
    if fits["player_id"].duplicated().any() or len(fits) != len(eligible):
        raise AssertionError("Every eligible player must receive exactly one Role Fit row")
    if not fits["position_group"].isin(ROLE_GROUPS).all():
        raise AssertionError("Goalkeepers cannot enter Role Fit")
    if not fits["role_distance"].ge(0).all():
        raise AssertionError("Role distance must be nonnegative")
    external = fits.loc[~fits["is_target_team_player"]]
    if external["player_team_name"].eq(target_team_name).any():
        raise AssertionError("Current target-team players entered recommendations")
    if not external["sample_support"].eq("limited").all():
        raise AssertionError("Every current external candidate must expose limited support")
    if external.duplicated(["position_group", "recommendation_rank"]).any():
        raise AssertionError("Recommendation ranks must be unique by position")
    for _, group in external.groupby("position_group", sort=False):
        ordered = group.sort_values("recommendation_rank")
        if not ordered["role_distance"].is_monotonic_increasing:
            raise AssertionError("Recommendations must rank ascending pure RMS distance")
    current = fits.loc[fits["is_target_team_player"]]
    if not current["calculation_scope"].eq("leave_self_out_target_role").all():
        raise AssertionError("Current players require leave-self-out roles")


def run_pipeline(
    team_research_path: Path = DEFAULT_TEAM_RESEARCH,
    role_research_path: Path = DEFAULT_ROLE_RESEARCH,
    fit_research_path: Path = DEFAULT_FIT_RESEARCH,
    research_metadata_path: Path = DEFAULT_RESEARCH_METADATA,
    players_path: Path = DEFAULT_PLAYERS,
    passes_path: Path = DEFAULT_PASSES,
    actions_path: Path = DEFAULT_ACTIONS,
    shots_path: Path = DEFAULT_SHOTS,
    archetypes_path: Path = DEFAULT_ARCHETYPES,
    team_output: Path = DEFAULT_TEAM_OUTPUT,
    role_output: Path = DEFAULT_ROLE_OUTPUT,
    fit_output: Path = DEFAULT_FIT_OUTPUT,
    metadata_output: Path = DEFAULT_METADATA_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    research = json.loads(research_metadata_path.read_text(encoding="utf-8"))
    if research["methodology_version"] != "V4.0":
        raise ValueError("Production V4 requires the frozen V4.0 research metadata")
    team_research = pd.read_parquet(team_research_path)
    role_research = pd.read_parquet(role_research_path)
    fit_research = pd.read_parquet(fit_research_path)
    players = pd.read_parquet(players_path)
    passes = pd.read_parquet(passes_path)
    actions = enrich_actions(pd.read_parquet(actions_path), players)
    shots = pd.read_parquet(shots_path)
    archetypes = pd.read_parquet(archetypes_path)
    positions = players[["player_id", "position_group"]].drop_duplicates("player_id")
    passes = passes.merge(positions, on="player_id", how="left", validate="many_to_one")
    shots = shots.merge(positions, on="player_id", how="left", validate="many_to_one")

    team = build_team_profile(team_research, TARGET_TEAM_ID, TARGET_TEAM_NAME)
    roles = build_role_profiles(role_research, research, TARGET_TEAM_ID, TARGET_TEAM_NAME)
    target_players = _target_player_profiles(
        passes, actions, shots, players, TARGET_TEAM_ID
    )
    statistics = research["normalization"]["statistics"]
    fits = build_player_role_fits(
        fit_research, role_research, roles, players, target_players, archetypes,
        statistics, TARGET_TEAM_ID, TARGET_TEAM_NAME,
    )
    external = fits.loc[~fits["is_target_team_player"]]
    metadata = {
        "methodology_versions": {
            "team_style": TEAM_VERSION,
            "position_roles": ROLE_VERSION,
            "role_fit": FIT_VERSION,
            "scouting_recommendations": RECOMMENDATION_VERSION,
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_team": {"team_id": TARGET_TEAM_ID, "team_name": TARGET_TEAM_NAME},
        "sample_scope": team.iloc[0]["sample_scope"],
        "qualified_team_ids": [TARGET_TEAM_ID],
        "team_descriptive_metrics": list(DESCRIPTIVE_FEATURES),
        "role_fit_features": list(FIT_FEATURES),
        "role_aggregation": "pooled events/actions by team_id and frozen primary position_group",
        "normalization": {
            "reference": "V3.3B eligible-player broad-position population means/stds",
            "statistics": statistics,
        },
        "distance": {
            "method": "RMS Euclidean",
            "formula": "sqrt(mean((player_position_z - role_position_z)^2))",
            "direction": "lower is closer",
            "display_score": None,
        },
        "eligibility": {
            "position_groups": list(ROLE_GROUPS),
            "minimum_passes": 50,
            "minimum_carries": 29,
            "goalkeepers": "unavailable",
        },
        "sample_support": {
            "limited": "fewer than 3 observed player matches",
            "higher": "at least 3 observed player matches",
            "changes_distance_or_rank": False,
        },
        "interpretation": (
            "Role Fit measures how closely a player's observed playing style resembles "
            "Bayer Leverkusen's observed positional-role style. It does not predict "
            "transfer success or future performance."
        ),
        "external_calculation": "full Leverkusen position role",
        "current_player_calculation": "Leverkusen role excluding that player's actions",
        "recommendation_rule": (
            "eligible external same-position players sorted by ascending RMS role distance, "
            "then player_id"
        ),
        "role_support_warnings": {
            str(row.position_group): str(row.support_message)
            for row in roles.itertuples(index=False)
        },
        "counts": {
            "team_profiles": len(team),
            "role_profiles": len(roles),
            "player_role_fits": len(fits),
            "external_candidates": len(external),
            "current_player_calibrations": int(fits["is_target_team_player"].sum()),
            "external_by_position": {
                group: int(external["position_group"].eq(group).sum()) for group in ROLE_GROUPS
            },
            "external_sample_support": {
                str(key): int(value) for key, value in external["sample_support"].value_counts().items()
            },
        },
        "excluded_from_fit_and_ranking": [
            "performance metrics", "outcomes", "archetype", "identity", "team preference",
            "sample support", "player fame",
        ],
        "artifacts": {
            "team_style_profiles": repository_relative_path(team_output),
            "team_role_profiles": repository_relative_path(role_output),
            "player_role_fits": repository_relative_path(fit_output),
        },
    }
    _atomic_parquet(team, team_output)
    _atomic_parquet(roles, role_output)
    _atomic_parquet(fits, fit_output)
    _atomic_json(metadata, metadata_output)
    print("FootyScout production V4 artifacts complete")
    print(f"Team profiles: {len(team):,}")
    print(f"Role profiles: {len(roles):,}")
    print(f"Player Role Fits: {len(fits):,}")
    print(f"External candidates: {len(external):,}")
    return team, roles, fits, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-output", type=Path, default=DEFAULT_TEAM_OUTPUT)
    parser.add_argument("--role-output", type=Path, default=DEFAULT_ROLE_OUTPUT)
    parser.add_argument("--fit-output", type=Path, default=DEFAULT_FIT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        team_output=args.team_output,
        role_output=args.role_output,
        fit_output=args.fit_output,
        metadata_output=args.metadata_output,
    )


if __name__ == "__main__":
    main()
