from __future__ import annotations

import copy

import pandas as pd
import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import configure_mappers

from analytics.player_similarity import broad_position_group
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
from scripts.load_database import dataframe_records, transform_source_frames


def sample_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profile_template: dict[str, object] = {
        "player_name": "Player One",
        "team_id": 100,
        "team_name": "Team One",
        "position": "Center Back",
        "matches_observed": 1,
        "pass_attempts": 1,
        "overall_reliable": False,
        "passes_completed": 1,
        "actual_completion_rate": 1.0,
        "expected_completions": 0.8,
        "expected_completion_rate": 0.8,
        "completions_above_expected": 0.2,
        "completion_above_expected_pp": 20.0,
        "pressure_attempts": 0,
        "pressure_completed": 0,
        "pressure_actual_completion_rate": None,
        "pressure_expected_completion_rate": None,
        "pressure_completions_above_expected": None,
        "pressure_above_expected_pp": None,
        "pressure_pass_rate": 0.0,
        "progressive_attempts": 0,
        "progressive_completed": 0,
        "progressive_actual_completion_rate": None,
        "progressive_expected_completion_rate": None,
        "progressive_completions_above_expected": None,
        "progressive_above_expected_pp": None,
        "progressive_pass_rate": 0.0,
        "long_pass_attempts": 0,
        "long_pass_completed": 0,
        "long_pass_actual_completion_rate": None,
        "long_pass_expected_completion_rate": None,
        "long_pass_completions_above_expected": None,
        "long_pass_above_expected_pp": None,
        "average_forward_distance": 10.0,
        "net_forward_distance_per_100_passes": 1000.0,
        "positive_forward_distance_per_100_passes": 1000.0,
        "final_third_entries": 0,
        "final_third_entries_per_100_passes": 0.0,
        "pressure_reliable": False,
        "progressive_reliable": False,
        "long_pass_reliable": False,
    }
    first_profile = {"player_id": 1, **profile_template}
    second_profile = {
        "player_id": 2,
        **profile_template,
        "player_name": "Player Two",
        "team_id": 200,
        "team_name": "Team Two",
        "position": "Left Back",
    }
    profiles = pd.DataFrame([first_profile, second_profile])
    similarities = pd.DataFrame(
        [
            {
                "player_id": 1,
                "similar_player_id": 2,
                "rank": 1,
                "rms_distance": 0.4,
                "similarity_score": 87.5,
                "same_position_group": True,
                "position_group": "DEF",
                "similar_position_group": "DEF",
                "closest_feature_1": "long_pass_rate",
                "closest_feature_2": "pressure_pass_rate",
                "closest_feature_3": "expected_completion_rate",
                "query_matches_observed": 5,
                "candidate_matches_observed": 2,
                "pair_support_matches": 2,
                "sample_support": "limited",
                "sample_support_explanation": "Limited sample for one profile.",
                "methodology_version": "V3.3B",
                "distance_contributions": "{}",
            }
        ]
    )
    feature_rows = []
    oof_rows = []
    for pass_index, profile in enumerate((first_profile, second_profile)):
        feature = {
            "match_id": 10,
            "player_id": profile["player_id"],
            "team_id": profile["team_id"],
            "position": profile["position"],
            "completed": 1,
            "start_x": 20.0,
            "start_y": 30.0,
            "end_x": 40.0,
            "end_y": 35.0,
            "pass_length": 20.62,
            "pass_angle": 0.24,
            "forward_distance": 20.0,
            "lateral_distance": 5.0,
            "distance_to_goal_before": 100.5,
            "distance_to_goal_after": 80.15,
            "distance_toward_goal": 20.35,
            "under_pressure": False,
            "progressive": False,
            "pass_height": "Ground Pass",
            "body_part": "Right Foot",
            "pass_type": "Regular",
            "start_zone": "defensive_centre",
            "end_zone": "middle_centre",
        }
        feature_rows.append(feature)
        oof_rows.append(
            {
                "pass_index": pass_index,
                **{
                    column: feature[column]
                    for column in (
                        "match_id",
                        "player_id",
                        "team_id",
                        "position",
                        "completed",
                        "start_x",
                        "start_y",
                        "end_x",
                        "end_y",
                        "pass_length",
                        "forward_distance",
                        "under_pressure",
                        "progressive",
                    )
                },
                "expected_completion": 0.8,
                "fold": 1,
            }
        )
    return profiles, similarities, pd.DataFrame(oof_rows), pd.DataFrame(feature_rows)


def test_orm_relationships_and_one_to_one_profile() -> None:
    configure_mappers()

    assert set(inspect(Player).relationships.keys()) == {
        "profile",
        "passes",
        "shots",
        "shooting_profile",
        "similarities",
        "similar_to",
        "attacking_actions",
        "attacking_profile",
        "intelligence_profile",
        "percentiles",
        "archetype",
        "role_fits",
    }
    assert not inspect(Player).relationships["profile"].uselist
    assert "player" in inspect(PlayerProfile).relationships
    assert {"player", "similar_player"}.issubset(
        inspect(PlayerSimilarity).relationships.keys()
    )
    assert "player" in inspect(Pass).relationships
    assert "player" in inspect(Shot).relationships
    assert "player" in inspect(PlayerShootingProfile).relationships
    assert "player" in inspect(AttackingAction).relationships
    assert "player" in inspect(PlayerAttackingProfile).relationships
    assert "player" in inspect(PlayerIntelligenceProfile).relationships
    assert "player" in inspect(PlayerPercentile).relationships
    assert "player" in inspect(PlayerArchetype).relationships


def test_primary_foreign_keys_constraints_and_indexes() -> None:
    assert [column.name for column in Player.__table__.primary_key] == ["player_id"]
    assert [column.name for column in PlayerProfile.__table__.primary_key] == ["player_id"]
    assert {column.name for column in PlayerSimilarity.__table__.primary_key} == {
        "player_id",
        "similar_player_id",
    }
    assert [column.name for column in Pass.__table__.primary_key] == ["pass_index"]
    assert [column.name for column in AttackingAction.__table__.primary_key] == [
        "action_id"
    ]
    assert [column.name for column in PlayerAttackingProfile.__table__.primary_key] == [
        "player_id"
    ]
    assert [column.name for column in PlayerIntelligenceProfile.__table__.primary_key] == [
        "player_id"
    ]
    assert {column.name for column in PlayerPercentile.__table__.primary_key} == {
        "player_id", "metric_name",
    }
    assert [column.name for column in PlayerArchetype.__table__.primary_key] == [
        "player_id"
    ]
    assert [column.name for column in TeamStyleProfile.__table__.primary_key] == ["team_id"]
    assert {column.name for column in TeamRoleProfile.__table__.primary_key} == {
        "team_id", "position_group",
    }
    assert {column.name for column in PlayerRoleFit.__table__.primary_key} == {
        "target_team_id", "player_id",
    }

    similarity_constraints = {constraint.name for constraint in PlayerSimilarity.__table__.constraints}
    pass_constraints = {constraint.name for constraint in Pass.__table__.constraints}
    assert "ck_similarities_not_self" in similarity_constraints
    assert "uq_similarities_player_rank" in similarity_constraints
    assert "ck_similarities_score_range" in similarity_constraints
    assert "ck_passes_expected_completion" in pass_constraints

    player_indexes = {index.name for index in Player.__table__.indexes}
    pass_indexes = {index.name for index in Pass.__table__.indexes}
    similarity_indexes = {index.name for index in PlayerSimilarity.__table__.indexes}
    assert player_indexes == {
        "ix_players_player_name",
        "ix_players_position_group",
        "ix_players_team_name",
    }
    assert pass_indexes == {
        "ix_passes_match_id",
        "ix_passes_player_id",
        "ix_passes_player_match",
    }
    assert similarity_indexes == {
        "ix_player_similarities_player_id",
        "ix_player_similarities_similar_player_id",
    }
    assert {index.name for index in AttackingAction.__table__.indexes} == {
        "ix_attacking_actions_action_type",
        "ix_attacking_actions_attacking_value",
        "ix_attacking_actions_match_id",
        "ix_attacking_actions_pass_index",
        "ix_attacking_actions_player_id",
        "ix_attacking_actions_possession_id",
    }
    assert {index.name for index in PlayerPercentile.__table__.indexes} == {
        "ix_player_percentiles_metric_name",
        "ix_player_percentiles_player_id",
        "ix_player_percentiles_position_group",
    }
    assert {index.name for index in PlayerArchetype.__table__.indexes} == {
        "ix_player_archetypes_archetype_id",
        "ix_player_archetypes_position_group",
        "ix_player_archetypes_raw_cluster",
    }


def test_v3_intelligence_frames_are_persistable_and_normalized() -> None:
    profiles, similarities, oof, features = sample_sources()
    intelligence = profiles[["player_id", "matches_observed"]].copy()
    intelligence["position_group"] = intelligence["player_id"].map({1: "DEF", 2: "DEF"})
    percentiles = pd.DataFrame([
        {
            "player_id": 1,
            "metric_name": "progressive_pass_rate",
            "family": "style",
            "raw_value": 0.2,
            "percentile": 75.0,
            "peer_position_group": "DEF",
            "peer_count": 12,
            "sample_count": 120,
            "eligible": True,
            "eligibility_reason": "eligible",
        }
    ])
    frames = transform_source_frames(
        profiles,
        similarities,
        oof,
        features,
        intelligence_profiles=intelligence,
        percentiles=percentiles,
    )
    assert frames.counts["player_intelligence_profiles"] == 2
    assert frames.counts["player_percentiles"] == 1
    assert frames.player_percentiles.loc[0, "percentile"] == 75.0


def test_v3_archetype_frame_is_validated_and_persistable() -> None:
    profiles, similarities, oof, features = sample_sources()
    archetypes = pd.DataFrame(
        [
            {
                "player_id": 1,
                "archetype_id": "direct_progressor",
                "archetype_name": "Direct Progressor",
                "raw_cluster_id": 0,
                "position_group": "DEF",
                "centroid_distance": 0.8,
                "second_centroid_distance": 2.0,
                "separation_margin": 0.6,
                "eligible": True,
                "model_version": "V3.2C",
                "expected_completion_rate_position_z": -0.5,
                "pressure_pass_rate_position_z": -0.1,
                "progressive_pass_rate_position_z": 0.7,
                "long_pass_rate_position_z": 0.6,
                "positive_forward_distance_per_100_passes_position_z": 0.8,
                "carry_share_of_actions_position_z": -0.3,
            }
        ]
    )
    frames = transform_source_frames(
        profiles, similarities, oof, features, archetypes=archetypes
    )
    assert frames.counts["player_archetypes"] == 1
    assert frames.player_archetypes.loc[0, "archetype_id"] == "direct_progressor"

    invalid = archetypes.copy()
    invalid.loc[0, "position_group"] = "GK"
    with pytest.raises(ValueError, match="outfield"):
        transform_source_frames(
            profiles, similarities, oof, features, archetypes=invalid
        )


def test_v4_frames_are_validated_and_persistable() -> None:
    profiles, similarities, oof, features = sample_sources()
    team_style = pd.DataFrame(
        [{column.name: (904 if column.name == "team_id" else 1)
          for column in TeamStyleProfile.__table__.columns}],
        dtype=object,
    )
    team_style.loc[0, "team_name"] = "Bayer Leverkusen"
    team_style.loc[0, "methodology_version"] = "V4.1"
    team_style.loc[0, "sample_scope"] = "34-match product sample"
    team_roles = pd.DataFrame(
        [{column.name: (904 if column.name == "team_id" else 1)
          for column in TeamRoleProfile.__table__.columns}],
        dtype=object,
    )
    team_roles.loc[0, "team_name"] = "Bayer Leverkusen"
    team_roles.loc[0, "position_group"] = "DEF"
    team_roles.loc[0, "methodology_version"] = "V4.2"
    team_roles.loc[0, "aggregation_method"] = "pooled_events_actions"
    team_roles.loc[0, "contributors"] = "[]"
    team_roles.loc[0, "support_level"] = "established"
    team_roles.loc[0, "support_message"] = "Observed role"
    role_fits = pd.DataFrame(
        [{column.name: (904 if column.name == "target_team_id" else 1)
          for column in PlayerRoleFit.__table__.columns}],
        dtype=object,
    )
    for column, value in {
        "target_team_name": "Bayer Leverkusen", "player_name": "Player One",
        "player_team_name": "Team One", "position": "Center Back",
        "position_group": "DEF", "is_target_team_player": False,
        "calculation_scope": "full_target_role", "recommendation_rank": 1,
        "closest_feature_1": "long_pass_rate",
        "closest_feature_2": "pressure_pass_rate",
        "closest_feature_3": "expected_completion_rate",
        "largest_difference": "carry_share_of_actions", "feature_gaps": "{}",
        "distance_contributions": "{}", "sample_support": "limited",
        "sample_support_message": "Limited", "role_support_message": "Observed role",
        "methodology_version": "V4.3", "archetype_id": None, "archetype_name": None,
    }.items():
        role_fits.loc[0, column] = value
    frames = transform_source_frames(
        profiles, similarities, oof, features,
        team_style_profiles=team_style,
        team_role_profiles=team_roles,
        player_role_fits=role_fits,
    )
    assert frames.counts["team_style_profiles"] == 1
    assert frames.counts["team_role_profiles"] == 1
    assert frames.counts["player_role_fits"] == 1


def test_parquet_to_database_transformation_and_position_mapping() -> None:
    frames = transform_source_frames(*sample_sources())

    assert frames.counts == {
        "players": 2,
        "player_profiles": 2,
        "passes": 2,
        "player_similarities": 1,
    }
    players = frames.players.set_index("player_id")
    assert players.loc[1, "position_group"] == broad_position_group("Center Back") == "DEF"
    assert players.loc[2, "position_group"] == broad_position_group("Left Back") == "DEF"
    assert frames.passes["expected_completion"].tolist() == [0.8, 0.8]
    assert frames.passes["pass_index"].tolist() == [0, 1]


def test_null_subset_metrics_become_python_none() -> None:
    frames = transform_source_frames(*sample_sources())
    records = dataframe_records(frames.player_profiles)

    assert records[0]["pressure_actual_completion_rate"] is None
    assert records[0]["progressive_above_expected_pp"] is None
    assert records[0]["long_pass_completions_above_expected"] is None


def test_action_value_frames_are_validated_and_preserve_nullable_player_links() -> None:
    actions = pd.DataFrame(
        [
            {
                "action_id": "action-1",
                "pass_index": 0,
                "match_id": 10,
                "possession_id": 7,
                "event_index": 20,
                "player_id": 999,
                "team_id": 100,
                "action_type": "Pass",
                "start_x": 20.0,
                "start_y": 30.0,
                "end_x": 40.0,
                "end_y": 35.0,
                "state_value_before": 0.02,
                "state_value_after": 0.08,
                "attacking_value": 0.06,
                "success": True,
                "under_pressure": False,
                "progressive": True,
                "expected_completion": 0.8,
                "pass_risk": 0.2,
                "risk_reward_category": "routine_high_value",
                "fold": 1,
            }
        ]
    )
    attacking_profiles = pd.DataFrame(
        [
            {
                "player_id": 1,
                "matches_observed": 1,
                "actions": 1,
                "passes": 1,
                "carries": 0,
                "total_attacking_value": 0.06,
                "attacking_value_per_100_actions": 6.0,
                "total_pass_value": 0.06,
                "pass_value_per_100_passes": 6.0,
                "total_carry_value": 0.0,
                "carry_value_per_100_carries": None,
                "positive_value_actions": 1,
                "positive_value_action_rate": 1.0,
                "progressive_action_value": 0.06,
                "progressive_value_per_100_actions": 6.0,
                "pressure_action_value": 0.0,
                "pressure_value_per_100_actions": 0.0,
                "attacking_value_reliable": False,
                "pass_value_reliable": False,
                "carry_value_reliable": False,
            }
        ]
    )

    frames = transform_source_frames(
        *sample_sources(),
        attacking_actions=actions,
        attacking_profiles=attacking_profiles,
    )

    assert frames.counts["attacking_actions"] == 1
    assert frames.counts["player_attacking_profiles"] == 1
    assert pd.isna(frames.attacking_actions.loc[0, "player_id"])
    assert frames.attacking_actions.loc[0, "pass_index"] == 0


def test_loader_rejects_duplicate_pass_indices() -> None:
    profiles, similarities, oof, features = sample_sources()
    invalid_oof = copy.deepcopy(oof)
    invalid_oof.loc[1, "pass_index"] = 0

    with pytest.raises(ValueError, match="pass_index"):
        transform_source_frames(profiles, similarities, invalid_oof, features)


def test_loader_rejects_invalid_similarity_reference() -> None:
    profiles, similarities, oof, features = sample_sources()
    invalid_similarities = copy.deepcopy(similarities)
    invalid_similarities.loc[0, "similar_player_id"] = 999

    with pytest.raises(ValueError, match="valid profile player"):
        transform_source_frames(profiles, invalid_similarities, oof, features)


def test_loader_rejects_misaligned_pass_features() -> None:
    profiles, similarities, oof, features = sample_sources()
    invalid_features = copy.deepcopy(features)
    invalid_features.loc[1, "match_id"] = 999

    with pytest.raises(ValueError, match="alignment"):
        transform_source_frames(profiles, similarities, oof, invalid_features)
