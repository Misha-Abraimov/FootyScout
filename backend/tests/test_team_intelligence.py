from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.player_similarity import SIMILARITY_FEATURE_COLUMNS
from analytics.team_intelligence import FIT_COLUMNS, ROLE_COLUMNS, TEAM_COLUMNS
from analytics.team_role_research import FIT_FEATURES

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
pytestmark = pytest.mark.artifacts


def test_production_v4_artifact_counts_and_schema() -> None:
    teams = pd.read_parquet(PROCESSED / "team_style_profiles.parquet")
    roles = pd.read_parquet(PROCESSED / "team_role_profiles.parquet")
    fits = pd.read_parquet(PROCESSED / "player_role_fits.parquet")
    assert list(teams.columns) == TEAM_COLUMNS
    assert list(roles.columns) == ROLE_COLUMNS
    assert list(fits.columns) == FIT_COLUMNS
    assert len(teams) == 1
    assert len(roles) == 3
    assert len(fits) == 133
    assert teams.iloc[0]["team_id"] == 904
    assert teams.iloc[0]["matches_observed"] == 34
    assert set(roles["position_group"]) == {"DEF", "MID", "FWD"}


def test_role_fit_contract_ranking_and_support_invariants() -> None:
    fits = pd.read_parquet(PROCESSED / "player_role_fits.parquet")
    assert list(FIT_FEATURES) == SIMILARITY_FEATURE_COLUMNS
    assert fits["player_id"].is_unique
    assert fits["role_distance"].ge(0).all()
    assert not fits["position_group"].eq("GK").any()
    external = fits.loc[~fits["is_target_team_player"]]
    assert len(external) == 113
    assert external["sample_support"].eq("limited").all()
    assert external["player_team_name"].ne("Bayer Leverkusen").all()
    for _, group in external.groupby("position_group"):
        ordered = group.sort_values("recommendation_rank")
        assert ordered["role_distance"].is_monotonic_increasing
        assert ordered["recommendation_rank"].tolist() == list(range(1, len(group) + 1))


def test_role_fit_explanations_match_stored_gaps_and_rms() -> None:
    fits = pd.read_parquet(PROCESSED / "player_role_fits.parquet")
    for row in fits.itertuples(index=False):
        gaps = json.loads(row.feature_gaps)
        ordered = sorted(FIT_FEATURES, key=lambda feature: (gaps[feature], FIT_FEATURES.index(feature)))
        assert [row.closest_feature_1, row.closest_feature_2, row.closest_feature_3] == ordered[:3]
        assert row.largest_difference == ordered[-1]
        assert row.role_distance == pytest.approx(
            float(np.sqrt(np.mean([gaps[feature] ** 2 for feature in FIT_FEATURES])))
        )
        contributions = json.loads(row.distance_contributions)
        assert sum(contributions.values()) == pytest.approx(1.0)


def test_current_players_use_leave_self_out_and_fwd_warning_is_persisted() -> None:
    fits = pd.read_parquet(PROCESSED / "player_role_fits.parquet")
    roles = pd.read_parquet(PROCESSED / "team_role_profiles.parquet")
    current = fits.loc[fits["is_target_team_player"]]
    assert len(current) == 20
    assert current["calculation_scope"].eq("leave_self_out_target_role").all()
    assert current["recommendation_rank"].isna().all()
    for row in current.itertuples(index=False):
        assert str(row.role_contributor_count) in row.role_support_message
        assert "other" in row.role_support_message
    fwd = roles.loc[roles["position_group"].eq("FWD")].iloc[0]
    assert fwd["contributor_count"] == 3
    assert fwd["matches_observed"] == 33
    assert "Limited contributor diversity" in fwd["support_message"]


def test_no_performance_archetype_or_identity_field_enters_fit_vector() -> None:
    forbidden = {
        "completion_above_expected_pp", "goals_minus_xg",
        "attacking_value_per_100_actions", "archetype_id", "player_id", "team_id",
    }
    assert forbidden.isdisjoint(FIT_FEATURES)
