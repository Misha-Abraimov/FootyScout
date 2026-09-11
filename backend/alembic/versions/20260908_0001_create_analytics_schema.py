"""Create FootyScout analytics persistence schema.

Revision ID: 20260908_0001
Revises:
Create Date: 2026-09-08
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260908_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "players",
        sa.Column(
            "player_id", sa.BigInteger(), autoincrement=False, nullable=False
        ),
        sa.Column("player_name", sa.String(length=200), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("team_name", sa.String(length=200), nullable=False),
        sa.Column("position", sa.String(length=100), nullable=False),
        sa.Column("position_group", sa.String(length=3), nullable=False),
        sa.Column("matches_observed", sa.Integer(), nullable=False),
        sa.Column("pass_attempts", sa.Integer(), nullable=False),
        sa.Column("overall_reliable", sa.Boolean(), nullable=False),
        sa.CheckConstraint("matches_observed >= 0", name="ck_players_matches_nonnegative"),
        sa.CheckConstraint("pass_attempts >= 0", name="ck_players_attempts_nonnegative"),
        sa.CheckConstraint(
            "position_group IN ('GK', 'DEF', 'MID', 'FWD')",
            name="ck_players_position_group",
        ),
        sa.PrimaryKeyConstraint("player_id"),
    )
    op.create_index("ix_players_player_name", "players", ["player_name"])
    op.create_index("ix_players_team_name", "players", ["team_name"])
    op.create_index("ix_players_position_group", "players", ["position_group"])

    op.create_table(
        "player_profiles",
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("passes_completed", sa.Integer(), nullable=False),
        sa.Column("actual_completion_rate", sa.Float(), nullable=False),
        sa.Column("expected_completions", sa.Float(), nullable=False),
        sa.Column("expected_completion_rate", sa.Float(), nullable=False),
        sa.Column("completions_above_expected", sa.Float(), nullable=False),
        sa.Column("completion_above_expected_pp", sa.Float(), nullable=False),
        sa.Column("pressure_attempts", sa.Integer(), nullable=False),
        sa.Column("pressure_completed", sa.Integer(), nullable=False),
        sa.Column("pressure_actual_completion_rate", sa.Float(), nullable=True),
        sa.Column("pressure_expected_completion_rate", sa.Float(), nullable=True),
        sa.Column("pressure_completions_above_expected", sa.Float(), nullable=True),
        sa.Column("pressure_above_expected_pp", sa.Float(), nullable=True),
        sa.Column("pressure_pass_rate", sa.Float(), nullable=False),
        sa.Column("progressive_attempts", sa.Integer(), nullable=False),
        sa.Column("progressive_completed", sa.Integer(), nullable=False),
        sa.Column("progressive_actual_completion_rate", sa.Float(), nullable=True),
        sa.Column("progressive_expected_completion_rate", sa.Float(), nullable=True),
        sa.Column("progressive_completions_above_expected", sa.Float(), nullable=True),
        sa.Column("progressive_above_expected_pp", sa.Float(), nullable=True),
        sa.Column("progressive_pass_rate", sa.Float(), nullable=False),
        sa.Column("long_pass_attempts", sa.Integer(), nullable=False),
        sa.Column("long_pass_completed", sa.Integer(), nullable=False),
        sa.Column("long_pass_actual_completion_rate", sa.Float(), nullable=True),
        sa.Column("long_pass_expected_completion_rate", sa.Float(), nullable=True),
        sa.Column("long_pass_completions_above_expected", sa.Float(), nullable=True),
        sa.Column("long_pass_above_expected_pp", sa.Float(), nullable=True),
        sa.Column("average_forward_distance", sa.Float(), nullable=False),
        sa.Column("net_forward_distance_per_100_passes", sa.Float(), nullable=False),
        sa.Column("positive_forward_distance_per_100_passes", sa.Float(), nullable=False),
        sa.Column("final_third_entries", sa.Integer(), nullable=False),
        sa.Column("final_third_entries_per_100_passes", sa.Float(), nullable=False),
        sa.Column("pressure_reliable", sa.Boolean(), nullable=False),
        sa.Column("progressive_reliable", sa.Boolean(), nullable=False),
        sa.Column("long_pass_reliable", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "actual_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_actual_rate",
        ),
        sa.CheckConstraint(
            "expected_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_expected_rate",
        ),
        sa.CheckConstraint(
            "pressure_actual_completion_rate IS NULL OR "
            "pressure_actual_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_pressure_actual_rate",
        ),
        sa.CheckConstraint(
            "pressure_expected_completion_rate IS NULL OR "
            "pressure_expected_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_pressure_expected_rate",
        ),
        sa.CheckConstraint(
            "progressive_actual_completion_rate IS NULL OR "
            "progressive_actual_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_progressive_actual_rate",
        ),
        sa.CheckConstraint(
            "progressive_expected_completion_rate IS NULL OR "
            "progressive_expected_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_progressive_expected_rate",
        ),
        sa.CheckConstraint(
            "long_pass_actual_completion_rate IS NULL OR "
            "long_pass_actual_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_long_actual_rate",
        ),
        sa.CheckConstraint(
            "long_pass_expected_completion_rate IS NULL OR "
            "long_pass_expected_completion_rate BETWEEN 0 AND 1",
            name="ck_profiles_long_expected_rate",
        ),
        sa.CheckConstraint("pressure_pass_rate BETWEEN 0 AND 1", name="ck_profiles_pressure_pass_rate"),
        sa.CheckConstraint(
            "progressive_pass_rate BETWEEN 0 AND 1",
            name="ck_profiles_progressive_pass_rate",
        ),
        sa.ForeignKeyConstraint(["player_id"], ["players.player_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("player_id"),
    )

    op.create_table(
        "player_similarities",
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("similar_player_id", sa.BigInteger(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("cosine_similarity", sa.Float(), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.Column("same_position_group", sa.Boolean(), nullable=False),
        sa.Column("position_group", sa.String(length=3), nullable=False),
        sa.Column("similar_position_group", sa.String(length=3), nullable=False),
        sa.Column("closest_feature_1", sa.String(length=100), nullable=False),
        sa.Column("closest_feature_2", sa.String(length=100), nullable=False),
        sa.Column("closest_feature_3", sa.String(length=100), nullable=False),
        sa.CheckConstraint("player_id <> similar_player_id", name="ck_similarities_not_self"),
        sa.CheckConstraint(
            "cosine_similarity BETWEEN -1 AND 1",
            name="ck_similarities_cosine_range",
        ),
        sa.CheckConstraint(
            "similarity_score BETWEEN 0 AND 100",
            name="ck_similarities_score_range",
        ),
        sa.CheckConstraint("rank >= 1", name="ck_similarities_rank_positive"),
        sa.ForeignKeyConstraint(["player_id"], ["players.player_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["similar_player_id"],
            ["players.player_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("player_id", "similar_player_id"),
        sa.UniqueConstraint("player_id", "rank", name="uq_similarities_player_rank"),
    )
    op.create_index(
        "ix_player_similarities_player_id",
        "player_similarities",
        ["player_id"],
    )
    op.create_index(
        "ix_player_similarities_similar_player_id",
        "player_similarities",
        ["similar_player_id"],
    )

    op.create_table(
        "passes",
        sa.Column(
            "pass_index", sa.BigInteger(), autoincrement=False, nullable=False
        ),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("player_id", sa.BigInteger(), nullable=True),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("position", sa.String(length=100), nullable=False),
        sa.Column("completed", sa.Boolean(), nullable=False),
        sa.Column("expected_completion", sa.Float(), nullable=False),
        sa.Column("fold", sa.Integer(), nullable=False),
        sa.Column("start_x", sa.Float(), nullable=False),
        sa.Column("start_y", sa.Float(), nullable=False),
        sa.Column("end_x", sa.Float(), nullable=False),
        sa.Column("end_y", sa.Float(), nullable=False),
        sa.Column("pass_length", sa.Float(), nullable=False),
        sa.Column("pass_angle", sa.Float(), nullable=False),
        sa.Column("forward_distance", sa.Float(), nullable=False),
        sa.Column("lateral_distance", sa.Float(), nullable=False),
        sa.Column("distance_to_goal_before", sa.Float(), nullable=False),
        sa.Column("distance_to_goal_after", sa.Float(), nullable=False),
        sa.Column("distance_toward_goal", sa.Float(), nullable=False),
        sa.Column("under_pressure", sa.Boolean(), nullable=False),
        sa.Column("progressive", sa.Boolean(), nullable=False),
        sa.Column("pass_height", sa.String(length=100), nullable=False),
        sa.Column("body_part", sa.String(length=100), nullable=False),
        sa.Column("pass_type", sa.String(length=100), nullable=False),
        sa.Column("start_zone", sa.String(length=100), nullable=False),
        sa.Column("end_zone", sa.String(length=100), nullable=False),
        sa.CheckConstraint(
            "expected_completion BETWEEN 0 AND 1",
            name="ck_passes_expected_completion",
        ),
        sa.CheckConstraint("fold >= 1", name="ck_passes_fold_positive"),
        sa.ForeignKeyConstraint(["player_id"], ["players.player_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("pass_index"),
    )
    op.create_index("ix_passes_player_id", "passes", ["player_id"])
    op.create_index("ix_passes_match_id", "passes", ["match_id"])
    op.create_index("ix_passes_player_match", "passes", ["player_id", "match_id"])


def downgrade() -> None:
    op.drop_table("passes")
    op.drop_table("player_similarities")
    op.drop_table("player_profiles")
    op.drop_table("players")
