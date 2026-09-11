"""Add production V4 team intelligence and Role Fit tables.

Revision ID: 20260911_0008
Revises: 20260910_0007
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260911_0008"
down_revision: str | None = "20260910_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "team_style_profiles",
        sa.Column("team_id", sa.BigInteger(), primary_key=True, autoincrement=False),
        sa.Column("team_name", sa.String(200), nullable=False),
        sa.Column("methodology_version", sa.String(20), nullable=False),
        sa.Column("sample_scope", sa.String(500), nullable=False),
        sa.Column("matches_observed", sa.Integer(), nullable=False),
        sa.Column("contributors", sa.Integer(), nullable=False),
        sa.Column("passes", sa.Integer(), nullable=False),
        sa.Column("carries", sa.Integer(), nullable=False),
        sa.Column("actions", sa.Integer(), nullable=False),
        sa.Column("shots", sa.Integer(), nullable=False),
        sa.Column("expected_completion_rate", sa.Float(), nullable=False),
        sa.Column("pressure_pass_rate", sa.Float(), nullable=False),
        sa.Column("progressive_pass_rate", sa.Float(), nullable=False),
        sa.Column("long_pass_rate", sa.Float(), nullable=False),
        sa.Column("positive_forward_distance_per_100_passes", sa.Float(), nullable=False),
        sa.Column("carry_share_of_actions", sa.Float(), nullable=False),
        sa.Column("average_forward_distance", sa.Float(), nullable=False),
        sa.Column("final_third_entries_per_100_passes", sa.Float(), nullable=False),
        sa.Column("progressive_carry_rate", sa.Float(), nullable=False),
        sa.Column("progressive_action_rate", sa.Float(), nullable=False),
        sa.Column("pressure_action_rate", sa.Float(), nullable=False),
        sa.Column("shots_per_match", sa.Float(), nullable=False),
        sa.Column("xg_per_shot", sa.Float(), nullable=False),
        sa.Column("xg_per_match", sa.Float(), nullable=False),
        sa.Column("attacking_value_per_100_actions", sa.Float(), nullable=False),
        sa.CheckConstraint("matches_observed >= 1", name="ck_team_style_matches"),
        sa.CheckConstraint(
            "passes >= 0 AND carries >= 0 AND actions >= 0", name="ck_team_style_counts"
        ),
    )
    op.create_table(
        "team_role_profiles",
        sa.Column("team_id", sa.BigInteger(), primary_key=True, autoincrement=False),
        sa.Column("position_group", sa.String(3), primary_key=True),
        sa.Column("team_name", sa.String(200), nullable=False),
        sa.Column("methodology_version", sa.String(20), nullable=False),
        sa.Column("aggregation_method", sa.String(50), nullable=False),
        sa.Column("matches_observed", sa.Integer(), nullable=False),
        sa.Column("contributor_count", sa.Integer(), nullable=False),
        sa.Column("contributors", sa.Text(), nullable=False),
        sa.Column("passes", sa.Integer(), nullable=False),
        sa.Column("carries", sa.Integer(), nullable=False),
        sa.Column("actions", sa.Integer(), nullable=False),
        sa.Column("shots", sa.Integer(), nullable=False),
        sa.Column("support_level", sa.String(50), nullable=False),
        sa.Column("support_message", sa.String(500), nullable=False),
        sa.Column("expected_completion_rate", sa.Float(), nullable=False),
        sa.Column("pressure_pass_rate", sa.Float(), nullable=False),
        sa.Column("progressive_pass_rate", sa.Float(), nullable=False),
        sa.Column("long_pass_rate", sa.Float(), nullable=False),
        sa.Column("positive_forward_distance_per_100_passes", sa.Float(), nullable=False),
        sa.Column("carry_share_of_actions", sa.Float(), nullable=False),
        sa.Column("expected_completion_rate_z", sa.Float(), nullable=False),
        sa.Column("pressure_pass_rate_z", sa.Float(), nullable=False),
        sa.Column("progressive_pass_rate_z", sa.Float(), nullable=False),
        sa.Column("long_pass_rate_z", sa.Float(), nullable=False),
        sa.Column("positive_forward_distance_per_100_passes_z", sa.Float(), nullable=False),
        sa.Column("carry_share_of_actions_z", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "position_group IN ('DEF', 'MID', 'FWD')", name="ck_team_roles_outfield"
        ),
        sa.CheckConstraint(
            "matches_observed >= 1 AND contributor_count >= 1", name="ck_team_roles_support"
        ),
        sa.UniqueConstraint("team_id", "position_group", name="uq_team_roles_team_position"),
    )
    op.create_index("ix_team_role_profiles_team_id", "team_role_profiles", ["team_id"])
    op.create_table(
        "player_role_fits",
        sa.Column("target_team_id", sa.BigInteger(), primary_key=True, autoincrement=False),
        sa.Column(
            "player_id", sa.BigInteger(), sa.ForeignKey("players.player_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("target_team_name", sa.String(200), nullable=False),
        sa.Column("player_name", sa.String(200), nullable=False),
        sa.Column("player_team_name", sa.String(200), nullable=False),
        sa.Column("position", sa.String(100), nullable=False),
        sa.Column("position_group", sa.String(3), nullable=False),
        sa.Column("is_target_team_player", sa.Boolean(), nullable=False),
        sa.Column("calculation_scope", sa.String(60), nullable=False),
        sa.Column("role_distance", sa.Float(), nullable=False),
        sa.Column("recommendation_rank", sa.Integer(), nullable=True),
        sa.Column("closest_feature_1", sa.String(100), nullable=False),
        sa.Column("closest_feature_2", sa.String(100), nullable=False),
        sa.Column("closest_feature_3", sa.String(100), nullable=False),
        sa.Column("largest_difference", sa.String(100), nullable=False),
        sa.Column("feature_gaps", sa.Text(), nullable=False),
        sa.Column("distance_contributions", sa.Text(), nullable=False),
        sa.Column("player_matches_observed", sa.Integer(), nullable=False),
        sa.Column("player_pass_attempts", sa.Integer(), nullable=False),
        sa.Column("player_carries", sa.Integer(), nullable=False),
        sa.Column("sample_support", sa.String(20), nullable=False),
        sa.Column("sample_support_message", sa.String(300), nullable=False),
        sa.Column("role_matches_observed", sa.Integer(), nullable=False),
        sa.Column("role_contributor_count", sa.Integer(), nullable=False),
        sa.Column("role_actions", sa.Integer(), nullable=False),
        sa.Column("role_support_message", sa.String(500), nullable=False),
        sa.Column("archetype_id", sa.String(40), nullable=True),
        sa.Column("archetype_name", sa.String(100), nullable=True),
        sa.Column("methodology_version", sa.String(20), nullable=False),
        sa.CheckConstraint(
            "position_group IN ('DEF', 'MID', 'FWD')", name="ck_role_fits_outfield"
        ),
        sa.CheckConstraint("role_distance >= 0", name="ck_role_fits_distance"),
        sa.CheckConstraint(
            "sample_support IN ('limited', 'higher')", name="ck_role_fits_support"
        ),
        sa.UniqueConstraint(
            "target_team_id", "position_group", "recommendation_rank",
            name="uq_role_fits_target_position_rank",
        ),
    )
    op.create_index(
        "ix_player_role_fits_target_role", "player_role_fits",
        ["target_team_id", "position_group"],
    )
    op.create_index("ix_player_role_fits_player_id", "player_role_fits", ["player_id"])


def downgrade() -> None:
    op.drop_table("player_role_fits")
    op.drop_table("team_role_profiles")
    op.drop_table("team_style_profiles")
