"""Add expected-goals shots and player shooting profiles.

Revision ID: 20260909_0002
Revises: 20260908_0001
Create Date: 2026-09-09
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260909_0002"
down_revision: str | None = "20260908_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "shots",
        sa.Column("shot_id", sa.String(length=36), nullable=False),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("player_id", sa.BigInteger(), nullable=True),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("period", sa.Integer(), nullable=False),
        sa.Column("minute", sa.Integer(), nullable=False),
        sa.Column("second", sa.Integer(), nullable=False),
        sa.Column("start_x", sa.Float(), nullable=False),
        sa.Column("start_y", sa.Float(), nullable=False),
        sa.Column("distance", sa.Float(), nullable=False),
        sa.Column("angle", sa.Float(), nullable=False),
        sa.Column("goal", sa.Boolean(), nullable=False),
        sa.Column("expected_goal", sa.Float(), nullable=True),
        sa.Column("body_part", sa.String(length=100), nullable=False),
        sa.Column("shot_type", sa.String(length=100), nullable=False),
        sa.Column("technique", sa.String(length=100), nullable=False),
        sa.Column("play_pattern", sa.String(length=100), nullable=False),
        sa.Column("under_pressure", sa.Boolean(), nullable=False),
        sa.Column("first_time", sa.Boolean(), nullable=False),
        sa.Column("one_on_one", sa.Boolean(), nullable=False),
        sa.Column("open_goal", sa.Boolean(), nullable=False),
        sa.Column("penalty", sa.Boolean(), nullable=False),
        sa.Column("penalty_shootout", sa.Boolean(), nullable=False),
        sa.Column("model_eligible", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "expected_goal IS NULL OR expected_goal BETWEEN 0 AND 1",
            name="ck_shots_expected_goal",
        ),
        sa.CheckConstraint("goal IN (false, true)", name="ck_shots_goal_boolean"),
        sa.ForeignKeyConstraint(["player_id"], ["players.player_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("shot_id"),
    )
    op.create_index("ix_shots_player_id", "shots", ["player_id"])
    op.create_index("ix_shots_match_id", "shots", ["match_id"])
    op.create_index("ix_shots_player_match", "shots", ["player_id", "match_id"])
    op.create_table(
        "player_shooting_profiles",
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("shots", sa.Integer(), nullable=False),
        sa.Column("goals", sa.Integer(), nullable=False),
        sa.Column("total_xg", sa.Float(), nullable=False),
        sa.Column("xg_per_shot", sa.Float(), nullable=False),
        sa.Column("goals_minus_xg", sa.Float(), nullable=False),
        sa.Column("goals_per_shot", sa.Float(), nullable=False),
        sa.Column("matches_observed", sa.Integer(), nullable=False),
        sa.Column("shooting_reliable", sa.Boolean(), nullable=False),
        sa.CheckConstraint("shots >= 0", name="ck_shooting_profiles_shots"),
        sa.CheckConstraint("goals >= 0", name="ck_shooting_profiles_goals"),
        sa.CheckConstraint("total_xg >= 0", name="ck_shooting_profiles_total_xg"),
        sa.CheckConstraint("xg_per_shot BETWEEN 0 AND 1", name="ck_shooting_profiles_xg_per_shot"),
        sa.CheckConstraint("goals_per_shot BETWEEN 0 AND 1", name="ck_shooting_profiles_goals_per_shot"),
        sa.CheckConstraint("matches_observed >= 0", name="ck_shooting_profiles_matches"),
        sa.ForeignKeyConstraint(["player_id"], ["players.player_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("player_id"),
    )


def downgrade() -> None:
    op.drop_table("player_shooting_profiles")
    op.drop_index("ix_shots_player_match", table_name="shots")
    op.drop_index("ix_shots_match_id", table_name="shots")
    op.drop_index("ix_shots_player_id", table_name="shots")
    op.drop_table("shots")
