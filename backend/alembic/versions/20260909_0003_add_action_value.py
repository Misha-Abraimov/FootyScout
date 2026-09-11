"""Add attacking action values and player profiles.

Revision ID: 20260909_0003
Revises: 20260909_0002
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260909_0003"
down_revision: str | None = "20260909_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "attacking_actions",
        sa.Column("action_id", sa.String(length=36), primary_key=True),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("possession_id", sa.BigInteger(), nullable=False),
        sa.Column("event_index", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.BigInteger(), nullable=True),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("action_type", sa.String(length=20), nullable=False),
        sa.Column("start_x", sa.Float(), nullable=False),
        sa.Column("start_y", sa.Float(), nullable=False),
        sa.Column("end_x", sa.Float(), nullable=False),
        sa.Column("end_y", sa.Float(), nullable=False),
        sa.Column("state_value_before", sa.Float(), nullable=False),
        sa.Column("state_value_after", sa.Float(), nullable=False),
        sa.Column("attacking_value", sa.Float(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("under_pressure", sa.Boolean(), nullable=False),
        sa.Column("progressive", sa.Boolean(), nullable=False),
        sa.Column("expected_completion", sa.Float(), nullable=True),
        sa.Column("pass_risk", sa.Float(), nullable=True),
        sa.Column("risk_reward_category", sa.String(length=40), nullable=True),
        sa.Column("fold", sa.Integer(), nullable=False),
        sa.CheckConstraint("state_value_before >= 0", name="ck_actions_before_nonnegative"),
        sa.CheckConstraint("state_value_after >= 0", name="ck_actions_after_nonnegative"),
        sa.CheckConstraint("expected_completion IS NULL OR expected_completion BETWEEN 0 AND 1", name="ck_actions_expected_completion"),
        sa.CheckConstraint("pass_risk IS NULL OR pass_risk BETWEEN 0 AND 1", name="ck_actions_pass_risk"),
        sa.CheckConstraint("fold >= 1", name="ck_actions_fold_positive"),
        sa.ForeignKeyConstraint(["player_id"], ["players.player_id"], ondelete="SET NULL"),
    )
    for column in ["player_id", "match_id", "possession_id", "action_type", "attacking_value"]:
        op.create_index(f"ix_attacking_actions_{column}", "attacking_actions", [column])
    op.create_table(
        "player_attacking_profiles",
        sa.Column("player_id", sa.BigInteger(), primary_key=True),
        sa.Column("matches_observed", sa.Integer(), nullable=False),
        sa.Column("actions", sa.Integer(), nullable=False),
        sa.Column("passes", sa.Integer(), nullable=False),
        sa.Column("carries", sa.Integer(), nullable=False),
        sa.Column("total_attacking_value", sa.Float(), nullable=False),
        sa.Column("attacking_value_per_100_actions", sa.Float(), nullable=False),
        sa.Column("total_pass_value", sa.Float(), nullable=False),
        sa.Column("pass_value_per_100_passes", sa.Float(), nullable=True),
        sa.Column("total_carry_value", sa.Float(), nullable=False),
        sa.Column("carry_value_per_100_carries", sa.Float(), nullable=True),
        sa.Column("positive_value_actions", sa.Integer(), nullable=False),
        sa.Column("positive_value_action_rate", sa.Float(), nullable=False),
        sa.Column("progressive_action_value", sa.Float(), nullable=False),
        sa.Column("progressive_value_per_100_actions", sa.Float(), nullable=False),
        sa.Column("pressure_action_value", sa.Float(), nullable=False),
        sa.Column("pressure_value_per_100_actions", sa.Float(), nullable=False),
        sa.Column("attacking_value_reliable", sa.Boolean(), nullable=False),
        sa.Column("pass_value_reliable", sa.Boolean(), nullable=False),
        sa.Column("carry_value_reliable", sa.Boolean(), nullable=False),
        sa.CheckConstraint("actions >= 0 AND passes >= 0 AND carries >= 0", name="ck_attacking_profile_counts"),
        sa.CheckConstraint("positive_value_action_rate BETWEEN 0 AND 1", name="ck_attacking_profile_positive_rate"),
        sa.ForeignKeyConstraint(["player_id"], ["players.player_id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    op.drop_table("player_attacking_profiles")
    for column in ["attacking_value", "action_type", "possession_id", "match_id", "player_id"]:
        op.drop_index(f"ix_attacking_actions_{column}", table_name="attacking_actions")
    op.drop_table("attacking_actions")
