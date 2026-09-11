"""Add normalized V3.1 player intelligence and percentile tables.

Revision ID: 20260909_0005
Revises: 20260909_0004
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260909_0005"
down_revision: str | None = "20260909_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "player_intelligence_profiles",
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("position_group", sa.String(length=3), nullable=False),
        sa.Column("matches_observed", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "position_group IN ('GK', 'DEF', 'MID', 'FWD')",
            name="ck_intelligence_profiles_position_group",
        ),
        sa.CheckConstraint("matches_observed >= 0", name="ck_intelligence_profiles_matches"),
        sa.ForeignKeyConstraint(["player_id"], ["players.player_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("player_id"),
    )
    op.create_index(
        "ix_intelligence_profiles_position_group",
        "player_intelligence_profiles",
        ["position_group"],
    )
    op.create_table(
        "player_percentiles",
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("metric_name", sa.String(length=100), nullable=False),
        sa.Column("family", sa.String(length=20), nullable=False),
        sa.Column("raw_value", sa.Float(), nullable=True),
        sa.Column("percentile", sa.Float(), nullable=True),
        sa.Column("peer_position_group", sa.String(length=3), nullable=False),
        sa.Column("peer_count", sa.Integer(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("eligible", sa.Boolean(), nullable=False),
        sa.Column("eligibility_reason", sa.String(length=100), nullable=False),
        sa.CheckConstraint("family IN ('style', 'performance')", name="ck_percentiles_family"),
        sa.CheckConstraint(
            "peer_position_group IN ('GK', 'DEF', 'MID', 'FWD')",
            name="ck_percentiles_position_group",
        ),
        sa.CheckConstraint(
            "percentile IS NULL OR percentile BETWEEN 0 AND 100",
            name="ck_percentiles_range",
        ),
        sa.CheckConstraint(
            "peer_count >= 0 AND sample_count >= 0", name="ck_percentiles_counts"
        ),
        sa.ForeignKeyConstraint(["player_id"], ["players.player_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("player_id", "metric_name"),
    )
    op.create_index("ix_player_percentiles_player_id", "player_percentiles", ["player_id"])
    op.create_index(
        "ix_player_percentiles_position_group",
        "player_percentiles",
        ["peer_position_group"],
    )
    op.create_index(
        "ix_player_percentiles_metric_name", "player_percentiles", ["metric_name"]
    )


def downgrade() -> None:
    op.drop_index("ix_player_percentiles_metric_name", table_name="player_percentiles")
    op.drop_index("ix_player_percentiles_position_group", table_name="player_percentiles")
    op.drop_index("ix_player_percentiles_player_id", table_name="player_percentiles")
    op.drop_table("player_percentiles")
    op.drop_index(
        "ix_intelligence_profiles_position_group",
        table_name="player_intelligence_profiles",
    )
    op.drop_table("player_intelligence_profiles")
