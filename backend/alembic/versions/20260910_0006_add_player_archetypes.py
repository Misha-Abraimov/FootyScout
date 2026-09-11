"""Add V3.2C production player archetype assignments.

Revision ID: 20260910_0006
Revises: 20260909_0005
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260910_0006"
down_revision: str | None = "20260909_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "player_archetypes",
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("archetype_id", sa.String(length=40), nullable=False),
        sa.Column("archetype_name", sa.String(length=100), nullable=False),
        sa.Column("raw_cluster_id", sa.Integer(), nullable=False),
        sa.Column("position_group", sa.String(length=3), nullable=False),
        sa.Column("centroid_distance", sa.Float(), nullable=False),
        sa.Column("second_centroid_distance", sa.Float(), nullable=False),
        sa.Column("separation_margin", sa.Float(), nullable=False),
        sa.Column("eligible", sa.Boolean(), nullable=False),
        sa.Column("model_version", sa.String(length=40), nullable=False),
        sa.Column("expected_completion_rate_position_z", sa.Float(), nullable=False),
        sa.Column("pressure_pass_rate_position_z", sa.Float(), nullable=False),
        sa.Column("progressive_pass_rate_position_z", sa.Float(), nullable=False),
        sa.Column("long_pass_rate_position_z", sa.Float(), nullable=False),
        sa.Column(
            "positive_forward_distance_per_100_passes_position_z",
            sa.Float(),
            nullable=False,
        ),
        sa.Column("carry_share_of_actions_position_z", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "archetype_id IN ('direct_progressor', 'safe_circulator')",
            name="ck_player_archetypes_semantic_id",
        ),
        sa.CheckConstraint(
            "position_group IN ('DEF', 'MID', 'FWD')",
            name="ck_player_archetypes_outfield_position",
        ),
        sa.CheckConstraint(
            "centroid_distance >= 0 AND second_centroid_distance >= centroid_distance",
            name="ck_player_archetypes_distances",
        ),
        sa.CheckConstraint(
            "separation_margin BETWEEN 0 AND 1",
            name="ck_player_archetypes_separation",
        ),
        sa.CheckConstraint("eligible = true", name="ck_player_archetypes_eligible"),
        sa.ForeignKeyConstraint(["player_id"], ["players.player_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("player_id"),
    )
    op.create_index(
        "ix_player_archetypes_archetype_id", "player_archetypes", ["archetype_id"]
    )
    op.create_index(
        "ix_player_archetypes_position_group", "player_archetypes", ["position_group"]
    )
    op.create_index(
        "ix_player_archetypes_raw_cluster", "player_archetypes", ["raw_cluster_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_player_archetypes_raw_cluster", table_name="player_archetypes")
    op.drop_index("ix_player_archetypes_position_group", table_name="player_archetypes")
    op.drop_index("ix_player_archetypes_archetype_id", table_name="player_archetypes")
    op.drop_table("player_archetypes")
