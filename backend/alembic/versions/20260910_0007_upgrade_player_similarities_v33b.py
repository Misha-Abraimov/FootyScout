"""Upgrade production player similarities to V3.3B.

Revision ID: 20260910_0007
Revises: 20260910_0006
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260910_0007"
down_revision: str | None = "20260910_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_similarities_cosine_range", "player_similarities", type_="check"
    )
    op.drop_column("player_similarities", "cosine_similarity")
    op.add_column(
        "player_similarities",
        sa.Column("rms_distance", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "player_similarities",
        sa.Column("query_matches_observed", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "player_similarities",
        sa.Column(
            "candidate_matches_observed", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "player_similarities",
        sa.Column("pair_support_matches", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "player_similarities",
        sa.Column("sample_support", sa.String(length=20), nullable=False, server_default="limited"),
    )
    op.add_column(
        "player_similarities",
        sa.Column(
            "sample_support_explanation",
            sa.String(length=300),
            nullable=False,
            server_default="Legacy row pending V3.3B snapshot reload.",
        ),
    )
    op.add_column(
        "player_similarities",
        sa.Column("methodology_version", sa.String(length=20), nullable=False, server_default="V1"),
    )
    op.add_column(
        "player_similarities",
        sa.Column("distance_contributions", sa.Text(), nullable=False, server_default="{}"),
    )
    op.create_check_constraint(
        "ck_similarities_distance_nonnegative", "player_similarities", "rms_distance >= 0"
    )
    op.create_check_constraint(
        "ck_similarities_match_counts_nonnegative",
        "player_similarities",
        "query_matches_observed >= 0 AND candidate_matches_observed >= 0 "
        "AND pair_support_matches >= 0",
    )
    op.create_check_constraint(
        "ck_similarities_pair_support_weaker",
        "player_similarities",
        "pair_support_matches <= query_matches_observed "
        "AND pair_support_matches <= candidate_matches_observed",
    )
    op.create_check_constraint(
        "ck_similarities_sample_support",
        "player_similarities",
        "sample_support IN ('limited', 'higher')",
    )
    for column in (
        "rms_distance",
        "query_matches_observed",
        "candidate_matches_observed",
        "pair_support_matches",
        "sample_support",
        "sample_support_explanation",
        "methodology_version",
        "distance_contributions",
    ):
        op.alter_column("player_similarities", column, server_default=None)


def downgrade() -> None:
    op.drop_constraint(
        "ck_similarities_sample_support", "player_similarities", type_="check"
    )
    op.drop_constraint(
        "ck_similarities_pair_support_weaker", "player_similarities", type_="check"
    )
    op.drop_constraint(
        "ck_similarities_match_counts_nonnegative", "player_similarities", type_="check"
    )
    op.drop_constraint(
        "ck_similarities_distance_nonnegative", "player_similarities", type_="check"
    )
    for column in (
        "distance_contributions",
        "methodology_version",
        "sample_support_explanation",
        "sample_support",
        "pair_support_matches",
        "candidate_matches_observed",
        "query_matches_observed",
        "rms_distance",
    ):
        op.drop_column("player_similarities", column)
    op.add_column(
        "player_similarities",
        sa.Column("cosine_similarity", sa.Float(), nullable=False, server_default="0"),
    )
    op.alter_column("player_similarities", "cosine_similarity", server_default=None)
    op.create_check_constraint(
        "ck_similarities_cosine_range",
        "player_similarities",
        "cosine_similarity BETWEEN -1 AND 1",
    )
