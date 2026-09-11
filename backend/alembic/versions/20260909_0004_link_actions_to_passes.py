"""Link attacking pass rows to the frozen V1 pass identity.

Revision ID: 20260909_0004
Revises: 20260909_0003
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260909_0004"
down_revision: str | None = "20260909_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("attacking_actions")}
    if "pass_index" not in columns:
        op.add_column(
            "attacking_actions", sa.Column("pass_index", sa.BigInteger(), nullable=True)
        )
        inspector = sa.inspect(op.get_bind())
    unique_columns = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("attacking_actions")
    }
    if ("pass_index",) not in unique_columns:
        op.create_unique_constraint(
            "uq_attacking_actions_pass_index", "attacking_actions", ["pass_index"]
        )
    foreign_columns = {
        tuple(constraint["constrained_columns"])
        for constraint in inspector.get_foreign_keys("attacking_actions")
    }
    if ("pass_index",) not in foreign_columns:
        op.create_foreign_key(
            "fk_attacking_actions_pass_index",
            "attacking_actions",
            "passes",
            ["pass_index"],
            ["pass_index"],
            ondelete="SET NULL",
        )
    indexes = {index["name"] for index in inspector.get_indexes("attacking_actions")}
    if "ix_attacking_actions_pass_index" not in indexes:
        op.create_index("ix_attacking_actions_pass_index", "attacking_actions", ["pass_index"])


def downgrade() -> None:
    op.drop_index("ix_attacking_actions_pass_index", table_name="attacking_actions")
    op.drop_constraint("fk_attacking_actions_pass_index", "attacking_actions", type_="foreignkey")
    op.drop_constraint("uq_attacking_actions_pass_index", "attacking_actions", type_="unique")
    op.drop_column("attacking_actions", "pass_index")
