"""initial schema

Revision ID: 0001
Revises: (none)
Create Date: 2026-11-04 00:13:39.191506
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "approvals",
        sa.Column("approval_id", sa.String(length=64), nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("qualified_tool", sa.String(length=256), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(length=128), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("approval_id", name=op.f("pk_approvals")),
    )
    with op.batch_alter_table("approvals", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_approvals_event_id"), ["event_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_approvals_session_id"), ["session_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_approvals_status"), ["status"], unique=False)

    op.create_table(
        "decisions",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("event_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("direction", sa.String(length=16), nullable=False),
        sa.Column("server", sa.String(length=128), nullable=False),
        sa.Column("tool", sa.String(length=128), nullable=False),
        sa.Column("qualified_tool", sa.String(length=256), nullable=False),
        sa.Column(
            "args_redacted",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
        sa.Column("output_preview", sa.Text(), nullable=True),
        sa.Column("output_sha256", sa.String(length=64), nullable=True),
        sa.Column("output_chars", sa.Integer(), nullable=True),
        sa.Column("output_truncated", sa.Boolean(), nullable=False),
        sa.Column("upstream_ms", sa.Float(), nullable=True),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("risk", sa.Float(), nullable=False),
        sa.Column("attack_family", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("tier", sa.String(length=16), nullable=False),
        sa.Column("rule_id", sa.String(length=80), nullable=True),
        sa.Column("cached", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("enforced", sa.Boolean(), nullable=False),
        sa.Column(
            "logged_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_decisions")),
        sa.UniqueConstraint("event_id", name=op.f("uq_decisions_event_id")),
    )
    with op.batch_alter_table("decisions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_decisions_action"), ["action"], unique=False)
        batch_op.create_index(batch_op.f("ix_decisions_event_ts"), ["event_ts"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_decisions_qualified_tool"), ["qualified_tool"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_decisions_session_id"), ["session_id"], unique=False)
        batch_op.create_index("ix_decisions_session_ts", ["session_id", "event_ts"], unique=False)

    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_count", sa.Integer(), nullable=False),
        sa.Column("terminated", sa.Boolean(), nullable=False),
        sa.Column("terminated_reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("session_id", name=op.f("pk_sessions")),
    )
    op.create_table(
        "feedback",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "decision_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False
        ),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("correct_action", sa.String(length=32), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("author", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name=op.f("fk_feedback_decision_id_decisions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_feedback")),
    )
    with op.batch_alter_table("feedback", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_feedback_decision_id"), ["decision_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("feedback", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_feedback_decision_id"))

    op.drop_table("feedback")
    op.drop_table("sessions")
    with op.batch_alter_table("decisions", schema=None) as batch_op:
        batch_op.drop_index("ix_decisions_session_ts")
        batch_op.drop_index(batch_op.f("ix_decisions_session_id"))
        batch_op.drop_index(batch_op.f("ix_decisions_qualified_tool"))
        batch_op.drop_index(batch_op.f("ix_decisions_event_ts"))
        batch_op.drop_index(batch_op.f("ix_decisions_action"))

    op.drop_table("decisions")
    with op.batch_alter_table("approvals", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_approvals_status"))
        batch_op.drop_index(batch_op.f("ix_approvals_session_id"))
        batch_op.drop_index(batch_op.f("ix_approvals_event_id"))

    op.drop_table("approvals")
