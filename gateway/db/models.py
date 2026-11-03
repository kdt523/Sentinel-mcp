"""SQLAlchemy models for the decision log, sessions, approvals and analyst feedback.

Nothing here ever holds raw secrets: arguments are stored redacted, outputs as a redacted
preview plus sha256 and length of the full text (full raw capture is a separate,
sandbox-only file sink — see gateway/recorder.py).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Deterministic constraint names so Alembic migrations stay stable across databases.
NAMING = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

JsonType = JSON().with_variant(JSONB(), "postgresql")
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")  # sqlite autoincrement needs INTEGER


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


class SessionRow(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    event_count: Mapped[int] = mapped_column(Integer, default=0)
    terminated: Mapped[bool] = mapped_column(Boolean, default=False)
    terminated_reason: Mapped[str | None] = mapped_column(Text)


class DecisionRow(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    # event identity
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True)
    event_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    direction: Mapped[str] = mapped_column(String(16))
    server: Mapped[str] = mapped_column(String(128))
    tool: Mapped[str] = mapped_column(String(128))
    qualified_tool: Mapped[str] = mapped_column(String(256), index=True)
    # content (redacted)
    args_redacted: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    output_preview: Mapped[str | None] = mapped_column(Text)
    output_sha256: Mapped[str | None] = mapped_column(String(64))
    output_chars: Mapped[int | None] = mapped_column(Integer)
    output_truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    upstream_ms: Mapped[float | None] = mapped_column(Float)
    # decision
    action: Mapped[str] = mapped_column(String(32), index=True)
    risk: Mapped[float] = mapped_column(Float)
    attack_family: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text)
    tier: Mapped[str] = mapped_column(String(16))
    rule_id: Mapped[str | None] = mapped_column(String(80))
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(128))
    enforced: Mapped[bool] = mapped_column(Boolean)  # False in shadow mode
    logged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_decisions_session_ts", "session_id", "event_ts"),)


class ApprovalRow(Base):
    __tablename__ = "approvals"

    approval_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    qualified_tool: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(16), index=True)  # pending|approved|denied|expired
    reason: Mapped[str] = mapped_column(Text)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_by: Mapped[str | None] = mapped_column(String(128))
    note: Mapped[str | None] = mapped_column(Text)


class FeedbackRow(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    decision_id: Mapped[int] = mapped_column(
        ForeignKey("decisions.id", ondelete="CASCADE"), index=True
    )
    verdict: Mapped[str] = mapped_column(String(16))  # correct|wrong
    correct_action: Mapped[str | None] = mapped_column(String(32))
    note: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
