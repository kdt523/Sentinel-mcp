"""Core domain models shared by every gateway component.

``Action`` is the fixed six-action space. Names are part of the public contract (logs,
dataset labels, guard output) — do not rename without a coordinated migration.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

REASON_MAX_CHARS = 500


class Action(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    SANITIZE = "SANITIZE"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    REVOKE_TOOL = "REVOKE_TOOL"
    TERMINATE_SESSION = "TERMINATE_SESSION"


class Direction(StrEnum):
    REQUEST = "request"  # agent -> tool (tools/call arguments)
    OUTPUT = "output"  # tool -> agent (tool results, tool descriptions from tools/list)


class EventKind(StrEnum):
    TOOL_CALL = "tool_call"  # tools/call request or its result
    TOOL_DESCRIPTION = "tool_description"  # a tool's description from tools/list (output only)


class Tier(StrEnum):
    POLICY = "policy"
    FAST = "fast"
    GUARD = "guard"
    FALLBACK = "fallback"  # fail-mode decision after an error/timeout


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid4().hex


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Event(_Frozen):
    """One inspectable unit of traffic: a tool-call request or a tool output.

    Output text is stored truncated to a configurable maximum; ``output_sha256`` and
    ``output_chars`` always describe the FULL original text. Build output events with
    :meth:`from_output` so truncation and hashing are applied consistently.
    """

    session_id: str = Field(min_length=1)
    event_id: str = Field(default_factory=_new_id)
    ts: datetime = Field(default_factory=_utcnow)
    kind: EventKind = EventKind.TOOL_CALL
    direction: Direction
    server: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    args: dict[str, Any] | None = None
    output: str | None = None
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_chars: int | None = Field(default=None, ge=0)
    upstream_ms: float | None = Field(default=None, ge=0)

    @classmethod
    def from_output(
        cls,
        *,
        session_id: str,
        server: str,
        tool: str,
        full_text: str,
        max_chars: int,
        kind: EventKind = EventKind.TOOL_CALL,
        upstream_ms: float | None = None,
    ) -> Event:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        return cls(
            session_id=session_id,
            kind=kind,
            direction=Direction.OUTPUT,
            server=server,
            tool=tool,
            output=full_text[:max_chars],
            output_sha256=sha256_text(full_text),
            output_chars=len(full_text),
            upstream_ms=upstream_ms,
        )

    @property
    def qualified_tool(self) -> str:
        """Namespaced tool name as exposed to the agent: ``<server>.<tool>``."""
        return f"{self.server}.{self.tool}"

    @property
    def output_truncated(self) -> bool:
        return (
            self.output is not None
            and self.output_chars is not None
            and self.output_chars > len(self.output)
        )

    @model_validator(mode="after")
    def _validate_shape(self) -> Self:
        if self.ts.tzinfo is None:
            raise ValueError("ts must be timezone-aware")
        if self.direction is Direction.REQUEST:
            if self.kind is EventKind.TOOL_DESCRIPTION:
                raise ValueError("tool_description events are always direction=output")
            if self.args is None:
                raise ValueError("request events carry args")
            if any(
                v is not None
                for v in (self.output, self.output_sha256, self.output_chars, self.upstream_ms)
            ):
                raise ValueError("request events carry no output fields or upstream_ms")
            return self
        # direction == output
        if self.args is not None:
            raise ValueError("output events carry no args")
        if self.output is None or self.output_sha256 is None or self.output_chars is None:
            raise ValueError("output events need output, output_sha256 and output_chars")
        if len(self.output) > self.output_chars:
            raise ValueError("stored output longer than output_chars")
        if len(self.output) == self.output_chars and sha256_text(self.output) != self.output_sha256:
            raise ValueError("output_sha256 does not match untruncated output")
        return self


class Decision(_Frozen):
    """Verdict for one Event. Always explainable: action + risk + short reason."""

    action: Action
    risk: float = Field(ge=0.0, le=1.0)
    attack_family: str | None = None
    reason: str = Field(min_length=1, max_length=REASON_MAX_CHARS)
    tier: Tier  # tier that originally produced the verdict (kept when served from cache)
    cached: bool = False
    latency_ms: float = Field(default=0.0, ge=0)
    model_version: str = Field(min_length=1)
