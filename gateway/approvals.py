"""Human-approval requests in Redis.

Each approval is a hash with a TTL plus an entry in a pending index. Every status change
goes through one Lua compare-and-set that only succeeds from ``pending``, so a reviewer's
approve and the gateway's expiry/cancel can race safely: exactly one wins.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis
from redis.typing import EncodableT, FieldT

from gateway.models import ApprovalStatus, Decision, Event

KEY_PREFIX = "sentinel:approval"
PENDING_INDEX = "sentinel:approvals:pending"
PREVIEW_CHARS = 300

# KEYS[1]=approval hash, KEYS[2]=pending index; ARGV: id, new_status, resolved_at, by, note
_TRANSITION_LUA = """
local status = redis.call('HGET', KEYS[1], 'status')
if status ~= 'pending' then
  return status or false
end
redis.call('HSET', KEYS[1], 'status', ARGV[2], 'resolved_at', ARGV[3],
           'resolved_by', ARGV[4], 'note', ARGV[5])
redis.call('ZREM', KEYS[2], ARGV[1])
return 'ok'
"""


class Approval(BaseModel):
    model_config = ConfigDict(frozen=True)

    approval_id: str
    status: ApprovalStatus
    session_id: str
    event_id: str
    qualified_tool: str
    reason: str
    preview: str  # redacted args preview for the reviewer
    requested_at: datetime
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    note: str | None = None

    def db_row(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "qualified_tool": self.qualified_tool,
            "status": self.status.value,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "note": self.note,
        }


class TransitionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    status: ApprovalStatus | None  # current status (new one if ok, the winner's if not)


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


class ApprovalStore:
    def __init__(
        self, redis: Redis, *, ttl_s: int = 86_400, clock: Callable[[], float] = time.time
    ) -> None:
        self._r = redis
        self.ttl_s = ttl_s
        self._clock = clock

    @staticmethod
    def _k(approval_id: str) -> str:
        return f"{KEY_PREFIX}:{approval_id}"

    async def create(
        self, event: Event, decision: Decision, *, preview: str, approval_id: str | None = None
    ) -> Approval:
        """Store a pending approval. Callers may pre-allocate ``approval_id`` so they can
        clean up even if they are cancelled while this call is in flight."""
        now = self._clock()
        approval = Approval(
            approval_id=approval_id or uuid4().hex,
            status=ApprovalStatus.PENDING,
            session_id=event.session_id,
            event_id=event.event_id,
            qualified_tool=event.qualified_tool,
            reason=decision.reason,
            preview=preview[:PREVIEW_CHARS],
            requested_at=datetime.fromtimestamp(now, UTC),
        )
        mapping: dict[FieldT, EncodableT] = {
            "status": approval.status.value,
            "session_id": approval.session_id,
            "event_id": approval.event_id,
            "qualified_tool": approval.qualified_tool,
            "reason": approval.reason,
            "preview": approval.preview,
            "requested_at": repr(now),
        }
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.hset(self._k(approval.approval_id), mapping=mapping)
            pipe.expire(self._k(approval.approval_id), self.ttl_s)
            pipe.zadd(PENDING_INDEX, {approval.approval_id: now})
            await pipe.execute()
        return approval

    async def get(self, approval_id: str) -> Approval | None:
        raw = await self._r.hgetall(self._k(approval_id))
        if not raw:
            return None
        h = {_s(k): _s(v) for k, v in raw.items()}
        resolved = h.get("resolved_at")
        return Approval(
            approval_id=approval_id,
            status=ApprovalStatus(h["status"]),
            session_id=h["session_id"],
            event_id=h["event_id"],
            qualified_tool=h["qualified_tool"],
            reason=h["reason"],
            preview=h.get("preview", ""),
            requested_at=datetime.fromtimestamp(float(h["requested_at"]), UTC),
            resolved_at=datetime.fromtimestamp(float(resolved), UTC) if resolved else None,
            resolved_by=h.get("resolved_by") or None,
            note=h.get("note") or None,
        )

    async def status(self, approval_id: str) -> ApprovalStatus | None:
        v = await self._r.hget(self._k(approval_id), "status")
        return None if v is None else ApprovalStatus(_s(v))

    async def transition(
        self, approval_id: str, to: ApprovalStatus, *, by: str = "", note: str = ""
    ) -> TransitionResult:
        """Move ``pending`` -> ``to``. Fails (ok=False) if already resolved or unknown."""
        if to is ApprovalStatus.PENDING:
            raise ValueError("cannot transition to pending")
        out = await self._r.eval(
            _TRANSITION_LUA, 2, self._k(approval_id), PENDING_INDEX,
            approval_id, to.value, repr(self._clock()), by[:128], note[:500],
        )  # fmt: skip
        if out is None:
            return TransitionResult(ok=False, status=None)
        val = _s(out)
        if val == "ok":
            return TransitionResult(ok=True, status=to)
        return TransitionResult(ok=False, status=ApprovalStatus(val))

    async def list_pending(self, limit: int = 100) -> list[Approval]:
        ids = await self._r.zrange(PENDING_INDEX, 0, limit - 1)
        out = []
        for raw_id in ids:
            a = await self.get(_s(raw_id))
            if a is not None and a.status is ApprovalStatus.PENDING:
                out.append(a)
        return out
