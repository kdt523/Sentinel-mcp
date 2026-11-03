"""Per-session state in Redis: history, files read, secret taint, egress, decayed risk,
revoked tools, termination and tool-description hashes.

All mutations are atomic per call (MULTI/EXEC pipelines or Lua scripts), every key carries
a TTL that is refreshed on write, and nothing secret is stored: history previews are
redacted and taint is recorded as detector LABELS, never values.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis

from gateway.config import Settings
from gateway.detectors import (
    egress_targets,
    paths_in_args,
    redact_secrets,
    secret_content_labels,
    secret_path_labels,
)
from gateway.models import Action, Decision, Direction, Event

KEY_PREFIX = "sentinel:sess"
PREVIEW_CHARS = 120
SUMMARY_MAX_CHARS = 1_200
_PASSED = frozenset({Action.ALLOW, Action.SANITIZE})  # call actually reached the tool / agent

# KEYS[1]=meta hash; ARGV: add, now, half_life, ttl. Returns new accumulator as a string
# (Lua numbers would otherwise be truncated to integers on the way back).
_RISK_LUA = """
local acc = tonumber(redis.call('HGET', KEYS[1], 'risk_accum') or '0')
local ts = tonumber(redis.call('HGET', KEYS[1], 'risk_ts') or ARGV[2])
local now = tonumber(ARGV[2])
local dt = now - ts
if dt < 0 then dt = 0 end
acc = acc * math.pow(0.5, dt / tonumber(ARGV[3])) + tonumber(ARGV[1])
redis.call('HSET', KEYS[1], 'risk_accum', tostring(acc), 'risk_ts', ARGV[2])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[4]))
return tostring(acc)
"""

# KEYS[1]=desc hash; ARGV: tool, sha, ttl. Returns previous sha or false (compare-and-set).
_DESC_LUA = """
local old = redis.call('HGET', KEYS[1], ARGV[1])
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
return old
"""


class CompactEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    ts: datetime
    direction: Direction
    tool: str  # qualified <server>.<tool>
    preview: str  # redacted, truncated
    action: Action
    risk: float


class DescriptionObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    first_seen: bool
    changed: bool
    previous_sha256: str | None


class SessionState(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    event_count: int = 0
    history: list[CompactEvent] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)
    secret_taint: bool = False
    taint_sources: list[str] = Field(default_factory=list)
    egress_targets: list[str] = Field(default_factory=list)
    risk: float = 0.0  # decayed to "now" at read time
    revoked_tools: list[str] = Field(default_factory=list)
    terminated: bool = False
    terminated_reason: str | None = None


def _preview(event: Event) -> str:
    raw = json.dumps(event.args, ensure_ascii=False) if event.args is not None else event.output
    text = redact_secrets(" ".join((raw or "").split()))
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1] + "…"


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


class SessionTracker:
    def __init__(
        self,
        redis: Redis,
        *,
        ttl_s: int = 86_400,
        history_len: int = 20,
        risk_half_life_s: float = 600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._r = redis
        self.ttl_s = ttl_s
        self.history_len = history_len
        self.half_life_s = risk_half_life_s
        self._clock = clock

    @classmethod
    def from_settings(cls, redis: Redis, settings: Settings) -> SessionTracker:
        return cls(
            redis,
            ttl_s=settings.session_ttl_s,
            history_len=settings.session_history_len,
            risk_half_life_s=settings.session_risk_half_life_s,
        )

    @staticmethod
    def _k(session_id: str, part: str) -> str:
        return f"{KEY_PREFIX}:{session_id}:{part}"

    # ------------------------------------------------------------------ writes

    async def record(
        self, event: Event, decision: Decision, *, request: Event | None = None
    ) -> None:
        """Record one decided event. For outputs, pass the originating request event."""
        sid = event.session_id
        k = {p: self._k(sid, p) for p in ("meta", "history", "files", "taint", "egress")}
        compact = CompactEvent(
            ts=event.ts,
            direction=event.direction,
            tool=event.qualified_tool,
            preview=_preview(event),
            action=decision.action,
            risk=decision.risk,
        )
        passed = decision.action in _PASSED
        files: set[str] = set()
        taint: set[str] = set()
        egress: set[str] = set()
        if event.direction is Direction.REQUEST and event.args is not None:
            egress = egress_targets(event.args)
            if passed:
                files = set(paths_in_args(event.args))
        elif event.output is not None and passed:
            taint = secret_content_labels(event.output)
            if request is not None and request.args is not None:
                for path in paths_in_args(request.args):
                    taint |= secret_path_labels(path)

        async with self._r.pipeline(transaction=True) as pipe:
            pipe.rpush(k["history"], compact.model_dump_json())
            pipe.ltrim(k["history"], -self.history_len, -1)
            pipe.hincrby(k["meta"], "event_count", 1)
            if files:
                pipe.sadd(k["files"], *sorted(files))
            if taint:
                pipe.sadd(k["taint"], *sorted(taint))
            if egress:
                pipe.sadd(k["egress"], *sorted(egress))
            for key in k.values():
                pipe.expire(key, self.ttl_s)
            await pipe.execute()
        if decision.risk > 0:
            await self._add_risk(sid, decision.risk)

    async def _add_risk(self, session_id: str, risk: float) -> float:
        out = await self._r.eval(
            _RISK_LUA,
            1,
            self._k(session_id, "meta"),
            repr(risk),
            repr(self._clock()),
            repr(self.half_life_s),
            str(self.ttl_s),
        )
        return float(_s(out))

    async def revoke_tool(self, session_id: str, qualified_tool: str) -> None:
        key = self._k(session_id, "revoked")
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.sadd(key, qualified_tool)
            pipe.expire(key, self.ttl_s)
            await pipe.execute()

    async def terminate(self, session_id: str, reason: str) -> None:
        key = self._k(session_id, "meta")
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping={"terminated": "1", "terminated_reason": reason[:500]})
            pipe.expire(key, self.ttl_s)
            await pipe.execute()

    async def observe_description(
        self, session_id: str, qualified_tool: str, sha256: str
    ) -> DescriptionObservation:
        """Atomically store the tool's description hash and report whether it changed."""
        prev = await self._r.eval(
            _DESC_LUA, 1, self._k(session_id, "desc"), qualified_tool, sha256, str(self.ttl_s)
        )
        previous = None if prev is None else _s(prev)
        return DescriptionObservation(
            first_seen=previous is None,
            changed=previous is not None and previous != sha256,
            previous_sha256=previous,
        )

    # ------------------------------------------------------------------ reads

    async def is_terminated(self, session_id: str) -> bool:
        return _s(await self._r.hget(self._k(session_id, "meta"), "terminated") or b"0") == "1"

    async def is_revoked(self, session_id: str, qualified_tool: str) -> bool:
        return bool(await self._r.sismember(self._k(session_id, "revoked"), qualified_tool))

    async def get(self, session_id: str) -> SessionState:
        k = {
            p: self._k(session_id, p)
            for p in ("meta", "history", "files", "taint", "egress", "revoked")
        }
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.hgetall(k["meta"])
            pipe.lrange(k["history"], 0, -1)
            pipe.smembers(k["files"])
            pipe.smembers(k["taint"])
            pipe.smembers(k["egress"])
            pipe.smembers(k["revoked"])
            meta_raw, hist, files, taint, egress, revoked = await pipe.execute()
        meta = {_s(a): _s(b) for a, b in meta_raw.items()}
        risk = 0.0
        if "risk_accum" in meta:
            dt = max(0.0, self._clock() - float(meta["risk_ts"]))
            risk = float(meta["risk_accum"]) * math.pow(0.5, dt / self.half_life_s)
        taint_list = sorted(_s(x) for x in taint)
        return SessionState(
            session_id=session_id,
            event_count=int(meta.get("event_count", "0")),
            history=[CompactEvent.model_validate_json(_s(h)) for h in hist],
            files_read=sorted(_s(x) for x in files),
            secret_taint=bool(taint_list),
            taint_sources=taint_list,
            egress_targets=sorted(_s(x) for x in egress),
            risk=risk,
            revoked_tools=sorted(_s(x) for x in revoked),
            terminated=meta.get("terminated") == "1",
            terminated_reason=meta.get("terminated_reason"),
        )


def summary(state: SessionState, *, max_recent: int = 5) -> str:
    """Short, secret-free text describing the session (used in the guard prompt later)."""

    def _list(items: list[str], limit: int = 6) -> str:
        if not items:
            return "none"
        shown = ", ".join(items[:limit])
        return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")

    lines = [
        f"Events so far: {state.event_count}. Cumulative risk: {state.risk:.2f}.",
        f"Files read: {_list(state.files_read)}.",
        (
            f"Secret taint: YES ({_list(state.taint_sources)})."
            if state.secret_taint
            else "Secret taint: no."
        ),
        f"Egress targets: {_list(state.egress_targets)}.",
        f"Revoked tools: {_list(state.revoked_tools)}.",
    ]
    if state.terminated:
        lines.append(f"Session TERMINATED: {state.terminated_reason or 'no reason recorded'}.")
    recent = state.history[-max_recent:]
    if recent:
        lines.append("Recent events:")
        lines += [
            f"- [{e.direction.value}] {e.tool} {e.preview} -> {e.action.value} ({e.risk:.2f})"
            for e in recent
        ]
    text = "\n".join(lines)
    return text if len(text) <= SUMMARY_MAX_CHARS else text[: SUMMARY_MAX_CHARS - 1] + "…"


def create_redis(settings: Settings) -> Redis:
    """Redis client with explicit timeouts (never hang the request path)."""
    client: Redis = Redis.from_url(
        settings.redis_url,
        socket_connect_timeout=settings.redis_connect_timeout_s,
        socket_timeout=settings.redis_socket_timeout_s,
        health_check_interval=30,
    )
    return client
