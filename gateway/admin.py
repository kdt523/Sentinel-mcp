"""Admin and probe HTTP API.

Unauthenticated (for orchestrator probes): ``GET /healthz``, ``GET /readyz`` (status only).
Everything else requires the admin API key (``Authorization: Bearer <key>`` or
``X-API-Key``), compared in constant time. With no key configured, admin routes are
disabled (503) rather than open.
"""

from __future__ import annotations

import asyncio
import hmac
from dataclasses import dataclass
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis

from gateway.approvals import Approval, ApprovalStore
from gateway.config import Settings
from gateway.detectors import redact_deep
from gateway.models import ApprovalStatus
from gateway.policy import PolicyEngine
from gateway.recorder import DecisionRecorder
from gateway.session import SessionTracker, summary
from gateway.upstream import UpstreamManager

log = structlog.get_logger("sentinel.admin")
PROBE_TIMEOUT_S = 1.0


@dataclass
class AdminDeps:
    settings: Settings
    redis: Redis
    tracker: SessionTracker
    approvals: ApprovalStore
    recorder: DecisionRecorder
    upstreams: UpstreamManager
    policy: PolicyEngine


class ResolveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approver: str = Field(min_length=1, max_length=128)
    note: str | None = Field(default=None, max_length=500)


def _deps(request: Request) -> AdminDeps:
    deps: AdminDeps = request.app.state.admin
    return deps


def require_admin(request: Request, deps: AdminDeps = Depends(_deps)) -> None:  # noqa: B008
    key = deps.settings.admin_api_key
    if key is None:
        raise HTTPException(503, "admin API disabled: SENTINEL_ADMIN_API_KEY is not set")
    auth = request.headers.get("authorization", "")
    provided = (
        auth[7:] if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    )
    if not hmac.compare_digest(provided.encode(), key.get_secret_value().encode()):
        raise HTTPException(401, "invalid or missing admin API key")


async def _check(coro: Any) -> str | None:
    """None when healthy, else a short error label (never a secret / connection string)."""
    try:
        await asyncio.wait_for(coro, PROBE_TIMEOUT_S)
        return None
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__


async def _readiness(deps: AdminDeps) -> dict[str, Any]:
    redis_err, db_err = await asyncio.gather(
        _check(deps.redis.ping()), _check(deps.recorder.ping())
    )
    upstreams = deps.upstreams.status()
    # Deciding needs Redis and the upstreams; the decision log is async and failure-tolerant,
    # so the database is reported but does not gate readiness.
    ready = redis_err is None and all(u["connected"] for u in upstreams.values())
    return {
        "ready": ready,
        "redis": {"ok": redis_err is None, "error": redis_err},
        "database": {"ok": db_err is None, "error": db_err},
        "upstreams": upstreams,
        "decision_log": {
            "pending": deps.recorder.pending,
            "written": deps.recorder.written,
            "dropped": deps.recorder.dropped,
            "failed": deps.recorder.failed,
            "raw_capture": deps.recorder.capturing,
        },
        "shadow_mode": deps.settings.shadow_mode,
        "policy": deps.policy.model_version,
    }


def _approval_json(a: Approval) -> dict[str, Any]:
    data: dict[str, Any] = redact_deep(a.model_dump(mode="json"))
    return data


def build_admin_router() -> APIRouter:
    router = APIRouter()
    admin = [Depends(require_admin)]

    @router.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    async def readyz(deps: AdminDeps = Depends(_deps)) -> JSONResponse:  # noqa: B008
        ready = (await _readiness(deps))["ready"]
        return JSONResponse({"status": "ready" if ready else "not_ready"},
                            status_code=200 if ready else 503)  # fmt: skip

    @router.get("/readyz/details", dependencies=admin)
    async def readyz_details(deps: AdminDeps = Depends(_deps)) -> JSONResponse:  # noqa: B008
        body = await _readiness(deps)
        return JSONResponse(body, status_code=200 if body["ready"] else 503)

    @router.get("/approvals", dependencies=admin)
    async def list_approvals(deps: AdminDeps = Depends(_deps)) -> dict[str, Any]:  # noqa: B008
        pending = await deps.approvals.list_pending()
        return {"count": len(pending), "approvals": [_approval_json(a) for a in pending]}

    async def _resolve(approval_id: str, to: ApprovalStatus, body: ResolveBody,
                       deps: AdminDeps) -> dict[str, Any]:  # fmt: skip
        res = await deps.approvals.transition(approval_id, to, by=body.approver,
                                              note=body.note or "")  # fmt: skip
        if res.status is None:
            raise HTTPException(404, "approval not found")
        if not res.ok:
            raise HTTPException(409, f"approval is {res.status.value}, not pending")
        approval = await deps.approvals.get(approval_id)
        assert approval is not None
        deps.recorder.submit_approval(approval.db_row())
        log.info("approval_resolved", approval_id=approval_id, status=to.value,
                 approver=body.approver)  # fmt: skip
        return _approval_json(approval)

    @router.post("/approvals/{approval_id}/approve", dependencies=admin)
    async def approve(approval_id: str, body: ResolveBody,
                      deps: AdminDeps = Depends(_deps)) -> dict[str, Any]:  # noqa: B008  # fmt: skip
        return await _resolve(approval_id, ApprovalStatus.APPROVED, body, deps)

    @router.post("/approvals/{approval_id}/deny", dependencies=admin)
    async def deny(approval_id: str, body: ResolveBody,
                   deps: AdminDeps = Depends(_deps)) -> dict[str, Any]:  # noqa: B008  # fmt: skip
        return await _resolve(approval_id, ApprovalStatus.DENIED, body, deps)

    @router.get("/sessions/{session_id}", dependencies=admin)
    async def get_session(session_id: str,
                          deps: AdminDeps = Depends(_deps)) -> dict[str, Any]:  # noqa: B008  # fmt: skip
        state = await deps.tracker.get(session_id)
        if state.event_count == 0 and not state.terminated and not state.revoked_tools:
            raise HTTPException(404, "unknown session")
        # Session state holds labels and redacted previews only; redact again defensively.
        data: dict[str, Any] = redact_deep(state.model_dump(mode="json"))
        data["summary"] = summary(state)
        return data

    return router
