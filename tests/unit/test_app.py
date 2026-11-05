"""The assembled gateway app: probes, admin auth, approvals API and session inspection."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import fakeredis
import httpx2
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.app import GatewayApp, create_app
from gateway.config import Settings
from gateway.db.models import ApprovalRow, Base
from tools_servers.email_send.server import build_server as build_email
from tools_servers.filesystem.server import build_server as build_fs
from tools_servers.sandbox import prepare_sandbox

REPO = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8000"
KEY = "fake-admin-key-for-tests"
AUTH = {"Authorization": f"Bearer {KEY}"}


def sink_for(engine: Any) -> Any:
    """Synchronous decision-log sink writing to the same SQLite file as ``engine``."""
    from gateway.db.sink import create_sink

    return create_sink(engine.url.render_as_string(hide_password=False))


async def _query(tmp_path: Path, stmt: Any) -> Any:
    """Run one read against the decision-log SQLite file, then dispose the engine (no
    leaked aiosqlite threads outliving the event loop)."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'log.db'}")
    try:
        async with engine.connect() as conn:
            return (await conn.execute(stmt)).all()
    finally:
        await engine.dispose()


@asynccontextmanager
async def running(tmp_path: Path, *, key: str | None = KEY,
                  extra: dict[str, Any] | None = None) -> AsyncIterator[GatewayApp]:  # fmt: skip
    settings = Settings(
        admin_api_key=SecretStr(key) if key else None,
        shadow_mode=False,
        policy_path=REPO / "policies" / "default.yaml",
        approval_timeout_s=10,
        approval_poll_interval_s=0.02,
    )
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'log.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sink = sink_for(engine)
    await engine.dispose()  # schema created; the app writes through its own sync sink
    targets: dict[str, Any] = {
        "filesystem": build_fs(prepare_sandbox(tmp_path / "fs")),
        "email_send": build_email(tmp_path / "outbox.jsonl", ("acme.example",)),
        **(extra or {}),
    }
    app = create_app(settings, redis=fakeredis.FakeAsyncRedis(), sink=sink,
                     upstream_targets=targets)  # fmt: skip
    async with app.api.router.lifespan_context(app.api):
        with anyio.fail_after(60):
            yield app


def http(app: GatewayApp) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE)


@asynccontextmanager
async def mcp_client(app: GatewayApp) -> AsyncIterator[Client]:
    hc = http(app)
    async with hc, Client(streamable_http_client(f"{BASE}/mcp", http_client=hc)) as c:
        yield c


def text(res: Any) -> str:
    return "".join(c.text for c in res.content if isinstance(c, TextContent))


# --- probes & auth ---------------------------------------------------------------------------


async def test_probes_are_open_and_minimal(tmp_path: Path) -> None:
    async with running(tmp_path) as app, http(app) as c:
        health = await c.get("/healthz")
        ready = await c.get("/readyz")
    assert health.status_code == 200 and health.json() == {"status": "ok"}
    assert ready.status_code == 200 and ready.json() == {"status": "ready"}  # status only


async def test_readyz_reports_not_ready_when_an_upstream_is_down(tmp_path: Path) -> None:
    broken = {"broken": StdioServerParameters(command=str(REPO / "missing.exe"), args=[])}
    async with running(tmp_path, extra=broken) as app, http(app) as c:
        ready = await c.get("/readyz")
        details = await c.get("/readyz/details", headers=AUTH)
    assert ready.status_code == 503 and ready.json() == {"status": "not_ready"}
    body = details.json()
    assert details.status_code == 503 and body["upstreams"]["broken"]["connected"] is False
    assert body["redis"]["ok"] and body["database"]["ok"] and body["policy"].startswith("policy:")


async def test_admin_auth(tmp_path: Path) -> None:
    async with running(tmp_path) as app, http(app) as c:
        missing = await c.get("/readyz/details")
        wrong = await c.get("/readyz/details", headers={"Authorization": "Bearer nope"})
        bearer = await c.get("/readyz/details", headers=AUTH)
        header = await c.get("/approvals", headers={"X-API-Key": KEY})
    assert missing.status_code == wrong.status_code == 401
    assert bearer.status_code == 200 and header.status_code == 200


async def test_admin_disabled_without_key(tmp_path: Path) -> None:
    async with running(tmp_path, key=None) as app, http(app) as c:
        r = await c.get("/approvals", headers=AUTH)
        probe = await c.get("/readyz")
    assert r.status_code == 503 and "disabled" in r.json()["detail"]
    assert probe.status_code == 200


# --- approvals -------------------------------------------------------------------------------


async def _wait_pending(c: httpx2.AsyncClient) -> dict[str, Any]:
    for _ in range(500):
        body = (await c.get("/approvals", headers=AUTH)).json()
        if body["count"]:
            approval: dict[str, Any] = body["approvals"][0]
            return approval
        await asyncio.sleep(0.01)
    raise AssertionError("no pending approval appeared")


async def test_approve_records_approver_note_and_conflicts(tmp_path: Path) -> None:
    async with running(tmp_path) as app, http(app) as admin, mcp_client(app) as agent:
        args = {"to": "partner@vendor.example", "subject": "s", "body": "agenda"}
        call = asyncio.create_task(agent.call_tool("email_send.send", args))
        pending = await _wait_pending(admin)
        aid = pending["approval_id"]
        ok = await admin.post(f"/approvals/{aid}/approve", headers=AUTH,
                              json={"approver": "kd", "note": "known vendor"})  # fmt: skip
        again = await admin.post(f"/approvals/{aid}/deny", headers=AUTH, json={"approver": "kd"})
        unknown = await admin.post("/approvals/nope/approve", headers=AUTH, json={"approver": "kd"})
        bad = await admin.post(f"/approvals/{aid}/approve", headers=AUTH, json={})
        res = await call
        await asyncio.wait_for(app.api.state.admin.recorder._queue.join(), 10)
        [row] = await _query(tmp_path, select(ApprovalRow))
    assert pending["status"] == "pending" and pending["qualified_tool"] == "email_send.send"
    assert ok.status_code == 200 and ok.json()["status"] == "approved"
    assert ok.json()["resolved_by"] == "kd" and ok.json()["note"] == "known vendor"
    assert again.status_code == 409 and "approved, not pending" in again.json()["detail"]
    assert unknown.status_code == 404 and bad.status_code == 422
    assert not res.is_error
    assert (row.status, row.resolved_by, row.note) == ("approved", "kd", "known vendor")


async def test_deny_reaches_agent_with_reason(tmp_path: Path) -> None:
    async with running(tmp_path) as app, http(app) as admin, mcp_client(app) as agent:
        args = {"to": "x@vendor.example", "subject": "s", "body": "b"}
        call = asyncio.create_task(agent.call_tool("email_send.send", args))
        aid = (await _wait_pending(admin))["approval_id"]
        await admin.post(f"/approvals/{aid}/deny", headers=AUTH,
                         json={"approver": "kd", "note": "not approved vendor"})  # fmt: skip
        res = await call
    assert res.is_error and "rejected by kd (not approved vendor)" in text(res)
    assert not (tmp_path / "outbox.jsonl").exists()


# --- sessions --------------------------------------------------------------------------------


async def test_session_view_is_redacted(tmp_path: Path) -> None:
    async with running(tmp_path) as app, http(app) as admin:
        async with mcp_client(app) as agent:
            await agent.call_tool("filesystem.read_file", {"path": ".env"})
        await asyncio.wait_for(app.api.state.admin.recorder._queue.join(), 10)
        from gateway.db.models import DecisionRow

        sid = (await _query(tmp_path, select(DecisionRow.session_id)))[0][0]
        view = await admin.get(f"/sessions/{sid}", headers=AUTH)
        missing = await admin.get("/sessions/nope", headers=AUTH)
        noauth = await admin.get(f"/sessions/{sid}")
    body = view.json()
    assert view.status_code == 200 and body["secret_taint"] is True
    assert "path:dotenv" in body["taint_sources"] and "Taint: SECRET" in body["summary"]
    assert "sk-fake-0000" not in view.text and "fake-db-password" not in view.text
    assert missing.status_code == 404 and noauth.status_code == 401
