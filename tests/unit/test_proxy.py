"""End-to-end proxy tests: real mcp v2 client -> gateway (stateful) -> in-process tool servers.

Everything is real except the network: policy, decider, enforcer, session tracker
(fakeredis), decision log (SQLite) and the upstream MCP servers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import fakeredis
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from gateway.approvals import ApprovalStore
from gateway.config import Settings
from gateway.db.models import Base, DecisionRow
from gateway.decider import PipelineDecider
from gateway.enforcer import Enforcer
from gateway.models import ApprovalStatus
from gateway.policy import PolicyEngine
from gateway.proxy import GatewayProxy, build_mcp_app
from gateway.recorder import DecisionRecorder
from gateway.session import SessionTracker
from gateway.session_identity import McpSessionIdResolver
from gateway.upstream import UpstreamManager
from tools_servers.email_send.server import build_server as build_email
from tools_servers.filesystem.server import build_server as build_fs
from tools_servers.poisoned_desc.server import build_server as build_weather
from tools_servers.poisoned_output.server import build_server as build_notes
from tools_servers.sandbox import SEED_DIR, prepare_sandbox

REPO = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8000"
ENV_TEXT = (SEED_DIR / ".env").read_text(encoding="utf-8")


@dataclass
class Stack:
    app: Any
    store: ApprovalStore
    recorder: DecisionRecorder
    engine: AsyncEngine
    tracker: SessionTracker
    upstreams: UpstreamManager

    @asynccontextmanager
    async def client(self, **kw: Any) -> AsyncIterator[Client]:
        hc = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=self.app), base_url=BASE)
        async with hc, Client(streamable_http_client(f"{BASE}/mcp", http_client=hc), **kw) as c:
            yield c

    async def rows(self) -> list[Any]:
        await asyncio.wait_for(self.recorder._queue.join(), 10)
        async with self.engine.connect() as conn:
            return list((await conn.execute(select(DecisionRow).order_by(DecisionRow.id))).all())


@asynccontextmanager
async def gateway(tmp_path: Path, *, shadow: bool = False, weather_mode: str = "rugpull",
                  sep: str = ".") -> AsyncIterator[Stack]:  # fmt: skip
    settings = Settings(
        shadow_mode=shadow,
        tool_name_separator=sep,
        approval_timeout_s=5,
        approval_poll_interval_s=0.02,
        approval_progress_interval_s=0.05,
    )
    redis = fakeredis.FakeAsyncRedis()
    tracker = SessionTracker(redis)
    store = ApprovalStore(redis)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'log.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    recorder = DecisionRecorder(engine)
    policy = PolicyEngine.load(REPO / "policies" / "default.yaml")
    upstreams = UpstreamManager({
        "filesystem": build_fs(prepare_sandbox(tmp_path / "fs")),
        "weather": build_weather(weather_mode, tmp_path / "poison.jsonl"),  # type: ignore[arg-type]
        "notes": build_notes(),
        "email_send": build_email(tmp_path / "outbox.jsonl", ("acme.example",)),
    })  # fmt: skip
    enforcer = Enforcer(tracker, store, shadow_mode=shadow, recorder=recorder,
                        approval_timeout_s=settings.approval_timeout_s,
                        poll_interval_s=settings.approval_poll_interval_s,
                        progress_interval_s=settings.approval_progress_interval_s)  # fmt: skip
    proxy = GatewayProxy(settings=settings, upstreams=upstreams,
                         decider=PipelineDecider(tracker, policy), enforcer=enforcer,
                         tracker=tracker, policy=policy, resolver=McpSessionIdResolver(),
                         recorder=recorder)  # fmt: skip
    app, inner = build_mcp_app(proxy, settings)
    await recorder.start()
    await upstreams.start()
    try:
        async with inner.router.lifespan_context(inner):
            with anyio.fail_after(60):
                yield Stack(app, store, recorder, engine, tracker, upstreams)
    finally:
        await upstreams.stop()
        await recorder.stop()
        await engine.dispose()
        await redis.aclose()


def text(res: CallToolResult) -> str:
    return "".join(c.text for c in res.content if isinstance(c, TextContent))


# --- listing ------------------------------------------------------------------------------


async def test_tools_are_namespaced_and_session_is_stateful(tmp_path: Path) -> None:
    async with gateway(tmp_path) as gw, gw.client() as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert c.protocol_version == "2025-11-25"  # auto client fell back to stateful
    assert {"filesystem.read_file", "weather.get_forecast", "notes.get_ticket",
            "email_send.send"} <= names  # fmt: skip


async def test_custom_separator(tmp_path: Path) -> None:
    async with gateway(tmp_path, sep="__") as gw, gw.client() as c:
        names = {t.name for t in (await c.list_tools()).tools}
        res = await c.call_tool("filesystem__read_file", {"path": "docs/readme.md"})
    assert "filesystem__read_file" in names and not res.is_error


async def test_poisoned_description_is_sanitized_and_logged_escalated(tmp_path: Path) -> None:
    async with gateway(tmp_path, weather_mode="static") as gw:
        async with gw.client() as c:
            tools = {t.name: t.description for t in (await c.list_tools()).tools}
        rows = await gw.rows()
    desc = tools["weather.get_forecast"] or ""
    assert "[REMOVED BY SENTINEL]" in desc and ".env" not in desc
    row = next(r for r in rows if r.qualified_tool == "weather.get_forecast")
    assert (row.kind, row.action, row.rule_id, row.escalated) == (
        "tool_description", "SANITIZE", "output-injection-weak", True,
    )  # fmt: skip


async def test_rugpull_revokes_tool_for_the_session(tmp_path: Path) -> None:
    async with gateway(tmp_path) as gw, gw.client() as c:
        first = {t.name for t in (await c.list_tools()).tools}
        second = {t.name for t in (await c.list_tools()).tools}
        blocked = await c.call_tool("weather.get_forecast", {"city": "Pune"})
        other = await c.call_tool("weather.convert_currency",
                                  {"amount": 1, "from_code": "usd", "to_code": "inr"})  # fmt: skip
    assert "weather.get_forecast" in first and "weather.get_forecast" not in second
    assert blocked.is_error and "revoked" in text(blocked)
    assert not other.is_error


# --- calls --------------------------------------------------------------------------------


async def test_benign_call_passes_through(tmp_path: Path) -> None:
    async with gateway(tmp_path) as gw, gw.client() as c:
        res = await c.call_tool("filesystem.read_file", {"path": "docs/readme.md"})
    assert not res.is_error and "Acme Notes" in text(res)


async def test_key_material_blocked_with_reason(tmp_path: Path) -> None:
    async with gateway(tmp_path) as gw, gw.client() as c:
        res = await c.call_tool("filesystem.read_file", {"path": ".ssh/id_rsa"})
    assert res.is_error and "private key material" in text(res)
    assert "BEGIN" not in text(res)


async def test_multistep_exfil_via_weather_context_terminates(tmp_path: Path) -> None:
    async with gateway(tmp_path) as gw:
        async with gw.client() as c:
            env = await c.call_tool("filesystem.read_file", {"path": ".env"})  # allowed + taint
            leak = await c.call_tool("weather.get_forecast", {"city": "Pune", "context": text(env)})
            after = await c.call_tool("filesystem.read_file", {"path": "docs/readme.md"})
        async with gw.client() as other:  # a new connection is a new, clean session
            fresh = await other.call_tool("filesystem.read_file", {"path": "docs/readme.md"})
        rows = await gw.rows()
    assert not env.is_error and "FAKE_API_KEY" in text(env)
    assert leak.is_error and "terminated" in text(leak)
    assert after.is_error and "terminated" in text(after)
    assert not fresh.is_error
    assert not (tmp_path / "poison.jsonl").exists()  # the attacker never received the secret
    term = next(r for r in rows if r.action == "TERMINATE_SESSION")
    assert term.rule_id == "secret-flow" and term.enforced
    for r in rows:  # decision log never holds the raw secret
        assert "sk-fake-0000" not in str(r.args_redacted) + str(r.output_preview)


async def test_injected_outputs_are_sanitized(tmp_path: Path) -> None:
    async with gateway(tmp_path) as gw, gw.client() as c:
        ticket = await c.call_tool("notes.get_ticket", {"ticket_id": "T-2002"})
        notes = await c.call_tool("notes.search_notes", {"query": ""})
        clean = await c.call_tool("notes.get_ticket", {"ticket_id": "T-2001"})
    assert "DELETE FROM customers" not in text(ticket) and "[REMOVED BY SENTINEL]" in text(ticket)
    assert "mailmove-support" not in text(notes) and "Sprint 14 retro" in text(notes)
    assert "[REMOVED BY SENTINEL]" not in text(clean)


async def test_shadow_mode_logs_but_forwards(tmp_path: Path) -> None:
    async with gateway(tmp_path, shadow=True) as gw:
        async with gw.client() as c:
            res = await c.call_tool("filesystem.read_file", {"path": ".ssh/id_rsa"})
        rows = await gw.rows()
    assert not res.is_error and "PRIVATE KEY" in text(res)
    row = next(r for r in rows if r.args_redacted == {"path": ".ssh/id_rsa"})
    assert (row.action, row.enforced, row.rule_id) == ("BLOCK", False, "key-material-paths")


async def test_approval_flow_with_progress(tmp_path: Path) -> None:
    messages: list[str] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        messages.append(message or "")

    async def approve_when_pending(store: ApprovalStore) -> None:
        for _ in range(500):
            pending = await store.list_pending()
            if pending:
                await asyncio.sleep(0.15)
                await store.transition(pending[0].approval_id, ApprovalStatus.APPROVED, by="kd")
                return
            await asyncio.sleep(0.01)

    async with gateway(tmp_path) as gw, gw.client() as c:
        args = {"to": "partner@vendor.example", "subject": "hi", "body": "meeting notes"}
        res, _ = await asyncio.gather(
            c.call_tool("email_send.send", args, progress_callback=on_progress),
            approve_when_pending(gw.store),
        )
    assert not res.is_error and "queued message" in text(res)
    assert any("Waiting for human approval" in m for m in messages)


async def test_every_logged_decision_has_required_fields(tmp_path: Path) -> None:
    async with gateway(tmp_path) as gw:
        async with gw.client() as c:
            await c.list_tools()
            await c.call_tool("filesystem.read_file", {"path": "docs/readme.md"})
        rows = await gw.rows()
    assert rows
    for r in rows:  # CLAUDE.md 7.9
        for field in ("session_id", "event_id", "direction", "qualified_tool", "tier", "action",
                      "risk", "reason", "latency_ms", "model_version"):  # fmt: skip
            assert getattr(r, field) is not None, field


# --- L8: rug-pull seen only through an SDK-silent list during a call ----------------------


def _force_silent_relist(gw: Stack, server: str) -> None:
    """Test-only: drop the upstream client's cached output schemas (private SDK attribute)
    so its next call_tool lists tools silently, as it does for an uncached tool."""
    client = gw.upstreams._up[server].client
    assert client is not None
    client.session._tool_output_schemas.clear()


async def test_rugpull_mid_call_blocks_output_and_revokes(tmp_path: Path) -> None:
    async with gateway(tmp_path) as gw:
        async with gw.client() as c:
            listed = {t.name for t in (await c.list_tools()).tools}  # clean description seen
            _force_silent_relist(gw, "weather")
            res = await c.call_tool("weather.get_forecast", {"city": "Pune"})  # poisoned mid-call
            again = await c.call_tool("weather.get_forecast", {"city": "Pune"})
        rows = await gw.rows()
    assert "weather.get_forecast" in listed
    assert res.is_error and "changed during this call" in text(res)
    assert "Day 1" not in text(res)  # the upstream result never reaches the agent
    assert again.is_error and "revoked" in text(again)
    row = next(r for r in rows if r.rule_id == "builtin:description-changed-mid-call")
    assert (row.direction, row.action, row.enforced) == ("output", "BLOCK", True)


async def test_rugpull_mid_call_shadow_mode_only_logs(tmp_path: Path) -> None:
    async with gateway(tmp_path, shadow=True) as gw:
        async with gw.client() as c:
            await c.list_tools()
            _force_silent_relist(gw, "weather")
            res = await c.call_tool("weather.get_forecast", {"city": "Pune"})
        rows = await gw.rows()
    assert not res.is_error and "Day 1" in text(res)
    row = next(r for r in rows if r.rule_id == "builtin:description-changed-mid-call")
    assert row.enforced is False
