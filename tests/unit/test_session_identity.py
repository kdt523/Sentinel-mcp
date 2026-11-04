"""Session identity: stateful Mcp-Session-Id resolver + refusal of the stateless MCP era.

These tests PIN mcp v2 client/server behaviour the gateway's security model depends on
(docs/threat-model.md section 3). If an SDK upgrade changes it, they must fail loudly.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from starlette.applications import Starlette

from gateway.session_identity import (
    STATELESS_REFUSAL,
    McpSessionIdResolver,
    SessionResolutionError,
    StatefulOnlyMiddleware,
)

BASE = "http://127.0.0.1:8000"


class Recorder:
    def __init__(self) -> None:
        self.sids: list[str] = []

    def _resolve(self, ctx: Any) -> None:
        try:
            self.sids.append(McpSessionIdResolver().resolve(ctx))
        except SessionResolutionError:
            self.sids.append("REFUSED")

    async def list_tools(self, ctx: Any, params: Any) -> ListToolsResult:
        self._resolve(ctx)
        tool = Tool(name="a.b", description="d", input_schema={"type": "object"})
        return ListToolsResult(tools=[tool])

    async def call_tool(self, ctx: Any, params: Any) -> CallToolResult:
        self._resolve(ctx)
        return CallToolResult(content=[TextContent(type="text", text="ok")])


@asynccontextmanager
async def stack() -> AsyncIterator[tuple[Recorder, Starlette]]:
    """Opened inside each test: anyio task groups must close in the task that opened them."""
    rec = Recorder()
    server = Server("gw", on_list_tools=rec.list_tools, on_call_tool=rec.call_tool)
    app = server.streamable_http_app()
    async with app.router.lifespan_context(app):
        yield rec, app


async def run_client(app: Any, mode: str, calls: int = 2) -> str:
    hc = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE)
    with anyio.fail_after(15):  # a refusal must never turn into a hang
        transport = streamable_http_client(f"{BASE}/mcp", http_client=hc)
        async with hc, Client(transport, mode=mode) as c:
            await c.list_tools()
            for _ in range(calls):
                await c.call_tool("a.b", {})
            return str(c.protocol_version)


def leaf(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc


async def test_pin_default_client_is_stateless_without_guard() -> None:
    """Why the guard exists: by default mcp v2 negotiates the stateless era (no session)."""
    async with stack() as (rec, app):
        assert await run_client(app, "auto") == "2026-07-28"
    assert set(rec.sids) == {"REFUSED"}


async def test_pin_auto_client_falls_back_to_stateful_behind_guard() -> None:
    async with stack() as (rec, app):
        assert await run_client(StatefulOnlyMiddleware(app), "auto") == "2025-11-25"
    assert len(rec.sids) == 3 and len(set(rec.sids)) == 1 and rec.sids[0] != "REFUSED"


async def test_each_connection_gets_its_own_session() -> None:
    async with stack() as (rec, app):
        guarded = StatefulOnlyMiddleware(app)
        await run_client(guarded, "auto", calls=0)
        await run_client(guarded, "legacy", calls=0)
    assert len(set(rec.sids)) == 2 and "REFUSED" not in rec.sids


async def test_modern_only_client_refused_with_reason() -> None:
    async with stack() as (rec, app):
        with pytest.raises(BaseException) as info:
            await run_client(StatefulOnlyMiddleware(app), "2026-07-28")
    err = leaf(info.value)
    assert isinstance(err, MCPError), repr(err)
    assert "requires a stateful MCP session" in str(err)
    assert "stateless 2026-07-28 protocol is not supported" in str(err)
    assert rec.sids == []  # never reached the MCP app


async def test_refusal_echoes_request_ids_including_batches() -> None:
    async with stack() as (_, app):
        transport = httpx2.ASGITransport(app=StatefulOnlyMiddleware(app))
        async with httpx2.AsyncClient(transport=transport, base_url=BASE) as hc:
            single = await hc.post(
                "/mcp", json={"jsonrpc": "2.0", "id": 7, "method": "server/discover"}
            )
            batch = await hc.post(
                "/mcp",
                headers={"MCP-Protocol-Version": "2026-07-28"},
                json=[
                    {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    {"jsonrpc": "2.0", "method": "notifications/x"},
                    {"jsonrpc": "2.0", "id": "b", "method": "tools/call"},
                ],
            )
    assert single.status_code == 200  # an RPC error, not a transport error -> client falls back
    assert single.json()["id"] == 7
    assert single.json()["error"]["message"] == STATELESS_REFUSAL
    assert [m["id"] for m in batch.json()] == [1, "b"]


def test_resolver_requires_session_header() -> None:
    class Ctx:
        request = None

    with pytest.raises(SessionResolutionError, match="requires a stateful MCP session"):
        McpSessionIdResolver().resolve(Ctx())  # type: ignore[arg-type]
