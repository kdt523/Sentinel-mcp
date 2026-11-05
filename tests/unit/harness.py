"""Shared harness: the complete gateway app, in-process, with every sandbox tool server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import fakeredis
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.app import GatewayApp, create_app
from gateway.config import Settings
from gateway.db.models import Base
from gateway.db.sink import create_sink
from tools_servers.email_send.server import build_server as build_email
from tools_servers.filesystem.server import build_server as build_fs
from tools_servers.poisoned_desc.server import build_server as build_weather
from tools_servers.poisoned_output.server import build_server as build_notes
from tools_servers.sandbox import prepare_sandbox

REPO = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8000"


@asynccontextmanager
async def gateway_app(tmp_path: Path, **settings_kw: Any) -> AsyncIterator[GatewayApp]:
    """Fresh gateway + fresh tool servers (the rug-pull server starts clean every time)."""
    settings = Settings(
        policy_path=REPO / "policies" / "default.yaml",
        shadow_mode=False,
        approval_timeout_s=2,
        approval_poll_interval_s=0.02,
        **settings_kw,
    )
    db = tmp_path / "log.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    targets = {
        "filesystem": build_fs(prepare_sandbox(tmp_path / "fs")),
        "email_send": build_email(tmp_path / "outbox.jsonl", ("acme.example",)),
        "weather": build_weather("rugpull", tmp_path / "poison.jsonl"),
        "notes": build_notes(),
    }
    app = create_app(settings, redis=fakeredis.FakeAsyncRedis(),
                     sink=create_sink(f"sqlite:///{db}"), upstream_targets=targets)  # fmt: skip
    async with app.api.router.lifespan_context(app.api):
        with anyio.fail_after(90):
            yield app


@asynccontextmanager
async def agent_client(app: GatewayApp) -> AsyncIterator[Client]:
    """A real mcp v2 client (auto mode) talking HTTP to the in-process gateway."""
    hc = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE)
    async with hc, Client(streamable_http_client(f"{BASE}/mcp", http_client=hc)) as client:
        yield client
