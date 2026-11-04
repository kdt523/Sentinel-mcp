"""SentinelMCP gateway application.

One ASGI app: ``/mcp`` -> the agent-facing MCP proxy (stateful sessions only); everything
else -> the FastAPI admin/probe API. Run with ``python -m gateway``.

Lifespan order (shutdown is the reverse, decision log flushed last):
recorder -> upstream tool servers -> MCP session manager.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.types import Receive, Scope, Send

from gateway.admin import AdminDeps, build_admin_router
from gateway.approvals import ApprovalStore
from gateway.config import Settings, get_settings
from gateway.decider import PipelineDecider
from gateway.enforcer import Enforcer
from gateway.logging import configure_logging
from gateway.policy import PolicyEngine
from gateway.proxy import GatewayProxy, build_mcp_app
from gateway.recorder import DecisionRecorder, create_db_engine
from gateway.session import SessionTracker, create_redis
from gateway.session_identity import McpSessionIdResolver
from gateway.upstream import UpstreamManager, load_servers

log = structlog.get_logger("sentinel.app")
MCP_PATH = "/mcp"


class GatewayApp:
    """Top-level ASGI router: MCP traffic to the proxy, the rest to FastAPI."""

    def __init__(self, api: FastAPI, mcp: Any) -> None:
        self.api = api
        self.mcp = mcp

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope["type"] == "http" and (path == MCP_PATH or path.startswith(MCP_PATH + "/")):
            await self.mcp(scope, receive, send)
        else:  # http + lifespan
            await self.api(scope, receive, send)


def create_app(
    settings: Settings | None = None,
    *,
    redis: Redis | None = None,
    engine: AsyncEngine | None = None,
    upstream_targets: dict[str, Any] | None = None,
) -> GatewayApp:
    """Wire the gateway. ``redis`` / ``engine`` / ``upstream_targets`` are injection points
    for tests; by default everything comes from settings and config/servers.yaml."""
    settings = settings or get_settings()
    redis = redis if redis is not None else create_redis(settings)
    engine = engine if engine is not None else create_db_engine(settings)
    policy = PolicyEngine.load(settings.policy_path)
    if upstream_targets is not None:
        upstreams = UpstreamManager(upstream_targets)
    else:
        upstreams = UpstreamManager.from_specs(load_servers(settings.servers_path))
    tracker = SessionTracker.from_settings(redis, settings)
    approvals = ApprovalStore(redis, ttl_s=settings.session_ttl_s)
    recorder = DecisionRecorder.from_settings(engine, settings)
    enforcer = Enforcer(
        tracker,
        approvals,
        shadow_mode=settings.shadow_mode,
        recorder=recorder,
        approval_timeout_s=settings.approval_timeout_s,
        poll_interval_s=settings.approval_poll_interval_s,
        progress_interval_s=settings.approval_progress_interval_s,
    )
    proxy = GatewayProxy(
        settings=settings,
        upstreams=upstreams,
        decider=PipelineDecider(tracker, policy),
        enforcer=enforcer,
        tracker=tracker,
        policy=policy,
        resolver=McpSessionIdResolver(),
        recorder=recorder,
    )
    mcp_app, mcp_inner = build_mcp_app(proxy, settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log.info("gateway_starting", shadow_mode=settings.shadow_mode, policy=policy.model_version,
                 servers=upstreams.servers)  # fmt: skip
        if settings.shadow_mode:
            log.warning("shadow_mode_enabled", detail="decisions are logged but NOT enforced")
        await recorder.start()
        try:
            await upstreams.start()
            try:
                async with mcp_inner.router.lifespan_context(mcp_inner):
                    log.info("gateway_ready", status=upstreams.status())
                    yield
            finally:
                await upstreams.stop()
        finally:
            await recorder.stop(timeout_s=settings.decision_log_flush_timeout_s)
            await redis.aclose()
            await engine.dispose()
            log.info("gateway_stopped")

    api = FastAPI(title="SentinelMCP gateway", version="0.1.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)  # fmt: skip
    api.state.admin = AdminDeps(
        settings=settings, redis=redis, engine=engine, tracker=tracker, approvals=approvals,
        recorder=recorder, upstreams=upstreams, policy=policy,
    )  # fmt: skip
    api.include_router(build_admin_router())
    return GatewayApp(api, mcp_app)


def main() -> None:
    import uvicorn

    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
        timeout_graceful_shutdown=int(settings.decision_log_flush_timeout_s) + 5,
    )
