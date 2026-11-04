"""Connections to upstream MCP tool servers (the gateway as MCP *client*).

Servers come from ``config/servers.yaml`` (pydantic-validated). Each is reached over
stdio (local subprocess), a Unix socket (``uds``, e.g. the network-less shell sandbox) or
streamable HTTP. One server failing to connect never prevents the others from serving.

Every ``tools/list`` response a session receives — including the ones the mcp v2 client
issues SILENTLY inside ``call_tool`` when a tool's schema is not cached — is passed to the
``on_tools_listed`` observer together with the agent session id taken from
``CURRENT_SESSION_ID``. That is how description changes (rug-pulls) are seen per session.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, Self

import anyio
import structlog
import yaml
from mcp import Client
from mcp.client.stdio import StdioServerParameters, get_default_environment
from mcp.types import CallToolResult, ListToolsResult, Tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gateway.transports import uds_transport

log = structlog.get_logger("sentinel.upstream")

# Agent session on whose behalf the current upstream request runs (set by the proxy).
CURRENT_SESSION_ID: ContextVar[str | None] = ContextVar("sentinel_session_id", default=None)

ToolsObserver = Callable[[str | None, str, list[Tool]], Awaitable[None]]


class UpstreamConfigError(ValueError):
    pass


class ServerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")  # namespace exposed to agents
    transport: Literal["stdio", "uds", "http"]
    enabled: bool = True
    # stdio
    command: str | None = None  # "{python}" = the gateway's own interpreter
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    # uds
    socket: str | None = None
    host: str | None = None  # Host header on the server's DNS-rebinding allowlist
    # http
    url: str | None = None
    call_timeout_s: float = Field(default=30.0, gt=0, le=600)

    @model_validator(mode="after")
    def _transport_fields(self) -> Self:
        need = {"stdio": ("command",), "uds": ("socket", "host"), "http": ("url",)}[self.transport]
        missing = [f for f in need if getattr(self, f) in (None, "")]
        if missing:
            raise ValueError(f"{self.transport} server '{self.name}' needs: {', '.join(missing)}")
        return self


class ServersFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    servers: list[ServerSpec]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        names = [s.name for s in self.servers]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate server names: {sorted(dupes)}")
        return self


def load_servers(path: Path) -> list[ServerSpec]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return [s for s in ServersFile.model_validate(raw).servers if s.enabled]
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise UpstreamConfigError(f"invalid servers file {path}: {exc}") from exc


def build_target(spec: ServerSpec) -> Any:
    """What ``mcp.Client`` should connect to for this spec."""
    if spec.transport == "stdio":
        assert spec.command is not None
        command = sys.executable if spec.command == "{python}" else spec.command
        return StdioServerParameters(
            command=command,
            args=spec.args,
            # Minimal safe environment + explicit per-server vars only: tool servers must never
            # inherit gateway secrets (admin key, database URL, ...).
            env={**get_default_environment(), **spec.env},
            cwd=spec.cwd,
        )
    if spec.transport == "uds":
        assert spec.socket is not None and spec.host is not None
        return uds_transport(spec.socket, host=spec.host, timeout_s=spec.call_timeout_s)
    return spec.url


class UpstreamUnavailableError(RuntimeError):
    pass


class _Upstream:
    def __init__(self, name: str, call_timeout_s: float) -> None:
        self.name = name
        self.call_timeout_s = call_timeout_s
        self.client: Client | None = None
        self.stack = contextlib.AsyncExitStack()
        self.error: str | None = None


class UpstreamManager:
    def __init__(
        self,
        targets: dict[str, Any],
        *,
        on_tools_listed: ToolsObserver | None = None,
        call_timeouts: dict[str, float] | None = None,
        default_call_timeout_s: float = 30.0,
    ) -> None:
        self._targets = targets
        self._observer = on_tools_listed
        timeouts = call_timeouts or {}
        self._up = {n: _Upstream(n, timeouts.get(n, default_call_timeout_s)) for n in targets}

    @classmethod
    def from_specs(cls, specs: list[ServerSpec], **kw: Any) -> UpstreamManager:
        return cls(
            {s.name: build_target(s) for s in specs},
            call_timeouts={s.name: s.call_timeout_s for s in specs},
            **kw,
        )

    def set_observer(self, observer: ToolsObserver | None) -> None:
        """Set the tools/list observer (the proxy registers itself after construction)."""
        self._observer = observer

    @property
    def servers(self) -> list[str]:
        return list(self._up)

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            n: {"connected": u.client is not None, "error": u.error} for n, u in self._up.items()
        }

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        for name, up in self._up.items():
            try:
                client = await up.stack.enter_async_context(Client(self._targets[name]))
            except Exception as exc:  # noqa: BLE001 - one bad upstream must not stop the rest
                up.error = f"{type(exc).__name__}: {exc}"[:300]
                log.error("upstream_connect_failed", server=name, error=up.error)
                await up.stack.aclose()
                up.stack = contextlib.AsyncExitStack()
                continue
            self._hook_list_tools(name, client)
            up.client, up.error = client, None
            log.info("upstream_connected", server=name)

    async def stop(self) -> None:
        # LIFO: each connection's anyio cancel scopes nest inside the ones opened before it.
        for name, up in reversed(self._up.items()):
            try:
                await up.stack.aclose()
            except Exception as exc:  # noqa: BLE001
                log.warning("upstream_close_failed", server=name, error=type(exc).__name__)
            up.client = None

    def _hook_list_tools(self, name: str, client: Client) -> None:
        session = client.session
        original = session.list_tools

        async def observed(*args: Any, **kwargs: Any) -> ListToolsResult:
            result = await original(*args, **kwargs)
            if self._observer is not None:
                await self._observer(CURRENT_SESSION_ID.get(), name, list(result.tools))
            return result

        session.list_tools = observed  # type: ignore[method-assign]

    # ------------------------------------------------------------------ requests

    def _client(self, server: str) -> tuple[_Upstream, Client]:
        up = self._up.get(server)
        if up is None:
            raise UpstreamUnavailableError(f"unknown server: {server}")
        if up.client is None:
            raise UpstreamUnavailableError(f"server {server} is not connected ({up.error})")
        return up, up.client

    async def list_tools(self, server: str, session_id: str | None) -> list[Tool]:
        up, client = self._client(server)
        token = CURRENT_SESSION_ID.set(session_id)
        try:
            with anyio.fail_after(up.call_timeout_s):
                return list((await client.list_tools()).tools)
        finally:
            CURRENT_SESSION_ID.reset(token)

    async def call_tool(
        self, server: str, tool: str, arguments: dict[str, Any], session_id: str | None
    ) -> CallToolResult:
        up, client = self._client(server)
        token = CURRENT_SESSION_ID.set(session_id)
        try:
            with anyio.fail_after(up.call_timeout_s):
                result = await client.call_tool(tool, arguments)
        finally:
            CURRENT_SESSION_ID.reset(token)
        if not isinstance(result, CallToolResult):  # input-required flows are not proxied
            raise UpstreamUnavailableError(f"{server}.{tool} returned {type(result).__name__}")
        return result
