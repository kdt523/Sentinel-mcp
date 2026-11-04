"""Which agent session does an MCP request belong to?

Session-level detection needs a stable, server-controlled identity per agent session.
``SessionResolver`` is the seam; the first implementation uses the stateful MCP
protocol's transport session id (``Mcp-Session-Id``, protocol 2025-11-25), assigned by the
server per connection. See docs/threat-model.md section 3.

The stateless MCP protocol (2026-07-28) has no session id, so the gateway declines it
(``StatefulOnlyMiddleware``): ``auto`` clients fall back to the stateful protocol, and
clients that only speak the stateless one are refused with an explicit reason (fail
closed). Phase 6 adds a gateway-issued token resolver (agent authentication).
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from mcp.server.context import ServerRequestContext
from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from starlette.types import ASGIApp, Message, Receive, Scope, Send

STATELESS_REFUSAL = (
    "SentinelMCP requires a stateful MCP session (protocol 2025-11-25 with Mcp-Session-Id). "
    "The stateless 2026-07-28 protocol is not supported: session-level security checks "
    "(taint tracking, multi-step exfiltration, tool-description changes) need a "
    "per-connection session id. Connect with a client that supports the 2025-11-25 "
    "protocol (mcp v2 clients in 'auto' mode fall back automatically)."
)
METHOD_NOT_FOUND = -32601
INVALID_REQUEST = -32600
_MAX_BODY = 4 * 1024 * 1024


class SessionResolutionError(Exception):
    """The request cannot be attributed to an agent session; refuse it."""


class SessionResolver(Protocol):
    def resolve(self, ctx: ServerRequestContext[Any, Any]) -> str:
        """Return the session id or raise ``SessionResolutionError``."""
        ...


class McpSessionIdResolver:
    """Session = the stateful MCP transport session (``Mcp-Session-Id``)."""

    def resolve(self, ctx: ServerRequestContext[Any, Any]) -> str:
        request = ctx.request
        headers = getattr(request, "headers", None)
        session_id = headers.get("mcp-session-id") if headers is not None else None
        if not session_id:
            raise SessionResolutionError(STATELESS_REFUSAL)
        return str(session_id)


def _rpc_error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


class StatefulOnlyMiddleware:
    """Decline the stateless MCP era before it reaches the MCP app.

    ``server/discover`` (the stateless handshake probe) and any request labelled with a
    stateless-era ``MCP-Protocol-Version`` get a JSON-RPC error inside an HTTP 200, so
    ``auto`` clients treat it as an RPC error and fall back to ``initialize()``.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        body, more = b"", True
        while more:
            message = await receive()
            body += message.get("body", b"")
            more = message.get("more_body", False)
            if len(body) > _MAX_BODY:
                await self._reply(send, _rpc_error(None, INVALID_REQUEST, "request too large"))
                return

        modern_header = headers.get("mcp-protocol-version") in MODERN_PROTOCOL_VERSIONS
        messages = self._messages(body)
        is_discover = any(m.get("method") == "server/discover" for m in messages)
        if is_discover or modern_header:
            # Answer every request in the body with ITS OWN id: a client waiting on id N
            # must receive the refusal for id N, or it hangs until its own timeout.
            errors = [
                _rpc_error(m["id"], METHOD_NOT_FOUND, STATELESS_REFUSAL)
                for m in messages
                if "id" in m and "method" in m
            ]
            if not errors:  # notifications only: nothing to answer, just refuse
                errors = [_rpc_error(None, METHOD_NOT_FOUND, STATELESS_REFUSAL)]
            await self._reply(send, errors[0] if len(errors) == 1 else errors)
            return

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    def _messages(body: bytes) -> list[dict[str, Any]]:
        try:
            data = json.loads(body)
        except ValueError:
            return []
        items = data if isinstance(data, list) else [data]
        return [m for m in items if isinstance(m, dict)]

    @staticmethod
    async def _reply(send: Send, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
        body = json.dumps(payload).encode()
        await send({
            "type": "http.response.start", "status": 200,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode())],
        })  # fmt: skip
        await send({"type": "http.response.body", "body": body})
