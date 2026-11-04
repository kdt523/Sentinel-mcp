"""The MCP proxy: the agent-facing MCP server that inspects everything in both directions.

``tools/list``
    Aggregates upstream tools as ``<server><sep><tool>``. Every description an upstream
    returns (explicit or SDK-silent lists) is checked per session: its sha256 is compared
    with the last one seen in this session (rug-pull detection), it becomes a
    ``tool_description`` event, and it is decided and enforced. Hidden/blocked/revoked
    tools are omitted; sanitized descriptions are rewritten.

``tools/call``
    request event -> decide (fail mode) -> enforce (approvals + progress) -> upstream ->
    output event -> decide -> enforce (sanitize) -> agent. Every event is recorded in the
    session tracker and the decision log.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any

import structlog
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)
from starlette.types import ASGIApp

from gateway.config import FailMode, Settings
from gateway.decider import PipelineDecider
from gateway.detectors import iter_strings
from gateway.enforcer import EnforcementResult, Enforcer, decide_with_fail_mode
from gateway.models import Action, Decision, Direction, Event, EventKind, sha256_text
from gateway.policy import PolicyEngine
from gateway.recorder import DecisionRecorder
from gateway.sanitizer import sanitize
from gateway.session import SessionTracker
from gateway.session_identity import (
    SessionResolutionError,
    SessionResolver,
    StatefulOnlyMiddleware,
)
from gateway.upstream import UpstreamManager, UpstreamUnavailableError

log = structlog.get_logger("sentinel.proxy")

_HIDDEN = object()
# Effective descriptions for the explicit tools/list currently being served (per task).
_LIST_VIEW: ContextVar[dict[str, Any] | None] = ContextVar("sentinel_list_view", default=None)


def flatten_content(result: CallToolResult) -> str:
    """Text the agent will see, for inspection. Non-text blocks become short placeholders."""
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            kind = getattr(block, "type", type(block).__name__)
            mime = getattr(block, "mime_type", None) or getattr(block, "mimeType", None)
            parts.append(f"[{kind}{' ' + mime if mime else ''} content]")
    text = "\n".join(parts)
    # Structured content reaches the agent too: inspect any string it carries that is not
    # already part of the text blocks, so an injection cannot hide there.
    extra = [s for s in iter_strings(result.structured_content or {}) if s and s not in text]
    return "\n".join([text, *extra]) if extra else text


def sanitize_result(result: CallToolResult, sanitized_text: str) -> CallToolResult:
    """Replace text content with the sanitized text and sanitize every string inside the
    structured content, keeping its shape (clients validate it against the output schema)."""

    def clean(value: Any) -> Any:
        if isinstance(value, str):
            return sanitize(value).text
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    structured = result.structured_content
    return CallToolResult(
        content=[TextContent(type="text", text=sanitized_text)],
        structured_content=clean(structured) if structured is not None else None,
        is_error=result.is_error,
    )


def _error(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


class GatewayProxy:
    def __init__(
        self,
        *,
        settings: Settings,
        upstreams: UpstreamManager,
        decider: PipelineDecider,
        enforcer: Enforcer,
        tracker: SessionTracker,
        policy: PolicyEngine,
        resolver: SessionResolver,
        recorder: DecisionRecorder | None = None,
    ) -> None:
        self.settings = settings
        self.upstreams = upstreams
        self.decider = decider
        self.enforcer = enforcer
        self.tracker = tracker
        self.policy = policy
        self.resolver = resolver
        self.recorder = recorder
        self.sep = settings.tool_name_separator
        upstreams.set_observer(self.on_tools_listed)

    # ------------------------------------------------------------------ naming

    def exposed_name(self, server: str, tool: str) -> str:
        return f"{server}{self.sep}{tool}"

    def parse_name(self, name: str) -> tuple[str, str] | None:
        for server in self.upstreams.servers:
            prefix = f"{server}{self.sep}"
            if name.startswith(prefix) and len(name) > len(prefix):
                return server, name[len(prefix) :]
        return None

    def _fail_mode(self, qualified_tool: str) -> FailMode:
        return self.policy.fail_mode_for(qualified_tool) or self.settings.default_fail_mode

    # ------------------------------------------------------------------ pipeline helpers

    async def _decide(self, event: Event) -> Decision:
        return await decide_with_fail_mode(
            self.decider,
            event,
            fail_mode=self._fail_mode(event.qualified_tool),
            timeout_s=self.settings.decision_timeout_ms / 1000,
        )

    async def _record(
        self,
        event: Event,
        decision: Decision,
        result: EnforcementResult,
        *,
        request: Event | None = None,
        raw_output: str | None = None,
    ) -> None:
        try:
            await self.tracker.record(event, decision, request=request)
        except Exception as exc:  # noqa: BLE001 - tracking failure must not crash the call
            log.error("session_record_failed", session_id=event.session_id,
                      error=f"{type(exc).__name__}: {exc}"[:300])  # fmt: skip
        if self.recorder is not None:
            self.recorder.submit(event, decision, enforced=result.enforced, raw_output=raw_output)

    # ------------------------------------------------------------------ tools/list

    async def on_tools_listed(self, session_id: str | None, server: str, tools: list[Tool]) -> None:
        """Observer for EVERY upstream tools/list response (explicit or SDK-silent)."""
        view = _LIST_VIEW.get()
        if session_id is None:
            log.warning("tools_listed_without_session", server=server)
            return
        for tool in tools:
            qualified = f"{server}.{tool.name}"
            description = tool.description or ""
            obs = await self.tracker.observe_description(
                session_id, qualified, sha256_text(description)
            )
            event = Event.from_output(
                session_id=session_id,
                server=server,
                tool=tool.name,
                full_text=description,
                max_chars=self.settings.output_max_chars,
                kind=EventKind.TOOL_DESCRIPTION,
                description_changed=obs.changed,
                previous_description_sha256=obs.previous_sha256 if obs.changed else None,
            )
            decision = await self._decide(event)
            if decision.action is Action.REQUIRE_APPROVAL and not self.enforcer.shadow_mode:
                # Nobody can approve a tools/list in-band: hide the tool instead of waiting.
                decision = decision.model_copy(update={"action": Action.BLOCK})
            result = await self.enforcer.enforce(event, decision, full_output=description)
            if obs.first_seen or obs.changed or decision.action is not Action.ALLOW:
                await self._record(event, decision, result)
            if obs.changed:
                log.warning("tool_description_changed", session_id=session_id, tool=qualified,
                            action=decision.action.value)  # fmt: skip
            if view is not None:
                view[qualified] = result.output if result.forward else _HIDDEN

    async def list_tools(
        self, ctx: ServerRequestContext[Any, Any], params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        try:
            session_id = self.resolver.resolve(ctx)
        except SessionResolutionError as exc:
            log.warning("session_unresolved", method="tools/list")
            raise ValueError(str(exc)) from exc
        exposed: list[Tool] = []
        for server in self.upstreams.servers:
            view: dict[str, Any] = {}
            token = _LIST_VIEW.set(view)
            try:
                tools = await self.upstreams.list_tools(server, session_id)
            except (UpstreamUnavailableError, TimeoutError) as exc:
                log.warning("upstream_list_failed", server=server, error=str(exc)[:200])
                continue
            finally:
                _LIST_VIEW.reset(token)
            for tool in tools:
                effective = view.get(f"{server}.{tool.name}", _HIDDEN)
                if effective is _HIDDEN:
                    continue
                exposed.append(tool.model_copy(update={
                    "name": self.exposed_name(server, tool.name), "description": effective,
                }))  # fmt: skip
        return ListToolsResult(tools=exposed)

    # ------------------------------------------------------------------ tools/call

    async def call_tool(
        self, ctx: ServerRequestContext[Any, Any], params: CallToolRequestParams
    ) -> CallToolResult:
        try:
            session_id = self.resolver.resolve(ctx)
        except SessionResolutionError as exc:
            log.warning("session_unresolved", method="tools/call")
            return _error(str(exc))
        parsed = self.parse_name(params.name)
        if parsed is None:
            return _error(f"Unknown tool: {params.name}")
        server, tool = parsed
        args = dict(params.arguments or {})

        async def progress(done: float, total: float | None, message: str) -> None:
            await ctx.session.report_progress(done, total, message)

        # 1. request
        request = Event(
            session_id=session_id, direction=Direction.REQUEST, server=server, tool=tool, args=args
        )
        decision = await self._decide(request)
        verdict = await self.enforcer.enforce(request, decision, progress=progress)
        await self._record(request, decision, verdict)
        if not verdict.forward:
            return _error(verdict.message or "Blocked by SentinelMCP.")

        # 2. upstream
        t0 = time.perf_counter()
        try:
            upstream = await self.upstreams.call_tool(server, tool, args, session_id)
        except (UpstreamUnavailableError, TimeoutError) as exc:
            log.error("upstream_call_failed", server=server, tool=tool, error=str(exc)[:200])
            return _error(f"Upstream tool {server}.{tool} is unavailable ({type(exc).__name__}).")
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the agent
            log.error("upstream_call_error", server=server, tool=tool,
                      error=f"{type(exc).__name__}: {exc}"[:300])  # fmt: skip
            return _error(f"Upstream tool {server}.{tool} failed ({type(exc).__name__}).")
        upstream_ms = (time.perf_counter() - t0) * 1000

        # 3. output
        full_text = flatten_content(upstream)
        output = Event.from_output(
            session_id=session_id, server=server, tool=tool, full_text=full_text,
            max_chars=self.settings.output_max_chars, upstream_ms=upstream_ms,
        )  # fmt: skip
        out_decision = await self._decide(output)
        out_verdict = await self.enforcer.enforce(
            output, out_decision, progress=progress, full_output=full_text
        )
        await self._record(output, out_decision, out_verdict, request=request,
                           raw_output=full_text)  # fmt: skip
        if not out_verdict.forward:
            return _error(out_verdict.message or "Output blocked by SentinelMCP.")
        if out_verdict.sanitized:
            return sanitize_result(upstream, out_verdict.output or "")
        return upstream


def build_mcp_server(proxy: GatewayProxy) -> Server[Any]:
    return Server(
        "sentinel-mcp",
        version="0.1.0",
        instructions="Tools are provided through the SentinelMCP security gateway.",
        on_list_tools=proxy.list_tools,
        on_call_tool=proxy.call_tool,
    )


def build_mcp_app(proxy: GatewayProxy, settings: Settings) -> tuple[ASGIApp, Any]:
    """Agent-facing ASGI app (stateful sessions only) and the inner Starlette app, whose
    lifespan must run (it owns the MCP session manager)."""
    inner = build_mcp_server(proxy).streamable_http_app(
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=settings.mcp_allowed_hosts,
            allowed_origins=[],
        ),
    )
    return StatefulOnlyMiddleware(inner), inner
