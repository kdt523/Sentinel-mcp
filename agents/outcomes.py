"""Classify what the gateway did to a tool call, from the MCP result alone.

Each enforcement outcome has a distinct message prefix (``gateway.enforcer.MSG_*``), so an
agent can tell outcomes apart without any side channel:

==================  =====================================================================
ALLOW               result forwarded unchanged (includes calls a human approved)
SANITIZE            result forwarded with injected spans removed
BLOCK               call or output blocked (incl. terminated session / revoked tool checks)
REVOKE_TOOL         tool revoked for the session by this call's decision
TERMINATE_SESSION   session terminated by this call's decision
REQUIRE_APPROVAL    approval was required and the call was denied / expired
ERROR               the tool itself failed (not a gateway decision)
==================  =====================================================================
"""

from __future__ import annotations

from mcp.types import CallToolResult, TextContent

from gateway.enforcer import MSG_BLOCKED, MSG_DENIED, MSG_REVOKED, MSG_TERMINATED
from gateway.sanitizer import MARK

OUTCOMES = (
    "ALLOW",
    "SANITIZE",
    "BLOCK",
    "REVOKE_TOOL",
    "TERMINATE_SESSION",
    "REQUIRE_APPROVAL",
    "ERROR",
)


def result_text(result: CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


def classify(result: CallToolResult) -> str:
    text = result_text(result)
    if result.is_error:
        for prefix, outcome in (
            (MSG_TERMINATED, "TERMINATE_SESSION"),
            (MSG_REVOKED, "REVOKE_TOOL"),
            (MSG_DENIED, "REQUIRE_APPROVAL"),
            (MSG_BLOCKED, "BLOCK"),
        ):
            if text.startswith(prefix):
                return outcome
        return "ERROR"
    return "SANITIZE" if MARK in text else "ALLOW"
