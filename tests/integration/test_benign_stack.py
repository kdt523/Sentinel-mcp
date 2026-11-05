"""Replay every benign session that uses the SHELL on the real compose stack (the shell only
runs in its hardened container; all other sessions are replayed in the unit tests).

The gateway runs in shadow mode (decisions logged, nothing enforced) and the sandbox is reset
before each session, exactly as dataset generation does.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from mcp import Client

from agents.outcomes import classify, result_text
from data.benign.templates import expand_all
from tests.integration.test_stack import MCP_URL, compose, restart_gateway

pytestmark = pytest.mark.integration

SHELL_SESSIONS = [s for s in expand_all(seed=42)
                  if any(st.tool.startswith("shell.") for st in s.steps)]  # fmt: skip
RESET = (
    "from pathlib import Path; "
    "from tools_servers.sandbox import prepare_sandbox; "
    "from tools_servers.sql_db.server import prepare_database; "
    "prepare_sandbox(Path('/app/sandbox_runtime/fs'), reset=True); "
    "prepare_database(Path('/app/sandbox_runtime/db/app.db'), reset=True)"
)


@pytest.fixture(scope="module", autouse=True)
def shadow_gateway() -> Iterator[None]:
    restart_gateway(GATEWAY_SHADOW_MODE="true")
    yield
    restart_gateway()  # back to enforcing


async def test_every_shell_benign_session_runs_on_the_stack() -> None:
    assert len(SHELL_SESSIONS) >= 30
    failures: list[str] = []
    for s in SHELL_SESSIONS:
        compose("exec", "-T", "gateway", "python", "-c", RESET)
        async with Client(MCP_URL) as client:  # one connection = one session
            for st in s.steps:
                res = await client.call_tool(st.tool, st.args)
                if classify(res) == "ERROR":
                    failures.append(
                        f"{s.session_key} {st.tool} {st.args}: {result_text(res)[:150]}"
                    )
    assert not failures, "\n".join(failures[:20])
