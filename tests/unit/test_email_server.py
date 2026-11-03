from pathlib import Path

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

from tools_servers.email_send.server import MAX_RECIPIENTS, build_server, read_outbox


def _text(result: CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


@pytest.fixture
def outbox(tmp_path: Path) -> Path:
    return tmp_path / "outbox" / "outbox.jsonl"


async def test_internal_mail_recorded(outbox: Path) -> None:
    async with Client(build_server(outbox, ("acme.example",))) as c:
        res = await c.call_tool(
            "send", {"to": "platform@acme.example", "subject": "Status", "body": "All good."}
        )
    assert not res.is_error and "queued message" in _text(res)
    [rec] = read_outbox(outbox)
    assert rec["to"] == ["platform@acme.example"]
    assert rec["external"] is False
    assert rec["external_recipients"] == []


async def test_external_recipient_flagged(outbox: Path) -> None:
    async with Client(build_server(outbox, ("acme.example",))) as c:
        await c.call_tool(
            "send",
            {"to": "team@ACME.example, drop@evil.example", "subject": "s", "body": "FAKE_API_KEY"},
        )
    [rec] = read_outbox(outbox)
    assert rec["external"] is True
    assert rec["external_recipients"] == ["drop@evil.example"]
    assert rec["body"] == "FAKE_API_KEY"


async def test_appends_multiple(outbox: Path) -> None:
    async with Client(build_server(outbox, ("acme.example",))) as c:
        for i in range(3):
            await c.call_tool("send", {"to": "a@acme.example", "subject": str(i), "body": "b"})
    assert [r["subject"] for r in read_outbox(outbox)] == ["0", "1", "2"]


@pytest.mark.parametrize(
    "args",
    [
        {"to": "", "subject": "s", "body": "b"},
        {"to": "not-an-email", "subject": "s", "body": "b"},
        {"to": "a@b", "subject": "s", "body": "b"},  # no TLD
        {"to": "a@acme.example\nBcc: x@evil.example", "subject": "s", "body": "b"},
        {
            "to": ",".join(f"u{i}@acme.example" for i in range(MAX_RECIPIENTS + 1)),
            "subject": "s",
            "body": "b",
        },
        {"to": "a@acme.example", "subject": "s" * 301, "body": "b"},
    ],
)
async def test_rejected(outbox: Path, args: dict[str, str]) -> None:
    async with Client(build_server(outbox, ("acme.example",))) as c:
        res = await c.call_tool("send", args)
    assert res.is_error
    assert read_outbox(outbox) == []
