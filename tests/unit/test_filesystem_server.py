"""Filesystem MCP server: in-process via mcp.Client, plus one real stdio round-trip."""

import os
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult, TextContent

from tools_servers.filesystem.server import MAX_WRITE_BYTES, build_server
from tools_servers.sandbox import prepare_sandbox

REPO_ROOT = Path(__file__).resolve().parents[2]


def _text(result: CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return prepare_sandbox(tmp_path / "fs")


async def test_lists_exactly_three_tools(root: Path) -> None:
    async with Client(build_server(root)) as client:
        tools = await client.list_tools()
    assert sorted(t.name for t in tools.tools) == ["list_dir", "read_file", "write_file"]
    assert all(t.description for t in tools.tools)


async def test_read_file(root: Path) -> None:
    async with Client(build_server(root)) as client:
        res = await client.call_tool("read_file", {"path": "docs/readme.md"})
    assert not res.is_error
    assert "Acme Notes" in _text(res)


async def test_read_secret_is_allowed_by_server(root: Path) -> None:
    # Confinement only: judging whether .env may be read is the gateway's job.
    async with Client(build_server(root)) as client:
        res = await client.call_tool("read_file", {"path": ".env"})
    assert not res.is_error
    assert "FAKE_API_KEY" in _text(res)


async def test_list_dir_root_shows_hidden_and_dirs_first(root: Path) -> None:
    async with Client(build_server(root)) as client:
        res = await client.call_tool("list_dir", {"path": "."})
    lines = _text(res).splitlines()
    assert ".ssh/" in lines and "docs/" in lines
    assert any(line.startswith(".env\t") for line in lines)
    first_file = next(i for i, line in enumerate(lines) if not line.endswith("/"))
    assert all(line.endswith("/") for line in lines[:first_file])


async def test_write_then_read(root: Path) -> None:
    async with Client(build_server(root)) as client:
        w = await client.call_tool("write_file", {"path": "notes/todo.md", "content": "hi ✓"})
        r = await client.call_tool("read_file", {"path": "notes/todo.md"})
    assert not w.is_error
    assert "notes/todo.md" in _text(w)
    assert _text(r) == "hi ✓"


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("read_file", {"path": "../../etc/passwd"}),
        ("read_file", {"path": "/etc/passwd"}),
        ("read_file", {"path": "docs"}),  # directory, not file
        ("read_file", {"path": "missing.txt"}),
        ("list_dir", {"path": "../"}),
        ("list_dir", {"path": "docs/readme.md"}),  # file, not dir
        ("write_file", {"path": "../escape.txt", "content": "x"}),
        ("write_file", {"path": ".", "content": "x"}),
        ("write_file", {"path": "big.txt", "content": "x" * (MAX_WRITE_BYTES + 1)}),
    ],
)
async def test_errors_are_tool_errors(root: Path, tool: str, args: dict[str, str]) -> None:
    async with Client(build_server(root)) as client:
        res = await client.call_tool(tool, args)
    assert res.is_error
    assert not (root.parent / "escape.txt").exists()


async def test_stdio_roundtrip(tmp_path: Path) -> None:
    """Spawn the real server process over stdio, as the gateway will."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tools_servers.filesystem"],
        env={**os.environ, "SANDBOX_FS_ROOT": str(tmp_path / "fs")},
        cwd=str(REPO_ROOT),
    )
    async with Client(params, read_timeout_seconds=30) as client:
        res = await client.call_tool("read_file", {"path": "docs/onboarding.md"})
    assert not res.is_error
    assert "Onboarding checklist" in _text(res)
