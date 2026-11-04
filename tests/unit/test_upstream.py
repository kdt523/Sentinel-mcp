import asyncio
from pathlib import Path
from typing import Any

import pytest
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent, Tool

from gateway.upstream import (
    CURRENT_SESSION_ID,
    ServerSpec,
    UpstreamConfigError,
    UpstreamManager,
    UpstreamUnavailableError,
    build_target,
    load_servers,
)
from tools_servers.filesystem.server import build_server as build_fs
from tools_servers.poisoned_desc.server import FORECAST_CLEAN, FORECAST_POISONED
from tools_servers.poisoned_desc.server import build_server as build_weather
from tools_servers.sandbox import prepare_sandbox

REPO = Path(__file__).resolve().parents[2]


class Observer:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str, dict[str, str | None]]] = []

    async def __call__(self, session_id: str | None, server: str, tools: list[Tool]) -> None:
        self.calls.append((session_id, server, {t.name: t.description for t in tools}))


@pytest.fixture
def fs_root(tmp_path: Path) -> Path:
    return prepare_sandbox(tmp_path / "fs")


def slow_server() -> MCPServer:
    s = MCPServer(name="slow")

    @s.tool()
    async def wait(seconds: float) -> str:
        await asyncio.sleep(seconds)
        return "done"

    return s


async def test_routes_calls_and_lists_per_server(fs_root: Path, tmp_path: Path) -> None:
    obs = Observer()
    mgr = UpstreamManager(
        {"filesystem": build_fs(fs_root), "weather": build_weather("static", tmp_path / "l")},
        on_tools_listed=obs,
    )
    await mgr.start()
    try:
        assert mgr.status() == {
            "filesystem": {"connected": True, "error": None},
            "weather": {"connected": True, "error": None},
        }
        tools = await mgr.list_tools("filesystem", "sess-A")
        assert {t.name for t in tools} == {"read_file", "list_dir", "write_file"}
        res = await mgr.call_tool("filesystem", "read_file", {"path": "docs/readme.md"}, "sess-A")
        assert not res.is_error
        assert "Acme Notes" in "".join(c.text for c in res.content if isinstance(c, TextContent))
    finally:
        await mgr.stop()
    assert obs.calls[0][:2] == ("sess-A", "filesystem")


async def test_observer_sees_implicit_sdk_lists_with_session_id(tmp_path: Path) -> None:
    obs = Observer()
    mgr = UpstreamManager(
        {"weather": build_weather("rugpull", tmp_path / "l")}, on_tools_listed=obs
    )
    await mgr.start()
    try:
        # No explicit list first: the SDK lists silently inside call_tool.
        args = {"amount": 1, "from_code": "usd", "to_code": "inr"}
        await mgr.call_tool("weather", "convert_currency", args, "sess-B")
        await mgr.list_tools("weather", "sess-B")
    finally:
        await mgr.stop()
    assert [(sid, srv) for sid, srv, _ in obs.calls] == [("sess-B", "weather")] * 2
    assert obs.calls[0][2]["get_forecast"] == FORECAST_CLEAN  # the hidden list
    assert obs.calls[1][2]["get_forecast"] == FORECAST_POISONED  # rug-pull visible to observer
    assert CURRENT_SESSION_ID.get() is None  # context restored


async def test_one_bad_upstream_does_not_stop_others(fs_root: Path) -> None:
    bad = StdioServerParameters(command=str(REPO / "does-not-exist.exe"), args=[])
    mgr = UpstreamManager({"broken": bad, "filesystem": build_fs(fs_root)})
    await mgr.start()
    try:
        st = mgr.status()
        assert st["filesystem"]["connected"] and not st["broken"]["connected"]
        assert st["broken"]["error"]
        with pytest.raises(UpstreamUnavailableError, match="not connected"):
            await mgr.call_tool("broken", "x", {}, "s")
        with pytest.raises(UpstreamUnavailableError, match="unknown server"):
            await mgr.list_tools("nope", "s")
        assert await mgr.list_tools("filesystem", "s")
    finally:
        await mgr.stop()


async def test_call_timeout() -> None:
    mgr = UpstreamManager({"slow": slow_server()}, default_call_timeout_s=0.2)
    await mgr.start()
    try:
        with pytest.raises(TimeoutError):
            await mgr.call_tool("slow", "wait", {"seconds": 5}, "s")
        ok = await mgr.call_tool("slow", "wait", {"seconds": 0}, "s")  # still usable
        assert not ok.is_error
    finally:
        await mgr.stop()


async def test_real_stdio_server_from_spec(tmp_path: Path) -> None:
    spec = ServerSpec(name="filesystem", transport="stdio", command="{python}",
                      args=["-m", "tools_servers.filesystem"], cwd=str(REPO),
                      env={"SANDBOX_FS_ROOT": str(tmp_path / "fs")})  # fmt: skip
    mgr = UpstreamManager.from_specs([spec])
    await mgr.start()
    try:
        res = await mgr.call_tool("filesystem", "read_file", {"path": "docs/onboarding.md"}, "s")
        assert "Onboarding checklist" in "".join(
            c.text for c in res.content if isinstance(c, TextContent)
        )
    finally:
        await mgr.stop()


def test_stdio_env_never_inherits_gateway_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_ADMIN_API_KEY", "fake-admin-key")
    monkeypatch.setenv("SENTINEL_DATABASE_URL", "postgresql://u:p@h/db")
    spec = ServerSpec(name="x", transport="stdio", command="{python}", env={"A": "1"})
    target = build_target(spec)
    assert isinstance(target, StdioServerParameters) and target.env is not None
    assert target.env["A"] == "1"
    assert not any(k.startswith("SENTINEL_") for k in target.env)


# --- config ---------------------------------------------------------------------------------


def test_repo_servers_file_is_valid() -> None:
    names = [s.name for s in load_servers(REPO / "config" / "servers.yaml")]
    assert {"filesystem", "sql_db", "email_send", "http_fetch", "weather", "notes"} <= set(names)
    assert "shell" not in names  # disabled outside compose


@pytest.mark.parametrize(
    ("doc", "msg"),
    [
        ({"servers": [{"name": "a", "transport": "stdio"}]}, "needs: command"),
        ({"servers": [{"name": "a", "transport": "uds", "socket": "/s"}]}, "needs: host"),
        ({"servers": [{"name": "a", "transport": "http"}]}, "needs: url"),
        ({"servers": [{"name": "A-B", "transport": "http", "url": "u"}]}, "name"),
        ({"servers": [{"name": "a", "transport": "ftp"}]}, "transport"),
        ({"servers": [{"name": "a", "transport": "http", "url": "u", "x": 1}]}, "x"),
        ({"servers": [{"name": "a", "transport": "http", "url": "u"}] * 2}, "duplicate"),
    ],
)
def test_servers_file_validation(tmp_path: Path, doc: dict[str, Any], msg: str) -> None:
    import yaml

    f = tmp_path / "s.yaml"
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(UpstreamConfigError, match=msg):
        load_servers(f)
