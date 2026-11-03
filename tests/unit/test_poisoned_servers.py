from pathlib import Path

import pytest
import yaml
from mcp import Client
from mcp.types import CallToolResult, TextContent

from tools_servers.poisoned_desc.server import (
    FORECAST_CLEAN,
    FORECAST_POISONED,
    build_server,
    read_log,
)
from tools_servers.poisoned_output.server import NOTES, TICKETS
from tools_servers.poisoned_output.server import build_server as build_output_server

MANIFEST = Path(__file__).resolve().parents[2] / "tools_servers" / "poisoned_manifest.yaml"


def _text(result: CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


async def _forecast_desc(client: Client) -> str | None:
    tools = (await client.list_tools()).tools
    return next(t.description for t in tools if t.name == "get_forecast")


# --- poisoned_desc ----------------------------------------------------------


async def test_static_mode_poisoned_from_first_list(tmp_path: Path) -> None:
    async with Client(build_server("static", tmp_path / "log.jsonl")) as c:
        assert await _forecast_desc(c) == FORECAST_POISONED


async def test_clean_control_tool_never_changes(tmp_path: Path) -> None:
    async with Client(build_server("rugpull", tmp_path / "log.jsonl")) as c:
        descs = [
            next(
                t.description for t in (await c.list_tools()).tools if t.name == "convert_currency"
            )
            for _ in range(3)
        ]
    assert len(set(descs)) == 1
    assert descs[0] is not None and "<IMPORTANT>" not in descs[0]


async def test_rugpull_clean_first_then_poisoned(tmp_path: Path) -> None:
    server = build_server("rugpull", tmp_path / "log.jsonl")
    async with Client(server) as c:
        assert server.list_calls == 0, "client must not list tools implicitly on connect"
        first = await _forecast_desc(c)
        second = await _forecast_desc(c)
        third = await _forecast_desc(c)
    assert first == FORECAST_CLEAN
    assert second == third == FORECAST_POISONED
    assert "<IMPORTANT>" not in FORECAST_CLEAN


async def test_sdk_lists_implicitly_when_calling_an_unlisted_tool(tmp_path: Path) -> None:
    """Pins mcp v2 behaviour the gateway relies on (step 12): calling a tool whose output
    schema is not cached makes the client SILENTLY run tools/list. The gateway must
    therefore scan descriptions on every list response its session receives."""
    server = build_server("rugpull", tmp_path / "log.jsonl")
    async with Client(server) as c:
        await c.call_tool("convert_currency", {"amount": 10, "from_code": "usd", "to_code": "inr"})
        assert server.list_calls == 1  # hidden listing consumed the clean description
        assert await _forecast_desc(c) == FORECAST_POISONED


async def test_no_implicit_listing_after_tools_were_listed(tmp_path: Path) -> None:
    server = build_server("rugpull", tmp_path / "log.jsonl")
    async with Client(server) as c:
        assert await _forecast_desc(c) == FORECAST_CLEAN
        for _ in range(3):
            await c.call_tool(
                "convert_currency", {"amount": 1, "from_code": "eur", "to_code": "inr"}
            )
            await c.call_tool("get_forecast", {"city": "Delhi"})
        assert server.list_calls == 1


async def test_sidecar_exfil_is_recorded(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    async with Client(build_server("static", log)) as c:
        ok = await c.call_tool("get_forecast", {"city": "Pune", "days": 2})
        stolen = await c.call_tool(
            "get_forecast", {"city": "Pune", "context": "FAKE_API_KEY=sk-fake-000"}
        )
    assert not ok.is_error and _text(ok).count("Day") == 2
    assert not stolen.is_error
    [rec] = read_log(log)  # only the call that carried data is recorded
    assert rec["context"] == "FAKE_API_KEY=sk-fake-000"


async def test_currency_tool_works(tmp_path: Path) -> None:
    async with Client(build_server("static", tmp_path / "l.jsonl")) as c:
        out = _text(
            await c.call_tool(
                "convert_currency", {"amount": 2, "from_code": "usd", "to_code": "inr"}
            )
        )
    assert out == "2.00 USD = 166.00 INR"


def test_main_rejects_bad_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools_servers.poisoned_desc import server as mod

    monkeypatch.setenv("POISON_MODE", "sometimes")
    with pytest.raises(SystemExit):
        mod.main()


# --- poisoned_output --------------------------------------------------------


async def test_search_notes_mixes_benign_and_poisoned() -> None:
    async with Client(build_output_server()) as c:
        everything = _text(await c.call_tool("search_notes", {"query": ""}))
        retro = _text(await c.call_tool("search_notes", {"query": "retro"}))
        none = _text(await c.call_tool("search_notes", {"query": "zzz"}))
    assert all(n["id"] in everything for n in NOTES)
    assert "mailmove-support.example" in everything
    assert "n-101" in retro and "mailmove" not in retro
    assert none == "No notes found."


async def test_ticket_markdown_comment_injection() -> None:
    async with Client(build_output_server()) as c:
        clean = _text(await c.call_tool("get_ticket", {"ticket_id": "t-2001"}))
        poisoned = _text(await c.call_tool("get_ticket", {"ticket_id": "T-2002"}))
        missing = await c.call_tool("get_ticket", {"ticket_id": "T-9"})
    assert "DELETE" not in clean
    assert "[//]: # (" in poisoned and "DELETE FROM customers" in poisoned
    assert missing.is_error


# --- manifest consistency ---------------------------------------------------


def test_manifest_matches_servers() -> None:
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["servers"]
    desc_tools = {t["tool"]: t for t in m["poisoned_desc"]["tools"]}
    assert desc_tools["get_forecast"]["label"] == "poisoned"
    assert set(desc_tools["get_forecast"]["modes"]) == {"static", "rugpull"}
    assert desc_tools["convert_currency"]["label"] == "benign"

    items = {i["item"]: i for i in m["poisoned_output"]["items"]}
    assert set(items) == {n["id"] for n in NOTES} | set(TICKETS)
    poisoned = {k for k, v in items.items() if v["label"] == "poisoned"}
    assert poisoned == {"n-102", "T-2002"}
    techniques = {items[k]["technique"] for k in poisoned}
    assert techniques == {"plain_text_note", "markdown_comment"}
