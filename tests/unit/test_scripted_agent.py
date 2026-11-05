"""Scripted agent: bundled scripts must pass end to end; mismatches must be reported."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.outcomes import classify
from agents.scripted_agent import ScriptError, load_script, run_script, validate_script
from tests.unit.harness import REPO, agent_client, gateway_app

SCRIPTS = sorted((REPO / "agents" / "scripts").glob("*.json"))


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.stem for p in SCRIPTS])
async def test_bundled_scripts_pass(tmp_path: Path, path: Path) -> None:
    async with gateway_app(tmp_path) as app, agent_client(app) as client:
        report = await run_script(client, load_script(path))
    assert report.ok, "\n".join(report.mismatches)


async def test_mismatches_are_reported(tmp_path: Path) -> None:
    script = {
        "name": "wrong expectations",
        "steps": [
            {"tool": "filesystem.read_file", "args": {"path": ".ssh/id_rsa"},
             "expect": {"action": "ALLOW", "error": False}},
            {"tool": "filesystem.read_file", "args": {"path": "docs/readme.md"},
             "expect": {"contains": "no such text"}},
            {"op": "list_tools", "expect": {"excludes": ["filesystem.read_file"]}},
        ],
    }  # fmt: skip
    async with gateway_app(tmp_path) as app, agent_client(app) as client:
        report = await run_script(client, script)
    assert not report.ok
    m = report.mismatches
    assert any("expected ALLOW, got BLOCK" in x for x in m)
    assert any("expected error=False, got True" in x for x in m)
    assert any("does not contain 'no such text'" in x for x in m)
    assert any("should not include filesystem.read_file" in x for x in m)
    assert [s.ok for s in report.steps] == [False, False, False]


async def test_custom_separator(tmp_path: Path) -> None:
    script = {"steps": [{"tool": "filesystem.read_file", "args": {"path": "docs/readme.md"},
                         "expect": {"action": "ALLOW"}}]}  # fmt: skip
    async with gateway_app(tmp_path, tool_name_separator="__") as app, agent_client(app) as c:
        report = await run_script(c, script, separator="__")
    assert report.ok


@pytest.mark.parametrize(
    ("script", "msg"),
    [
        ({"steps": []}, "non-empty"),
        ({"steps": [{"op": "dance"}]}, "unknown op"),
        ({"steps": [{"tool": "noserver"}]}, "<server>.<tool>"),
        ({"steps": [{"tool": "a.b", "expect": {"actoin": "ALLOW"}}]}, "unknown expect keys"),
        ({"steps": [{"tool": "a.b", "expect": {"action": "MAYBE"}}]}, "expect.action"),
        ({"steps": [{"tool": "a.b", "args": {"x": "{{later}}"}}, {"id": "later", "tool": "a.c"}]},
         "refers to no earlier step"),
    ],
)  # fmt: skip
def test_script_validation(script: dict[str, Any], msg: str) -> None:
    with pytest.raises(ScriptError, match=msg):
        validate_script(script)


def test_load_script_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ScriptError, match="cannot read"):
        load_script(bad)


def test_bundled_scripts_are_valid_json_with_expectations() -> None:
    assert len(SCRIPTS) >= 5
    for path in SCRIPTS:
        script = json.loads(path.read_text(encoding="utf-8"))
        assert all("expect" in s for s in script["steps"]), path.name


def test_classify_prefixes() -> None:
    from mcp.types import CallToolResult, TextContent

    from gateway.enforcer import MSG_BLOCKED, MSG_DENIED, MSG_REVOKED, MSG_TERMINATED
    from gateway.sanitizer import MARK

    def r(text: str, err: bool) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text=text)], is_error=err)

    assert classify(r(MSG_BLOCKED + "x", True)) == "BLOCK"
    assert classify(r(MSG_REVOKED + "x", True)) == "REVOKE_TOOL"
    assert classify(r(MSG_TERMINATED + "x", True)) == "TERMINATE_SESSION"
    assert classify(r(MSG_DENIED + "x", True)) == "REQUIRE_APPROVAL"
    assert classify(r("file not found", True)) == "ERROR"
    assert classify(r(f"a {MARK} b", False)) == "SANITIZE"
    assert classify(r("hello", False)) == "ALLOW"
