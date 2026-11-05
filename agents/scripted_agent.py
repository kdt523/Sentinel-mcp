"""Deterministic scripted agent: replays a JSON script of MCP operations through the gateway.

Scripts double as tests: any step may carry an ``expect`` block and the runner reports
every mismatch (exit code 1 if there are any).

Script format::

    {
      "name": "multi-step exfiltration via a poisoned tool",
      "steps": [
        {"id": "env", "tool": "filesystem.read_file", "args": {"path": ".env"},
         "expect": {"action": "ALLOW"}},
        {"tool": "weather.get_forecast", "args": {"city": "Pune", "context": "{{env}}"},
         "expect": {"action": "TERMINATE_SESSION", "error": true}},
        {"op": "list_tools", "expect": {"excludes": ["weather.get_forecast"]}}
      ]
    }

- ``tool`` names use the gateway's internal form ``<server>.<tool>``; the runner converts
  them to the exposed form with ``--separator``.
- ``{{<id>}}`` inside any string argument is replaced by the text result of step ``<id>``.
- ``expect`` keys (all optional): ``action`` (see ``agents.outcomes``), ``error`` (bool),
  ``contains`` / ``not_contains`` (substring of the result text); for ``list_tools``:
  ``includes`` / ``excludes`` (internal tool names).

CLI::

    python -m agents.scripted_agent agents/scripts/benign_handbook.json [--url URL] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client

from agents.outcomes import OUTCOMES, classify, result_text

DEFAULT_URL = "http://127.0.0.1:8000/mcp"
_REF = re.compile(r"\{\{([A-Za-z0-9_-]+)\}\}")
_EXPECT_KEYS = {"action", "error", "contains", "not_contains", "includes", "excludes"}


class ScriptError(ValueError):
    """The script itself is malformed (not a gateway mismatch)."""


@dataclass
class StepResult:
    index: int
    op: str
    tool: str | None
    outcome: str | None
    is_error: bool | None
    ok: bool
    mismatches: list[str] = field(default_factory=list)
    preview: str = ""


@dataclass
class ScriptReport:
    name: str
    steps: list[StepResult]

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def mismatches(self) -> list[str]:
        return [f"step {s.index}: {m}" for s in self.steps for m in s.mismatches]


def load_script(path: Path) -> dict[str, Any]:
    try:
        script: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScriptError(f"cannot read script {path}: {exc}") from exc
    validate_script(script)
    return script


def validate_script(script: dict[str, Any]) -> None:
    steps = script.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ScriptError("script needs a non-empty 'steps' list")
    seen: set[str] = set()
    for i, step in enumerate(steps, 1):
        op = step.get("op", "call")
        if op not in ("call", "list_tools"):
            raise ScriptError(f"step {i}: unknown op {op!r}")
        if op == "call" and "." not in str(step.get("tool", "")):
            raise ScriptError(f"step {i}: 'tool' must be '<server>.<tool>'")
        expect = step.get("expect", {})
        unknown = set(expect) - _EXPECT_KEYS
        if unknown:
            raise ScriptError(f"step {i}: unknown expect keys {sorted(unknown)}")
        if "action" in expect and expect["action"] not in OUTCOMES:
            raise ScriptError(f"step {i}: expect.action must be one of {OUTCOMES}")
        for ref in _REF.findall(json.dumps(step.get("args", {}))):
            if ref not in seen:
                raise ScriptError(f"step {i}: '{{{{{ref}}}}}' refers to no earlier step id")
        if "id" in step:
            seen.add(str(step["id"]))


def _substitute(value: Any, results: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _REF.sub(lambda m: results[m.group(1)], value)
    if isinstance(value, dict):
        return {k: _substitute(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, results) for v in value]
    return value


def exposed(name: str, separator: str) -> str:
    server, _, tool = name.partition(".")
    return f"{server}{separator}{tool}"


async def run_script(
    client: Client, script: dict[str, Any], *, separator: str = "."
) -> ScriptReport:
    validate_script(script)
    results: dict[str, str] = {}
    report = ScriptReport(name=str(script.get("name", "script")), steps=[])
    for i, step in enumerate(script["steps"], 1):
        expect: dict[str, Any] = step.get("expect", {})
        op = step.get("op", "call")
        problems: list[str] = []
        if op == "list_tools":
            names = {t.name for t in (await client.list_tools()).tools}
            for t in expect.get("includes", []):
                if exposed(t, separator) not in names:
                    problems.append(f"list_tools should include {t}")
            for t in expect.get("excludes", []):
                if exposed(t, separator) in names:
                    problems.append(f"list_tools should not include {t}")
            report.steps.append(StepResult(i, op, None, None, None, not problems, problems,
                                           f"{len(names)} tools"))  # fmt: skip
            continue

        tool = str(step["tool"])
        args = _substitute(step.get("args", {}), results)
        result = await client.call_tool(exposed(tool, separator), args)
        text, outcome = result_text(result), classify(result)
        if "id" in step:
            results[str(step["id"])] = text
        if "action" in expect and outcome != expect["action"]:
            problems.append(f"{tool}: expected {expect['action']}, got {outcome} ({text[:120]!r})")
        if "error" in expect and bool(result.is_error) != bool(expect["error"]):
            problems.append(f"{tool}: expected error={expect['error']}, got {result.is_error}")
        if "contains" in expect and expect["contains"] not in text:
            problems.append(f"{tool}: result does not contain {expect['contains']!r}")
        if "not_contains" in expect and expect["not_contains"] in text:
            problems.append(f"{tool}: result unexpectedly contains {expect['not_contains']!r}")
        report.steps.append(StepResult(i, op, tool, outcome, bool(result.is_error), not problems,
                                       problems, " ".join(text.split())[:100]))  # fmt: skip
    return report


def _print_report(report: ScriptReport) -> None:
    print(f"== {report.name}")
    for s in report.steps:
        mark = "ok " if s.ok else "XX "
        label = s.tool or s.op
        print(f"{mark}{s.index:>2} {label:<28} {s.outcome or '-':<18} {s.preview}")
        for m in s.mismatches:
            print(f"      ! {m}")
    print(f"{'PASS' if report.ok else 'FAIL'}: {len(report.mismatches)} mismatch(es)")


async def _main_async(args: argparse.Namespace) -> int:
    reports = []
    for path in args.scripts:
        script = load_script(path)
        async with Client(args.url) as client:  # one connection = one session per script
            reports.append(await run_script(client, script, separator=args.separator))
    for r in reports:
        if args.json:
            print(json.dumps({"name": r.name, "ok": r.ok, "steps": [asdict(s) for s in r.steps]}))
        else:
            _print_report(r)
    return 0 if all(r.ok for r in reports) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agents.scripted_agent")
    parser.add_argument("scripts", nargs="+", type=Path)
    parser.add_argument("--url", default=os.environ.get("AGENT_GATEWAY_URL", DEFAULT_URL))
    parser.add_argument("--separator", default=os.environ.get("AGENT_TOOL_SEPARATOR", "."))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_main_async(args))
    except ScriptError as exc:
        print(f"script error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - connection problems etc.
        print(f"could not run scripts: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
