"""Measure how reliably the agent's LLM makes the RIGHT tool call as more tools are offered.

For each tool set, the same single-turn request is sent ``--tries`` times (temperature from
AGENT_TEMPERATURE, default 0) and we count responses that call the expected tool. Tool
schemas come from the running gateway (`make dev`), exactly as the agent sees them.

    python -m agents.measure_tool_calling --tries 20 \
        --set "filesystem" --set "filesystem,http_fetch,email_send" \
        --set "filesystem,sql_db,email_send,http_fetch,shell" --set "all"

Prints a Markdown table (tools offered, correct calls / tries).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from mcp import Client

from agents.langgraph_agent import (
    SYSTEM_PROMPT,
    AgentRun,
    AgentSettings,
    chat_model,
    filter_tools,
    llm_tool_name,
    mcp_tools_for_langchain,
)

DEFAULT_TASK = "Read the file docs/readme.md and tell me at what time pull requests are reviewed."
DEFAULT_EXPECT = "filesystem.read_file"


async def measure(settings: AgentSettings, sets: list[str], tries: int, task: str,
                  expect: str) -> list[tuple[str, int, int, int, float]]:  # fmt: skip
    rows = []
    expected = llm_tool_name(expect)
    async with Client(settings.gateway_url) as client, chat_model(settings) as llm:
        all_tools = list((await client.list_tools()).tools)
        for spec in sets:
            servers = [] if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
            tools = filter_tools(all_tools, servers, settings.gateway_tool_separator)
            bound = llm.bind_tools(mcp_tools_for_langchain(client, tools, AgentRun("m", "")))
            hits, t0 = 0, time.monotonic()
            for _ in range(tries):
                msg = await bound.ainvoke([("system", SYSTEM_PROMPT), ("user", task)])
                hits += any(tc["name"] == expected for tc in msg.tool_calls)
            rows.append((spec, len(tools), hits, tries, (time.monotonic() - t0) / tries))
            print(f"  {spec:<48} tools={len(tools):2d}  {hits}/{tries}", file=sys.stderr)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agents.measure_tool_calling")
    p.add_argument("--set", dest="sets", action="append", required=True,
                   help="comma-separated gateway servers, or 'all'")  # fmt: skip
    p.add_argument("--tries", type=int, default=20)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--expect", default=DEFAULT_EXPECT, help="expected gateway tool name")
    args = p.parse_args(argv)
    settings = AgentSettings()
    rows = asyncio.run(measure(settings, args.sets, args.tries, args.task, args.expect))
    print(f"\nModel `{settings.llm_model}`, temperature {settings.temperature}, task: "
          f"\"{args.task}\" (expected tool `{args.expect}`)\n")  # fmt: skip
    print("| Servers offered | Tools | Correct tool call | Rate | Avg s/try |")
    print("|---|---|---|---|---|")
    for spec, n_tools, hits, tries, secs in rows:
        print(f"| {spec} | {n_tools} | {hits}/{tries} | {hits / tries:.0%} | {secs:.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
