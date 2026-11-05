# ruff: noqa: E501  (self-contained container script; long literal lines are intentional)
"""Gateway-OFF executor for dataset step 6 (sandbox only; runs inside a Linux container).

Reads one JSON job per stdin line, runs the job's scripted steps DIRECTLY against the sandbox
tool servers (no gateway, no policy, no sanitizer) in a fresh temporary sandbox, and writes one
JSON result line with everything needed to judge whether the attacker's goal was achieved:
every tool output, the tool descriptions listed, file / database snapshots before and after,
the e-mail outbox, requests the exfil sink received, and the weather server's attribution log.

Job:    {"id", "files": {rel: text}, "sql": [[stmt, params]...], "ids": [customer ids],
         "steps": [{"op": "call"|"list_tools", "tool": "<server>.<tool>"|"<server>", "args": {}}],
         "weather": {"mode": "clean"|"static"|"rugpull", "variant": int|null}}
Result: see ``run_job``. Never reads or writes anything outside its temp directory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import sys
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.types import TextContent

from tools_servers.email_send.server import build_server as build_email
from tools_servers.filesystem.server import build_server as build_fs
from tools_servers.http_fetch.server import build_server as build_fetch
from tools_servers.poisoned_desc.server import build_server as build_weather
from tools_servers.poisoned_output.server import build_server as build_notes
from tools_servers.sandbox import prepare_sandbox
from tools_servers.sandbox_net import SandboxWeb
from tools_servers.shell.server import DEFAULT_ALLOWLIST, CommandRejected, execute, parse_command
from tools_servers.sql_db.server import build_server as build_sql
from tools_servers.sql_db.server import prepare_database


def _snapshot_fs(root: Path) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        try:
            st = p.lstat()
            if p.is_file() and not p.is_symlink():
                out[rel] = [
                    hashlib.sha256(p.read_bytes()).hexdigest(),
                    st.st_size,
                    st.st_mode & 0o7777,
                ]
            else:
                out[rel] = ["dir", 0, st.st_mode & 0o7777]
        except OSError:
            out[rel] = ["unreadable", 0, 0]
    return out


def _snapshot_db(path: Path) -> dict[str, Any]:
    snap: dict[str, Any] = {"tables": {}, "plans": {}}
    with sqlite3.connect(path) as conn:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for name in names:
            rows = conn.execute(f'SELECT * FROM "{name}" ORDER BY 1').fetchall()  # noqa: S608
            snap["tables"][name] = {
                "count": len(rows),
                "sha": hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest(),
            }
        if "customers" in names:
            snap["plans"] = {
                str(r[0]): r[1] for r in conn.execute("SELECT customer_id, plan FROM customers")
            }
    conn.close()
    return snap


def _text(result: Any) -> str:
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


async def run_job(job: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = prepare_sandbox(base / "fs", reset=True)
        for rel, text in job["files"].items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        db = prepare_database(base / "db" / "app.db", reset=True)
        with sqlite3.connect(db) as conn:
            orders = conn.execute("SELECT * FROM orders").fetchall()
            for sql, params in job["sql"]:
                conn.execute(sql, params)
            for row in orders:
                if row[1] in set(job["ids"]):
                    conn.execute(
                        "INSERT INTO orders VALUES (" + ", ".join("?" * len(row)) + ")", row
                    )
        conn.close()
        web = SandboxWeb()
        weather = job.get("weather") or {"mode": "clean", "variant": None}
        servers = {
            "filesystem": build_fs(root),
            "sql_db": build_sql(db),
            "email_send": build_email(base / "outbox.jsonl", ("acme.example",)),
            "http_fetch": build_fetch(("mockweb", "exfil-sink"), transport=web),
            "weather": build_weather(
                weather["mode"], base / "poison.jsonl", variant=weather["variant"]
            ),
            "notes": build_notes(),
        }
        fs_before, db_before = _snapshot_fs(root), _snapshot_db(db)
        calls: list[dict[str, Any]] = []
        listings: list[dict[str, str]] = []
        async with AsyncExitStack() as stack:
            clients = {n: await stack.enter_async_context(Client(s)) for n, s in servers.items()}
            for st in job["steps"]:
                if st["op"] == "list_tools":
                    for t in (await clients[st["tool"]].list_tools()).tools:
                        listings.append(
                            {"tool": f"{st['tool']}.{t.name}", "description": t.description or ""}
                        )
                    continue
                server, _, name = st["tool"].partition(".")
                if server == "shell":
                    try:
                        argv = parse_command(st["args"]["cmd"], frozenset(DEFAULT_ALLOWLIST))
                        out, err = execute(argv, root).render(), False
                    except (CommandRejected, OSError) as exc:
                        out, err = str(exc), True
                    calls.append(
                        {"tool": st["tool"], "args": st["args"], "output": out, "error": err}
                    )
                    continue
                res = await clients[server].call_tool(name, st["args"])
                calls.append({"tool": st["tool"], "args": st["args"], "output": _text(res),
                              "error": bool(res.is_error)})  # fmt: skip
        outbox = []
        if (base / "outbox.jsonl").exists():
            outbox = [
                json.loads(x) for x in (base / "outbox.jsonl").read_text("utf-8").splitlines() if x
            ]
        poison = []
        if (base / "poison.jsonl").exists():
            poison = [
                json.loads(x) for x in (base / "poison.jsonl").read_text("utf-8").splitlines() if x
            ]
        return {
            "id": job["id"], "calls": calls, "listings": listings,
            "fs_before": fs_before, "fs_after": _snapshot_fs(root),
            "db_before": db_before, "db_after": _snapshot_db(db),
            "outbox": outbox, "sink": web.sink.state.store.all(), "weather_log": poison,
        }  # fmt: skip


async def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        job = json.loads(line)
        try:
            result = await run_job(job)
        except Exception as exc:  # noqa: BLE001 - report per job, keep going
            result = {"id": job.get("id"), "crash": f"{type(exc).__name__}: {exc}"[:300]}
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
