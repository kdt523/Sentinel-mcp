"""Phase 2 step 5: replay composite sessions through a capturing gateway and build Samples.

Every session runs against a FRESH sandbox holding that session's own randomised fake
secrets and customers (``SessionValues``), through a gateway in shadow mode with raw
capture on. Raw capture is refused by the gateway unless it runs in sandbox mode
(``Settings._capture_only_in_sandbox``); this runner only ever starts sandbox gateways.

Backends:
- in-process (default): a fresh gateway per session with every tool server except the
  shell (fresh Redis, fresh rug-pull server, weather mode chosen per session);
- stack (``--stack``): sessions that use the shell run on the compose stack, whose gateway
  is restarted in shadow + capture + sandbox mode with ``config/servers.capture.yaml``.

Labelling (data/LABELING.md, via data/labeling.py) is applied to the events that actually
happened: request / output events of a step carry the step's attack facts; tool
descriptions are malicious only when the targeted tool's served description is poisoned
(``description_changed`` comes from the captured event); an output carrying an injection
that the tool stripped is benign and ``neutralised_by_tool``. Events after the first gold
TERMINATE_SESSION are marked ``post_termination``. Gateway decisions are stored only as
``gateway_decision`` (a baseline feature), never used as labels.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import html
import json
import sqlite3
import tempfile
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data.benign.templates import StepCall
from data.composer import Composite, Unit
from data.labeling import EventFacts, gold_action
from data.sandbox_values import SessionValues
from data.schema import (
    HISTORY_MAX,
    OUTPUT_MAX_CHARS,
    AttackFamily,
    CompactEvent,
    EventPayload,
    GatewayVerdict,
    Label,
    Sample,
    SessionContext,
)
from gateway.models import Action, Decision, Direction, Event, EventKind
from gateway.session import SessionState, _preview
from tools_servers.content_gen import ZW, ContentIdError, generate_note, generate_page
from tools_servers.poisoned_desc.server import FORECAST_CLEAN

REPO = Path(__file__).resolve().parents[1]
CLEAN_DESCRIPTIONS = {"weather.get_forecast": FORECAST_CLEAN}


class CaptureError(RuntimeError):
    """Captured events do not line up with the session's scripted steps."""


@dataclass(frozen=True)
class Captured:
    event: Event
    decision: Decision
    state_before: SessionState | None
    raw_output: str | None


@dataclass
class SessionResult:
    composite: Composite
    records: list[Captured]
    step_errors: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- sandbox


def prepare_session_sandbox(fs_root: Path, db_path: Path, values: SessionValues) -> None:
    """Fresh workspace + database holding THIS session's fake secrets and customers.

    Seed orders are kept for customers that exist in this session (ids 1001.. always do)."""
    from tools_servers.sandbox import prepare_sandbox
    from tools_servers.sql_db.server import prepare_database

    prepare_sandbox(fs_root, reset=True)
    values.render_into(fs_root)
    prepare_database(db_path, reset=True)
    with sqlite3.connect(db_path) as conn:
        orders = conn.execute("SELECT * FROM orders").fetchall()
        for sql, params in values.customers_sql():
            conn.execute(sql, params)
        ids = {c.customer_id for c in values.customers}
        placeholders = ", ".join("?" * len(orders[0])) if orders else ""
        for row in orders:
            if row[1] in ids:
                conn.execute(f"INSERT INTO orders VALUES ({placeholders})", row)  # noqa: S608
    conn.close()


def weather_config(c: Composite) -> tuple[str, int | None]:
    """(mode, variant) for the weather server: clean unless a unit asks for poisoning."""
    for u in c.units:
        cfg = dict(dict(u.upstreams).get("weather", ()))
        if cfg:
            v = cfg.get("variant")
            return cfg.get("mode", "static"), int(v) if v not in (None, "", "*") else None
    return "clean", None


# ----------------------------------------------------------------------------- capture io


def read_capture(capture_dir: Path) -> list[Captured]:
    out = []
    for path in sorted(capture_dir.glob("events-*.jsonl")):
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                raw = json.loads(line)
                sb = raw.get("state_before")
                out.append(Captured(
                    event=Event.model_validate(raw["event"]),
                    decision=Decision.model_validate(raw["decision"]),
                    state_before=SessionState.model_validate(sb) if sb else None,
                    raw_output=raw.get("raw_output"),
                ))  # fmt: skip
    return out


async def replay_steps(client: Any, steps: Sequence[StepCall]) -> list[str]:
    from agents.outcomes import classify, result_text

    errors = []
    for i, st in enumerate(steps):
        if st.op == "list_tools":
            await client.list_tools()
            continue
        res = await client.call_tool(st.tool, st.args)
        if classify(res) == "ERROR":
            errors.append(f"step {i} {st.tool}: {result_text(res)[:160]}")
    return errors


# ----------------------------------------------------------------------------- in-process


@asynccontextmanager
async def capturing_gateway(workdir: Path, weather_mode: str,
                            weather_variant: int | None) -> AsyncGenerator[Any]:  # fmt: skip
    """A fresh in-process gateway (shadow mode, raw capture, sandbox mode) over the
    session's sandbox. The capture is flushed when the context exits."""
    import anyio
    import fakeredis
    from sqlalchemy.ext.asyncio import create_async_engine

    from gateway.app import create_app
    from gateway.config import Settings
    from gateway.db.models import Base
    from gateway.db.sink import create_sink
    from tools_servers.email_send.server import build_server as build_email
    from tools_servers.filesystem.server import build_server as build_fs
    from tools_servers.http_fetch.server import build_server as build_fetch
    from tools_servers.poisoned_desc.server import build_server as build_weather
    from tools_servers.poisoned_output.server import build_server as build_notes
    from tools_servers.sandbox_net import SandboxWeb
    from tools_servers.sql_db.server import build_server as build_sql

    settings = Settings(
        policy_path=REPO / "policies" / "default.yaml",
        shadow_mode=True,
        capture_mode=True,
        sandbox_mode=True,  # raw capture is refused outside the sandbox
        capture_dir=workdir / "raw",
        approval_timeout_s=2,
        approval_poll_interval_s=0.02,
    )
    db = workdir / "log.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    targets = {
        "filesystem": build_fs(workdir / "fs"),
        "sql_db": build_sql(workdir / "db" / "app.db"),
        "email_send": build_email(workdir / "outbox.jsonl", ("acme.example",)),
        "http_fetch": build_fetch(("mockweb", "exfil-sink"), transport=SandboxWeb()),
        "weather": build_weather(
            weather_mode,  # type: ignore[arg-type]
            workdir / "poison.jsonl",
            variant=weather_variant,
        ),  # fmt: skip
        "notes": build_notes(),
    }
    app = create_app(settings, redis=fakeredis.FakeAsyncRedis(),
                     sink=create_sink(f"sqlite:///{db}"), upstream_targets=targets)  # fmt: skip
    async with app.api.router.lifespan_context(app.api):
        with anyio.fail_after(120):
            yield app


async def run_inprocess(c: Composite, workdir: Path) -> SessionResult:
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    if any(st.tool.startswith("shell.") for st in c.steps):
        raise ValueError(f"{c.session_key} uses the shell: run it with the stack backend")
    workdir.mkdir(parents=True, exist_ok=True)
    values = SessionValues.generate(c.values_seed)
    prepare_session_sandbox(workdir / "fs", workdir / "db" / "app.db", values)
    mode, variant = weather_config(c)
    base = "http://127.0.0.1:8000"
    async with capturing_gateway(workdir, mode, variant) as app:
        hc = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=base)
        async with hc, Client(streamable_http_client(f"{base}/mcp", http_client=hc)) as client:
            errors = await replay_steps(client, c.steps)
    return SessionResult(c, read_capture(workdir / "raw"), errors)


# ----------------------------------------------------------------------------- stack (shell)

STACK_MCP_URL = "http://127.0.0.1:8000/mcp"
STACK_CAPTURE = "/app/sandbox_runtime/capture"
STACK_CAPTURE_ENV = {
    "GATEWAY_SHADOW_MODE": "true",
    "GATEWAY_CAPTURE_MODE": "true",
    "GATEWAY_SANDBOX_MODE": "true",
    "GATEWAY_SERVERS_PATH": "config/servers.capture.yaml",
}
# Runs INSIDE the gateway container (which has tools_servers but not data/): fresh sandbox
# with the session's rendered files, reseeded customers, empty capture directory.
_STACK_PREPARE = """
import json, sqlite3, sys
from pathlib import Path
from tools_servers.sandbox import prepare_sandbox
from tools_servers.sql_db.server import prepare_database
d = json.load(sys.stdin)
fs = Path('/app/sandbox_runtime/fs')
prepare_sandbox(fs, reset=True)
for rel, text in d['files'].items():
    p = fs / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
db = prepare_database(Path('/app/sandbox_runtime/db/app.db'), reset=True)
with sqlite3.connect(db) as conn:
    orders = conn.execute('SELECT * FROM orders').fetchall()
    for sql, params in d['sql']:
        conn.execute(sql, params)
    for row in orders:
        if row[1] in d['ids']:
            conn.execute('INSERT INTO orders VALUES (' + ', '.join('?' * len(row)) + ')', row)
conn.close()
cap = Path(d['capture'])
cap.mkdir(parents=True, exist_ok=True)  # the recorder only creates it at startup
for old in cap.glob('events-*.jsonl'):
    old.unlink()
"""
_STACK_READ = (
    "import glob, sys; [sys.stdout.write(open(p, encoding='utf-8').read()) "
    f"for p in sorted(glob.glob('{STACK_CAPTURE}/events-*.jsonl'))]"
)


def _docker() -> str:
    import shutil

    for cand in ("docker", r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"):
        if found := shutil.which(cand) or (Path(cand).exists() and cand):
            return str(found)
    raise RuntimeError("docker CLI not found")


def _compose(*args: str, stdin: str | None = None, env: dict[str, str] | None = None,
             timeout: int = 300) -> str:  # fmt: skip
    import os
    import subprocess

    proc = subprocess.run([_docker(), "compose", *args], cwd=REPO, input=stdin,  # noqa: S603
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, env={**os.environ, **(env or {})})  # fmt: skip
    if proc.returncode != 0:
        raise RuntimeError(f"docker compose {' '.join(args[:3])}: {proc.stderr[-1500:]}")
    return proc.stdout


def restart_stack_gateway(env: dict[str, str] | None = None) -> None:
    """Recreate the stack's gateway (``STACK_CAPTURE_ENV`` for capture; None = defaults)."""
    _compose("up", "-d", "--force-recreate", "--no-deps", "--wait", "--wait-timeout", "180",
             "gateway", env=env)  # fmt: skip


def _session_payload(values: SessionValues) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        values.render_into(Path(tmp))
        files = {p.relative_to(tmp).as_posix(): p.read_text("utf-8")
                 for p in Path(tmp).rglob("*") if p.is_file()}  # fmt: skip
    return json.dumps({"files": files, "sql": values.customers_sql(),
                       "ids": [c.customer_id for c in values.customers],
                       "capture": STACK_CAPTURE})  # fmt: skip


def _parse_capture(text: str) -> list[Captured]:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "events-0.jsonl").write_text(text, "utf-8")
        return read_capture(Path(tmp))


async def run_on_stack(c: Composite, *, settle_s: float = 0.4,
                       timeout_s: float = 20) -> SessionResult:  # fmt: skip
    """Replay one composite on the running stack. The gateway must have been restarted
    with ``STACK_CAPTURE_ENV`` (shadow + capture + sandbox mode, clean weather)."""
    import time

    from mcp import Client

    if weather_config(c)[0] != "clean":
        raise ValueError(f"{c.session_key} needs a poisoned weather server: run in-process")
    values = SessionValues.generate(c.values_seed)
    await asyncio.to_thread(_compose, "exec", "-T", "gateway", "python", "-c",
                            _STACK_PREPARE, stdin=_session_payload(values))  # fmt: skip
    async with Client(STACK_MCP_URL) as client:  # one connection = one session
        errors = await replay_steps(client, c.steps)
    # The decision log writes asynchronously: wait until the capture stops growing.
    deadline, last, stable_since = time.monotonic() + timeout_s, -1, time.monotonic()
    min_events = 2 * sum(st.op == "call" for st in c.steps)
    text = ""
    while time.monotonic() < deadline:
        text = await asyncio.to_thread(_compose, "exec", "-T", "gateway", "python", "-c",
                                       _STACK_READ)  # fmt: skip
        n = text.count("\n")
        if n != last:
            last, stable_since = n, time.monotonic()
        elif n >= min_events and time.monotonic() - stable_since >= settle_s:
            break
        await asyncio.sleep(0.15)
    return SessionResult(c, _parse_capture(text), errors)


# ----------------------------------------------------------------------------- labelling


def _content_item(args: dict[str, Any]) -> Any:
    """The generated content item a fetch / note step reads, if any."""
    try:
        if "note_id" in args:
            return generate_note(str(args["note_id"]))
        url = str(args.get("url", ""))
        if "/g/" in url:
            return generate_page(url.rsplit("/g/", 1)[1].split("?")[0])
    except ContentIdError:
        return None
    return None


def injection_reached_agent(item: Any, output: str) -> bool:
    """Did the poisoned item's instruction survive the tool (e.g. not stripped as a
    comment in http_fetch text mode)?"""
    if item is None or not item.instruction:
        return True  # nothing to check: trust the step's facts
    plain = html.unescape(output).replace(ZW, "")
    if item.instruction in " ".join(plain.split()) or item.instruction in plain:
        return True
    blob = base64.b64encode(item.instruction.encode()).decode()
    return blob in output


def _facts(mark: Any, direction: Direction, kind: EventKind,
           description_changed: bool = False) -> EventFacts:  # fmt: skip
    return EventFacts(family=AttackFamily(mark.family), direction=direction, kind=kind,
                      leaked_class=mark.leaked_class,
                      prior_destructive_attempts=mark.prior_destructive_attempts,
                      description_changed=description_changed)  # fmt: skip


def _align(c: Composite, records: Sequence[Captured]) -> list[StepCall | None]:
    """The scripted step behind each captured non-description event (None for descriptions).
    Requests must match the steps' tools and arguments in order; outputs follow them."""
    calls = [st for st in c.steps if st.op == "call"]
    out: list[StepCall | None] = []
    i, current = 0, None
    for r in records:
        ev = r.event
        if ev.kind is EventKind.TOOL_DESCRIPTION:
            out.append(None)
            continue
        if ev.direction is Direction.REQUEST:
            if i >= len(calls):
                raise CaptureError(f"{c.session_key}: extra request {ev.qualified_tool}")
            st = calls[i]
            if st.tool != ev.qualified_tool or (ev.args or {}) != st.args:
                raise CaptureError(f"{c.session_key}: request {i} is {ev.qualified_tool} "
                                   f"{str(ev.args)[:80]}, expected {st.tool}")  # fmt: skip
            current = st
            i += 1
        elif current is None or current.tool != ev.qualified_tool:
            raise CaptureError(f"{c.session_key}: output of {ev.qualified_tool} without request")
        out.append(current)
    if i != len(calls):
        raise CaptureError(f"{c.session_key}: {len(calls) - i} scripted calls never captured")
    return out


def _focal(c: Composite) -> Unit:
    return c.units[c.focal_index]


def build_samples(result: SessionResult, *, generator: str = "scripted") -> list[Sample]:
    c, records = result.composite, result.records
    steps = _align(c, records)
    focal = _focal(c)
    attack = c.kind == "attack"
    session_family = AttackFamily(focal.family) if attack and focal.family else None
    targets = {st.attack.target_tool: st.attack for st in c.steps
               if st.attack is not None and st.attack.event == "description"}  # fmt: skip
    samples: list[Sample] = []
    history: list[CompactEvent] = []
    terminated = False
    for idx, (r, st) in enumerate(zip(records, steps, strict=True)):
        ev = r.event
        gold, family, neutralised = Action.ALLOW, None, False
        if ev.kind is EventKind.TOOL_DESCRIPTION:
            mark = targets.get(ev.qualified_tool)
            clean = CLEAN_DESCRIPTIONS.get(ev.qualified_tool)
            text = r.raw_output if r.raw_output is not None else (ev.output or "")
            if mark is not None and clean is not None and text != clean:
                gold = gold_action(_facts(mark, Direction.OUTPUT, EventKind.TOOL_DESCRIPTION,
                                          ev.description_changed))  # fmt: skip
                family = AttackFamily(mark.family)
        elif st is not None and st.attack is not None:
            m = st.attack
            if ev.direction is Direction.REQUEST and m.event == "request":
                gold = gold_action(_facts(m, Direction.REQUEST, EventKind.TOOL_CALL))
                family = AttackFamily(m.family)
            elif ev.direction is Direction.OUTPUT and m.event == "output":
                text = r.raw_output if r.raw_output is not None else (ev.output or "")
                if injection_reached_agent(_content_item(st.args), text):
                    gold = gold_action(_facts(m, Direction.OUTPUT, EventKind.TOOL_CALL))
                    family = AttackFamily(m.family)
                else:
                    neutralised = True
        malicious = gold is not Action.ALLOW
        payload = EventPayload(
            direction=ev.direction, kind=ev.kind, server=ev.server, tool=ev.tool,
            args=ev.args if ev.direction is Direction.REQUEST else None,
            output=None if ev.direction is Direction.REQUEST
            else (ev.output or "")[:OUTPUT_MAX_CHARS],
            output_sha256=ev.output_sha256, output_chars=ev.output_chars,
            description_changed=ev.description_changed,
        )  # fmt: skip
        sb = r.state_before or SessionState(session_id=ev.session_id)
        taint = [t for t, on in (("secret", sb.secret_taint), ("pii", sb.pii_taint)) if on]
        context = SessionContext(
            request_count=sb.request_count, files_read=list(sb.files_read),
            taint_classes=taint,
            taint_sources=list(sb.taint_sources), egress_targets=list(sb.egress_targets),
            revoked_tools=list(sb.revoked_tools),
        )  # fmt: skip
        d = r.decision
        samples.append(Sample(
            sample_id=hashlib.sha256(f"{c.session_key}:{idx}".encode()).hexdigest()[:16],
            session_id=c.session_key, step_idx=idx, history=history[-HISTORY_MAX:],
            context=context, event=payload,
            label=Label.MALICIOUS if malicious else Label.BENIGN, gold_action=gold,
            attack_family=family, session_family=session_family,
            attack_goal=focal.goal if attack else None,
            hard_negative=bool(st is not None and st.hard_negative and not malicious),
            source=focal.source, generator=generator, template_id=focal.template_id,
            seed_id=focal.parent or focal.seed_id, neutralised_by_tool=neutralised,
            gateway_decision=GatewayVerdict(action=d.action, tier=d.tier, rule_id=d.rule_id,
                                            risk=d.risk),
            post_termination=terminated, split=c.split,
        ))  # fmt: skip
        if gold is Action.TERMINATE_SESSION:
            terminated = True  # every later event never happens in an enforced deployment
        history.append(CompactEvent(direction=ev.direction, kind=ev.kind,
                                    tool=ev.qualified_tool, preview=_preview(ev)[:400],
                                    outcome=d.action))  # fmt: skip
    return samples


# ----------------------------------------------------------------------------- driver


def _uses_shell(c: Composite) -> bool:
    return any(st.tool.startswith("shell.") for st in c.steps)


async def run_all(composites: Sequence[Composite], out: Path, *, stack: bool = False,
                  workroot: Path | None = None) -> dict[str, int]:  # fmt: skip
    """Run every composite (shell ones on the stack when ``stack``) and append its samples
    to ``out`` (JSONL). Sessions whose captured events do not line up with their steps are
    reported and dropped, never silently kept."""
    stats = {"sessions": 0, "samples": 0, "skipped_shell": 0, "capture_errors": 0,
             "sessions_with_step_errors": 0, "post_termination_samples": 0,
             "attack_sessions_fully_neutralised": 0}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)

    def keep(result: SessionResult, fh: Any) -> None:
        stats["sessions_with_step_errors"] += bool(result.step_errors)
        try:
            samples = build_samples(result)
        except CaptureError as exc:
            stats["capture_errors"] += 1
            print(f"capture error: {exc}")
            return
        for smp in samples:
            fh.write(smp.model_dump_json() + "\n")
        stats["sessions"] += 1
        stats["samples"] += len(samples)
        stats["post_termination_samples"] += sum(x.post_termination for x in samples)
        if result.composite.kind == "attack" and not any(
            x.label is Label.MALICIOUS for x in samples
        ):
            stats["attack_sessions_fully_neutralised"] += 1

    local = [c for c in composites if not _uses_shell(c)]
    shell = [c for c in composites if _uses_shell(c)]
    with tempfile.TemporaryDirectory(dir=workroot) as tmp, out.open("a", encoding="utf-8") as fh:
        for n, c in enumerate(local):
            keep(await run_inprocess(c, Path(tmp) / f"s{n}"), fh)
        if not stack:
            stats["skipped_shell"] = len(shell)
            return stats
        await asyncio.to_thread(restart_stack_gateway, STACK_CAPTURE_ENV)
        try:
            for c in shell:
                keep(await run_on_stack(c), fh)
        finally:
            await asyncio.to_thread(restart_stack_gateway, None)  # back to enforcing
    return stats


def _composites(limit: int | None, seed: int) -> list[Composite]:
    from collections import Counter

    from data.attacks.templates import assign_attack_splits, compose_attacks
    from data.attacks.templates import expand_all as expand_attacks
    from data.benign.templates import expand_all as expand_benign
    from data.composer import (
        assign_splits,
        benign_counts_for_ratio,
        compose_benign,
        units_from_sessions,
    )
    from data.generators.mutators import changes_events, inherit_splits, mutate_all

    seeds = expand_attacks(seed=seed)
    splits = assign_attack_splits(seeds, seed=seed)
    by_id = {u.seed_id: u for u in seeds}
    mutants = [m for m in mutate_all(seeds, seed=seed) if changes_events(m, by_id)]
    splits |= inherit_splits(mutants, splits)
    benign = units_from_sessions(expand_benign(seed=seed))
    bsplits = assign_splits(benign, seed=seed)
    attacks = compose_attacks(seeds + mutants, splits, benign, bsplits, seed=seed)
    need = benign_counts_for_ratio(Counter(c.split for c in attacks))
    allc = attacks + compose_benign(benign, bsplits, n_per_split=need, seed=seed)
    return allc[:limit] if limit else allc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / "data" / "raw" / "samples.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--stack", action="store_true", help="also run shell sessions on the compose stack (docker)"
    )
    args = ap.parse_args()
    stats = asyncio.run(run_all(_composites(args.limit, args.seed), args.out, stack=args.stack))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
