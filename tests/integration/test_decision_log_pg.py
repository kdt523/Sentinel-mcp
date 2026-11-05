"""Migration + decision recorder against the real Postgres from docker compose (`make dev`).

Uses a throwaway database so the dev database is never touched.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from argparse import Namespace
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

from gateway.config import Settings
from gateway.models import Action, Decision, Direction, Event, Tier
from gateway.recorder import DecisionRecorder
from tools_servers.sandbox import SEED_DIR

pytestmark = pytest.mark.integration
REPO = Path(__file__).resolve().parents[2]


def _libpq(url: str) -> str:
    return make_url(url).set(drivername="postgresql").render_as_string(hide_password=False)


@pytest.fixture(scope="module")
def db_url() -> Iterator[str]:
    base = make_url(Settings().database_url)
    admin = _libpq(str(base.set(database="postgres").render_as_string(hide_password=False)))
    name = f"sentinel_itest_{uuid.uuid4().hex[:8]}"
    try:
        with psycopg.connect(admin, connect_timeout=3, autocommit=True) as conn:
            conn.execute(f'CREATE DATABASE "{name}"')
    except psycopg.OperationalError:
        pytest.skip("postgres not reachable (run `make dev`)")
    url = base.set(database=name).render_as_string(hide_password=False)
    try:
        yield url
    finally:
        with psycopg.connect(admin, connect_timeout=3, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def test_migration_upgrade_and_downgrade_on_postgres(db_url: str) -> None:
    cfg = Config(str(REPO / "alembic.ini"), cmd_opts=Namespace(x=[f"url={db_url}"]))
    command.upgrade(cfg, "head")
    with psycopg.connect(_libpq(db_url)) as conn:
        cols: dict[str, str] = dict(conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'decisions'"
        ).fetchall())  # fmt: skip
    assert cols["args_redacted"] == "jsonb"
    assert cols["id"] == "bigint"
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")  # leave migrated for the next test


async def test_recorder_writes_to_postgres(db_url: str) -> None:
    rec = DecisionRecorder.from_settings(Settings(database_url=db_url, decision_log_batch_size=4))
    await rec.start()
    env = (SEED_DIR / ".env").read_text(encoding="utf-8")
    for i in range(10):
        ev = Event(session_id="pg-s1", direction=Direction.REQUEST, server="weather",
                   tool="get_forecast", args={"city": "Pune", "context": env, "i": i})  # fmt: skip
        d = Decision(action=Action.TERMINATE_SESSION if i == 9 else Action.ALLOW,
                     risk=1.0 if i == 9 else 0.0, reason="r", tier=Tier.POLICY,
                     rule_id="secret-flow" if i == 9 else None, model_version="t")  # fmt: skip
        rec.submit(ev, d, enforced=True)
    rec.submit_approval({
        "approval_id": "a1", "event_id": "e1", "session_id": "pg-s1",
        "qualified_tool": "email_send.send", "status": "approved", "reason": "r",
        "requested_at": datetime.now(UTC), "resolved_at": datetime.now(UTC),
        "resolved_by": "kd", "note": "ok",
    })  # fmt: skip
    await rec.ping()
    await rec.stop()
    assert rec.written == 11 and rec.failed == rec.dropped == 0

    with psycopg.connect(_libpq(db_url)) as conn:
        n = conn.execute("SELECT count(*) FROM decisions").fetchone()
        sess = conn.execute(
            "SELECT event_count, terminated FROM sessions WHERE session_id='pg-s1'"
        ).fetchone()
        args = conn.execute(
            "SELECT args_redacted->>'context' FROM decisions ORDER BY id LIMIT 1"
        ).fetchone()
        approval = conn.execute("SELECT status, resolved_by FROM approvals").fetchone()
    assert n is not None and n[0] == 10
    assert sess == (10, True)  # upsert across 3 batches
    assert args is not None and "sk-fake" not in args[0] and "[REDACTED]" in args[0]
    assert approval == ("approved", "kd")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows event-loop check")
async def test_decision_log_writes_natively_on_windows_proactor_loop(db_url: str) -> None:
    """Regression: psycopg's async mode cannot run on the ProactorEventLoop, which the
    gateway needs on Windows for stdio subprocesses. The sync sink in a worker thread must
    work on that loop (this test runs WITHOUT any event-loop override)."""
    loop = asyncio.get_running_loop()
    assert isinstance(loop, asyncio.ProactorEventLoop)  # type: ignore[attr-defined,unused-ignore]
    rec = DecisionRecorder.from_settings(Settings(database_url=db_url))
    await rec.start()
    ev = Event(session_id="win-s1", direction=Direction.REQUEST, server="filesystem",
               tool="read_file", args={"path": "docs/readme.md"})  # fmt: skip
    rec.submit(ev, Decision(action=Action.ALLOW, risk=0, reason="r", tier=Tier.DEFAULT,
                            model_version="t"), enforced=True)  # fmt: skip
    await rec.ping()
    await rec.stop()
    assert rec.written == 1 and rec.failed == 0
    with psycopg.connect(_libpq(db_url)) as conn:
        row = conn.execute("SELECT count(*) FROM decisions WHERE session_id='win-s1'").fetchone()
    assert row == (1,)
