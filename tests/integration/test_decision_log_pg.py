"""Migration + decision recorder against the real Postgres from docker compose (`make dev`).

Uses a throwaway database so the dev database is never touched.
"""

from __future__ import annotations

import uuid
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url

from gateway.config import Settings
from gateway.models import Action, Decision, Direction, Event, Tier
from gateway.recorder import DecisionRecorder, create_db_engine
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
    settings = Settings(database_url=db_url)
    engine = create_db_engine(settings)
    rec = DecisionRecorder(engine, batch_size=4)
    await rec.start()
    env = (SEED_DIR / ".env").read_text(encoding="utf-8")
    for i in range(10):
        ev = Event(session_id="pg-s1", direction=Direction.REQUEST, server="weather",
                   tool="get_forecast", args={"city": "Pune", "context": env, "i": i})  # fmt: skip
        d = Decision(action=Action.TERMINATE_SESSION if i == 9 else Action.ALLOW,
                     risk=1.0 if i == 9 else 0.0, reason="r", tier=Tier.POLICY,
                     rule_id="secret-flow" if i == 9 else None, model_version="t")  # fmt: skip
        rec.submit(ev, d, enforced=True)
    await rec.stop()
    assert rec.written == 10 and rec.failed == rec.dropped == 0

    async with engine.connect() as conn:
        n = (await conn.execute(text("SELECT count(*) FROM decisions"))).scalar_one()
        sess = (await conn.execute(text(
            "SELECT event_count, terminated FROM sessions WHERE session_id='pg-s1'"
        ))).one()  # fmt: skip
        args = (await conn.execute(text(
            "SELECT args_redacted->>'context' FROM decisions ORDER BY id LIMIT 1"
        ))).scalar_one()  # fmt: skip
    await engine.dispose()
    assert n == 10
    assert sess.event_count == 10 and sess.terminated is True  # upsert across 3 batches
    assert "sk-fake" not in args and "[REDACTED]" in args
