import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from structlog.testing import capture_logs

from gateway import recorder as rec_mod
from gateway.db.models import Base, DecisionRow, SessionRow
from gateway.models import Action, Decision, Direction, Event, Tier
from gateway.recorder import DecisionRecorder, decision_row
from tools_servers.sandbox import SEED_DIR

ENV_TEXT = (SEED_DIR / ".env").read_text(encoding="utf-8")


def sink_for(engine: Any) -> Any:
    """Synchronous decision-log sink writing to the same SQLite file as ``engine``."""
    from gateway.db.sink import create_sink

    return create_sink(engine.url.render_as_string(hide_password=False))


SECRETS = ("sk-fake-0000", "fake-db-password-not-real", "fake-jwt-secret")


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'log.db'}")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rec_mod, "RETRY_DELAY_S", 0.01)


def req(sid: str = "s1", **args: Any) -> Event:
    return Event(session_id=sid, direction=Direction.REQUEST, server="filesystem",
                 tool="read_file", args=args or {"path": "docs/readme.md"})  # fmt: skip


def dec(action: Action = Action.ALLOW, reason: str = "ok", rule: str | None = None) -> Decision:
    return Decision(action=action, risk=0.5 if action is not Action.ALLOW else 0.0, reason=reason,
                    tier=Tier.POLICY if rule else Tier.DEFAULT, rule_id=rule,
                    model_version="policy:default@t")  # fmt: skip


async def rows(engine: AsyncEngine) -> list[Any]:  # sqlalchemy Row objects
    async with engine.connect() as conn:
        return list((await conn.execute(select(DecisionRow).order_by(DecisionRow.id))).all())


async def sessions(engine: AsyncEngine) -> dict[str, Any]:
    async with engine.connect() as conn:
        return {r.session_id: r for r in (await conn.execute(select(SessionRow))).all()}


async def test_writes_all_fields_and_session_rollup(engine: AsyncEngine) -> None:
    r = DecisionRecorder(sink_for(engine), batch_size=3)
    await r.start()
    for i in range(7):
        r.submit(req(path=f"docs/{i}.md"), dec(), enforced=False)
    r.submit(req(sid="s2"), dec(Action.TERMINATE_SESSION, "exfil", "secret-flow"), enforced=True)
    await r.stop()
    got = await rows(engine)
    assert len(got) == 8 and r.written == 8 and r.dropped == r.failed == 0
    last = got[-1]
    assert (last.action, last.tier, last.rule_id, last.enforced) == (
        "TERMINATE_SESSION", "policy", "secret-flow", True,
    )  # fmt: skip
    assert got[0].tier == "default" and got[0].rule_id is None and got[0].enforced is False
    sess = await sessions(engine)
    assert sess["s1"].event_count == 7 and not sess["s1"].terminated
    assert sess["s2"].terminated and sess["s2"].terminated_reason == "exfil"


async def test_shadow_terminate_does_not_mark_session_terminated(engine: AsyncEngine) -> None:
    r = DecisionRecorder(sink_for(engine))
    await r.start()
    r.submit(req(), dec(Action.TERMINATE_SESSION, "exfil", "secret-flow"), enforced=False)
    await r.stop()
    assert not (await sessions(engine))["s1"].terminated


async def test_no_raw_secrets_in_database(engine: AsyncEngine) -> None:
    r = DecisionRecorder(sink_for(engine))
    await r.start()
    leak = req(tool_arg=ENV_TEXT, nested={"x": ["sk-fake-0000000000000000000000000000SANDBOX"]})
    out = Event.from_output(session_id="s1", server="filesystem", tool="read_file",
                            full_text=ENV_TEXT, max_chars=16000)  # fmt: skip
    r.submit(leak, dec(), enforced=True)
    r.submit(out, dec(), enforced=True)
    await r.stop()
    dump = json.dumps([{c: str(getattr(x, c)) for c in x._fields} for x in await rows(engine)])
    for secret in SECRETS:
        assert secret not in dump
    assert "[REDACTED]" in dump
    stored_out = (await rows(engine))[1]
    assert stored_out.output_sha256 == out.output_sha256 and stored_out.output_chars == len(
        ENV_TEXT
    )


async def test_bounded_queue_drops_and_counts(engine: AsyncEngine) -> None:
    r = DecisionRecorder(sink_for(engine), queue_size=2)  # worker not started: nothing drains
    results = [r.submit(req(), dec(), enforced=True) for _ in range(5)]
    assert results == [True, True, False, False, False]
    assert r.dropped == 3 and r.submitted == 5 and r.pending == 2


async def test_submit_never_blocks_even_when_db_hangs(engine: AsyncEngine) -> None:
    import time

    sink = sink_for(engine)

    def hang(rows: list[dict[str, Any]]) -> None:  # blocking DB call stuck in its thread
        time.sleep(1.0)

    sink.write_decisions = hang
    r = DecisionRecorder(sink, queue_size=5)
    await r.start()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    for _ in range(20):
        r.submit(req(), dec(), enforced=True)
    assert loop.time() - t0 < 0.1
    assert r.dropped > 0
    await r.stop(timeout_s=0.2)  # flush times out -> remaining counted as dropped
    assert r.written == 0 and r.dropped + r.pending >= 15


async def test_db_failure_is_counted_and_worker_survives(tmp_path: Path) -> None:
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}")  # no tables yet
    r = DecisionRecorder(sink_for(eng))
    await r.start()
    with capture_logs() as logs:
        r.submit(req(), dec(), enforced=True)
        await asyncio.wait_for(r._queue.join(), 5)
    assert r.failed == 1 and r.written == 0
    assert any(e["event"] == "decision_log_write_failed" for e in logs)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    r.submit(req(), dec(), enforced=True)
    await r.stop()
    assert r.written == 1
    await eng.dispose()


async def test_bad_row_does_not_sink_its_batch(engine: AsyncEngine) -> None:
    r = DecisionRecorder(sink_for(engine), batch_size=50)
    dup = req()
    for ev in (req(), dup, dup, req()):  # duplicate event_id violates the unique constraint
        r.submit(ev, dec(), enforced=True)
    await r.start()
    await r.stop()
    assert r.written == 3 and r.failed == 1
    assert len(await rows(engine)) == 3


async def test_capture_mode_writes_raw_and_warns(engine: AsyncEngine, tmp_path: Path) -> None:
    cap = tmp_path / "raw"
    r = DecisionRecorder(sink_for(engine), capture_dir=cap)
    with capture_logs() as logs:
        await r.start()
    assert any(e["event"] == "raw_capture_enabled" and e["log_level"] == "warning" for e in logs)
    out = Event.from_output(session_id="s1", server="filesystem", tool="read_file",
                            full_text=ENV_TEXT * 3, max_chars=300)  # fmt: skip
    r.submit(out, dec(), enforced=False, raw_output=ENV_TEXT * 3)
    await r.stop()
    files = list(cap.glob("events-*.jsonl"))
    assert len(files) == 1
    raw = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert raw["raw_output"] == ENV_TEXT * 3  # full, unredacted, untruncated
    assert "sk-fake-0000" in raw["raw_output"]
    assert raw["decision"]["tier"] == "default" and raw["enforced"] is False
    db_dump = json.dumps([str(x) for x in await rows(engine)])
    assert "sk-fake-0000" not in db_dump  # the DB stays redacted even with capture on


async def test_no_capture_by_default(engine: AsyncEngine, tmp_path: Path) -> None:
    r = DecisionRecorder(sink_for(engine))
    assert not r.capturing
    await r.start()
    r.submit(req(), dec(), enforced=True)
    await r.stop()
    assert not any(tmp_path.rglob("events-*.jsonl"))


def test_decision_row_shape() -> None:
    row = decision_row(req(), dec(Action.BLOCK, "x", "r1"), enforced=True)
    assert {"session_id", "event_id", "direction", "qualified_tool", "tier", "rule_id", "action",
            "risk", "reason", "latency_ms", "model_version", "enforced"} <= set(row)  # fmt: skip
