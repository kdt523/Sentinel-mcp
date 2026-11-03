import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.approvals import ApprovalStore
from gateway.db.models import ApprovalRow, Base
from gateway.decider import PipelineDecider
from gateway.enforcer import Enforcer, decide_with_fail_mode
from gateway.models import Action, ApprovalStatus, Decision, Direction, Event, Tier
from gateway.policy import PolicyEngine
from gateway.recorder import DecisionRecorder
from gateway.session import SessionTracker
from tools_servers.poisoned_output.server import TICKETS

POLICY = Path(__file__).resolve().parents[2] / "policies" / "default.yaml"


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    r = fakeredis.FakeAsyncRedis()
    yield r
    await r.aclose()


@pytest.fixture
def tracker(redis: fakeredis.FakeAsyncRedis) -> SessionTracker:
    return SessionTracker(redis)


@pytest.fixture
def store(redis: fakeredis.FakeAsyncRedis) -> ApprovalStore:
    return ApprovalStore(redis)


def enforcer(tracker: SessionTracker, store: ApprovalStore, **kw: Any) -> Enforcer:
    kw.setdefault("shadow_mode", False)
    kw.setdefault("approval_timeout_s", 2.0)
    kw.setdefault("poll_interval_s", 0.02)
    kw.setdefault("progress_interval_s", 0.05)
    return Enforcer(tracker, store, **kw)


def req(tool: str = "email_send.send", **args: Any) -> Event:
    server, _, name = tool.partition(".")
    return Event(session_id="s1", direction=Direction.REQUEST, server=server, tool=name,
                 args=args or {"to": "x@evil.example", "subject": "s", "body": "b"})  # fmt: skip


def dec(action: Action, reason: str = "because") -> Decision:
    return Decision(action=action, risk=0.5, reason=reason, tier=Tier.POLICY, rule_id="r",
                    model_version="t")  # fmt: skip


class Progress:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[float, float | None, str]] = []
        self.fail = fail

    async def __call__(self, done: float, total: float | None, msg: str) -> None:
        self.calls.append((done, total, msg))
        if self.fail:
            raise RuntimeError("client gone")


# --- simple actions --------------------------------------------------------------------


async def test_allow_and_block(tracker: SessionTracker, store: ApprovalStore) -> None:
    e = enforcer(tracker, store)
    ok = await e.enforce(req(), dec(Action.ALLOW))
    blocked = await e.enforce(req(), dec(Action.BLOCK, "Destructive SQL statement (DROP)."))
    assert ok.forward and ok.enforced
    assert not blocked.forward and blocked.message is not None
    assert "Destructive SQL statement (DROP)." in blocked.message


async def test_revoke_and_terminate_side_effects(
    tracker: SessionTracker, store: ApprovalStore
) -> None:
    e = enforcer(tracker, store)
    r = await e.enforce(req("weather.get_forecast", city="x"), dec(Action.REVOKE_TOOL))
    assert not r.forward and await tracker.is_revoked("s1", "weather.get_forecast")
    t = await e.enforce(req(), dec(Action.TERMINATE_SESSION, "exfil"))
    assert not t.forward and t.message is not None and "terminated" in t.message
    assert await tracker.is_terminated("s1")


async def test_sanitize_output_and_request(tracker: SessionTracker, store: ApprovalStore) -> None:
    e = enforcer(tracker, store)
    out = Event.from_output(session_id="s1", server="notes", tool="get_ticket",
                            full_text=TICKETS["T-2002"], max_chars=10_000)  # fmt: skip
    s = await e.enforce(out, dec(Action.SANITIZE))
    assert s.forward and s.output is not None and "DELETE FROM" not in s.output
    assert s.sanitized == ("markdown_comment",)
    on_request = await e.enforce(req(), dec(Action.SANITIZE))
    assert not on_request.forward


async def test_shadow_mode_forwards_without_side_effects(
    tracker: SessionTracker, store: ApprovalStore
) -> None:
    e = enforcer(tracker, store, shadow_mode=True)
    for action in Action:
        res = await e.enforce(req(), dec(action))
        assert res.forward and not res.enforced
    assert not await tracker.is_terminated("s1")
    assert not await tracker.is_revoked("s1", "email_send.send")
    assert await store.list_pending() == []


# --- approvals -------------------------------------------------------------------------


async def _resolve_soon(
    store: ApprovalStore, to: ApprovalStatus, *, by: str = "kd", note: str = ""
) -> None:
    for _ in range(200):
        pending = await store.list_pending()
        if pending:
            await asyncio.sleep(0.05)
            await store.transition(pending[0].approval_id, to, by=by, note=note)
            return
        await asyncio.sleep(0.01)


async def test_approved_call_is_forwarded(tracker: SessionTracker, store: ApprovalStore) -> None:
    progress = Progress()
    e = enforcer(tracker, store)
    res, _ = await asyncio.gather(
        e.enforce(req(), dec(Action.REQUIRE_APPROVAL), progress=progress),
        _resolve_soon(store, ApprovalStatus.APPROVED),
    )
    assert res.forward and res.approval is not None
    assert res.approval.status is ApprovalStatus.APPROVED and res.approval.resolved_by == "kd"
    assert progress.calls and progress.calls[0][1] == 2.0
    assert "Waiting for human approval" in progress.calls[0][2]


async def test_denied_call_reports_reviewer(tracker: SessionTracker, store: ApprovalStore) -> None:
    e = enforcer(tracker, store)
    res, _ = await asyncio.gather(
        e.enforce(req(), dec(Action.REQUIRE_APPROVAL, "External email.")),
        _resolve_soon(store, ApprovalStatus.DENIED, note="not an approved vendor"),
    )
    assert not res.forward and res.message is not None
    assert "rejected by kd (not an approved vendor)" in res.message
    assert "External email." in res.message


async def test_timeout_is_deny_and_says_so(tracker: SessionTracker, store: ApprovalStore) -> None:
    e = enforcer(tracker, store, approval_timeout_s=0.2)
    res = await e.enforce(req(), dec(Action.REQUIRE_APPROVAL))
    assert not res.forward and res.message is not None
    assert "timed out after 0.2s" in res.message and "treated as denied" in res.message
    assert res.approval is not None and res.approval.status is ApprovalStatus.EXPIRED
    assert await store.list_pending() == []


async def test_timeout_denies_even_for_fail_open_tools(
    tracker: SessionTracker, store: ApprovalStore
) -> None:
    # Fail mode is about decider failures; a human not answering is never an ALLOW.
    e = enforcer(tracker, store, approval_timeout_s=0.1)
    res = await e.enforce(req("sql_db.query", sql="select 1"), dec(Action.REQUIRE_APPROVAL))
    assert not res.forward


async def test_agent_disconnect_cancels_approval(
    tracker: SessionTracker, store: ApprovalStore
) -> None:
    e = enforcer(tracker, store, approval_timeout_s=30)
    task = asyncio.create_task(e.enforce(req(), dec(Action.REQUIRE_APPROVAL)))
    for _ in range(100):
        if await store.list_pending():
            break
        await asyncio.sleep(0.01)
    [pending] = await store.list_pending()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await store.status(pending.approval_id) is ApprovalStatus.CANCELLED


async def test_human_decision_wins_race_with_expiry(store: ApprovalStore) -> None:
    a = await store.create(req(), dec(Action.REQUIRE_APPROVAL), preview="{}")
    assert (await store.transition(a.approval_id, ApprovalStatus.APPROVED, by="kd")).ok
    late = await store.transition(a.approval_id, ApprovalStatus.EXPIRED, by="sentinel")
    assert not late.ok and late.status is ApprovalStatus.APPROVED
    assert not (await store.transition("nope", ApprovalStatus.DENIED)).ok
    with pytest.raises(ValueError):
        await store.transition(a.approval_id, ApprovalStatus.PENDING)


async def test_progress_errors_do_not_break_waiting(
    tracker: SessionTracker, store: ApprovalStore
) -> None:
    e = enforcer(tracker, store)
    res, _ = await asyncio.gather(
        e.enforce(req(), dec(Action.REQUIRE_APPROVAL), progress=Progress(fail=True)),
        _resolve_soon(store, ApprovalStatus.APPROVED),
    )
    assert res.forward


async def test_approval_store_down_means_deny(tracker: SessionTracker, store: ApprovalStore,
                                              monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    async def boom(*a: Any, **k: Any) -> None:
        raise ConnectionError("redis down")

    monkeypatch.setattr(store, "create", boom)
    res = await enforcer(tracker, store).enforce(req(), dec(Action.REQUIRE_APPROVAL))
    assert not res.forward and res.message is not None and "denied" in res.message


async def test_approval_lifecycle_is_recorded(
    tracker: SessionTracker, store: ApprovalStore, tmp_path: Path
) -> None:
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'a.db'}")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    rec = DecisionRecorder(eng)
    await rec.start()
    e = enforcer(tracker, store, recorder=rec, approval_timeout_s=0.15)
    expired = await e.enforce(req(), dec(Action.REQUIRE_APPROVAL))
    approved, _ = await asyncio.gather(
        e.enforce(req(), dec(Action.REQUIRE_APPROVAL)),
        _resolve_soon(store, ApprovalStatus.APPROVED),
    )
    await rec.stop()
    async with eng.connect() as conn:
        got = {r.approval_id: r for r in (await conn.execute(select(ApprovalRow))).all()}
    await eng.dispose()
    assert expired.approval is not None and approved.approval is not None
    assert got[expired.approval.approval_id].status == "expired"
    assert got[approved.approval.approval_id].status == "approved"
    assert got[approved.approval.approval_id].resolved_by == "kd"


# --- fail modes -------------------------------------------------------------------------


class _Boom(PipelineDecider):
    def __init__(
        self, tracker: SessionTracker, exc: BaseException | None, delay: float = 0
    ) -> None:
        super().__init__(tracker, PolicyEngine.load(POLICY))
        self.exc, self.delay = exc, delay

    async def decide(self, event: Event) -> Decision:
        await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return await super().decide(event)


@pytest.mark.parametrize(("mode", "action"), [("closed", Action.BLOCK), ("open", Action.ALLOW)])
async def test_decider_error_follows_fail_mode(
    tracker: SessionTracker, mode: str, action: Action
) -> None:
    d = await decide_with_fail_mode(_Boom(tracker, ConnectionError("redis down")), req(),
                                    fail_mode=mode, timeout_s=1)  # type: ignore[arg-type]  # fmt: skip
    assert d.action is action and d.tier is Tier.FALLBACK
    assert d.rule_id == f"builtin:fail-{mode}" and "ConnectionError" in d.reason


async def test_decider_timeout_follows_fail_mode(tracker: SessionTracker) -> None:
    d = await decide_with_fail_mode(_Boom(tracker, None, delay=1.0), req(), fail_mode="closed",
                                    timeout_s=0.05)  # fmt: skip
    assert d.action is Action.BLOCK and "TimeoutError" in d.reason


async def test_cancellation_is_not_swallowed(tracker: SessionTracker) -> None:
    task = asyncio.create_task(
        decide_with_fail_mode(_Boom(tracker, None, delay=5), req(), fail_mode="open", timeout_s=10)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancel_while_create_in_flight_still_cancels(
    tracker: SessionTracker, store: ApprovalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the approval is already stored but create() has not returned yet."""
    real_create = store.create
    stored = asyncio.Event()

    async def slow_create(*a: Any, **k: Any) -> Any:
        approval = await real_create(*a, **k)
        stored.set()
        await asyncio.sleep(10)  # cancelled here, before the caller ever sees the approval
        return approval

    monkeypatch.setattr(store, "create", slow_create)
    task = asyncio.create_task(
        enforcer(tracker, store).enforce(req(), dec(Action.REQUIRE_APPROVAL))
    )
    await stored.wait()
    [pending] = await store.list_pending()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await store.status(pending.approval_id) is ApprovalStatus.CANCELLED
