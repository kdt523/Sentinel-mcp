from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from gateway.decider import RULE_REVOKED, RULE_TERMINATED, Decider, PipelineDecider
from gateway.models import Action, Decision, Direction, Event, Tier
from gateway.policy import PolicyContext, PolicyEngine
from gateway.session import SessionTracker

POLICY = Path(__file__).resolve().parents[2] / "policies" / "default.yaml"


class FakeGuard:
    def __init__(self, verdict: Decision | None) -> None:
        self.verdict = verdict
        self.calls: list[tuple[Event, PolicyContext]] = []

    async def decide(self, event: Event, ctx: PolicyContext) -> Decision | None:
        self.calls.append((event, ctx))
        return self.verdict


@pytest.fixture
async def tracker() -> AsyncIterator[SessionTracker]:
    r = fakeredis.FakeAsyncRedis()
    yield SessionTracker(r)
    await r.aclose()


@pytest.fixture(scope="module")
def policy() -> PolicyEngine:
    return PolicyEngine.load(POLICY)


def req(tool: str, **args: Any) -> Event:
    server, _, name = tool.partition(".")
    return Event(session_id="s1", direction=Direction.REQUEST, server=server, tool=name, args=args)


def test_fake_guard_satisfies_protocol() -> None:
    assert isinstance(FakeGuard(None), Decider)


async def test_no_opinion_is_default_tier_allow(
    tracker: SessionTracker, policy: PolicyEngine
) -> None:
    d = await PipelineDecider(tracker, policy).decide(
        req("filesystem.read_file", path="docs/readme.md")
    )
    assert d.action is Action.ALLOW and d.risk == 0.0
    assert d.tier is Tier.DEFAULT and d.rule_id is None
    assert d.escalated is False
    assert d.model_version == policy.model_version
    assert d.latency_ms > 0


async def test_policy_hit_carries_rule_id(tracker: SessionTracker, policy: PolicyEngine) -> None:
    d = await PipelineDecider(tracker, policy).decide(
        req("sql_db.execute", sql="DROP TABLE orders")
    )
    assert d.action is Action.BLOCK and d.tier is Tier.POLICY
    assert d.rule_id == "destructive-sql" and d.attack_family == "destructive"


async def test_escalation_without_guard_falls_back(
    tracker: SessionTracker, policy: PolicyEngine
) -> None:
    d = await PipelineDecider(tracker, policy).decide(req("shell.run", cmd="python3 -c 1"))
    assert d.action is Action.REQUIRE_APPROVAL
    assert d.tier is Tier.FALLBACK and d.rule_id == "python3-escalate"
    assert "escalated" in d.reason
    assert d.escalated is True


async def test_escalation_goes_to_guard(tracker: SessionTracker, policy: PolicyEngine) -> None:
    verdict = Decision(action=Action.BLOCK, risk=0.8, reason="runs code that reads secrets",
                       tier=Tier.GUARD, model_version="guard-test")  # fmt: skip
    guard = FakeGuard(verdict)
    d = await PipelineDecider(tracker, policy, guard=guard).decide(
        req("shell.run", cmd="python3 -c 'print(open(\".env\").read())'")
    )
    assert len(guard.calls) == 1
    assert d.action is Action.BLOCK and d.tier is Tier.GUARD
    assert d.rule_id == "python3-escalate"  # provenance of the escalation is kept
    assert d.escalated is True


async def test_guard_abstaining_falls_back(tracker: SessionTracker, policy: PolicyEngine) -> None:
    d = await PipelineDecider(tracker, policy, guard=FakeGuard(None)).decide(
        req("shell.run", cmd="python3 -V")
    )
    assert d.tier is Tier.FALLBACK and d.action is Action.REQUIRE_APPROVAL


async def test_guard_not_consulted_without_escalation(
    tracker: SessionTracker, policy: PolicyEngine
) -> None:
    guard = FakeGuard(None)
    decider = PipelineDecider(tracker, policy, guard=guard)
    await decider.decide(req("filesystem.read_file", path="docs/readme.md"))
    await decider.decide(req("sql_db.execute", sql="DROP TABLE orders"))
    assert guard.calls == []  # Phase 1: guard only serves escalations


async def test_terminated_session_blocks_everything(
    tracker: SessionTracker, policy: PolicyEngine
) -> None:
    await tracker.terminate("s1", "exfiltration")
    decider = PipelineDecider(tracker, policy)
    d = await decider.decide(req("filesystem.read_file", path="docs/readme.md"))
    out = Event.from_output(session_id="s1", server="filesystem", tool="read_file",
                            full_text="x", max_chars=100)  # fmt: skip
    d_out = await decider.decide(out)
    for dec in (d, d_out):
        assert dec.action is Action.BLOCK and dec.rule_id == RULE_TERMINATED


async def test_revoked_tool_blocked_others_not(
    tracker: SessionTracker, policy: PolicyEngine
) -> None:
    await tracker.revoke_tool("s1", "weather.get_forecast")
    decider = PipelineDecider(tracker, policy)
    revoked = await decider.decide(req("weather.get_forecast", city="Pune"))
    other = await decider.decide(
        req("weather.convert_currency", amount=1, from_code="usd", to_code="inr")
    )
    assert revoked.action is Action.BLOCK and revoked.rule_id == RULE_REVOKED
    assert other.action is Action.ALLOW and other.tier is Tier.DEFAULT


async def test_terminated_check_precedes_policy(
    tracker: SessionTracker, policy: PolicyEngine
) -> None:
    await tracker.terminate("s1", "x")
    d = await PipelineDecider(tracker, policy).decide(
        req("sql_db.execute", sql="DROP TABLE orders")
    )
    assert d.rule_id == RULE_TERMINATED
