"""Decision pipeline.

``Decider`` is the interface every tier implements: return a ``Decision`` or ``None`` (no
opinion). ``PipelineDecider`` chains the tiers and ALWAYS returns a decision:

1. built-in session checks — terminated session / revoked tool -> BLOCK
2. policy engine         — first matching rule; ``escalate`` rules go to the guard slot
3. (Phase 5: cache -> fast tier -> guard)
4. nobody had an opinion -> ALLOW, risk 0, ``tier=default``

Escalations never auto-allow by policy: they go to the guard; with no guard configured
(or the guard abstaining) the rule's fallback is used with ``tier=fallback``.
Errors/timeouts and fail modes are handled by the enforcer layer (step 11), not here.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from gateway.models import Action, Decision, Event, Tier
from gateway.policy import PolicyContext, PolicyEngine, build_context
from gateway.session import SessionTracker

RULE_TERMINATED = "builtin:session-terminated"
RULE_REVOKED = "builtin:tool-revoked"


@runtime_checkable
class Decider(Protocol):
    async def decide(self, event: Event, ctx: PolicyContext) -> Decision | None:
        """Return a decision for ``event`` or ``None`` when this tier has no opinion."""
        ...


class PipelineDecider:
    def __init__(
        self,
        tracker: SessionTracker,
        policy: PolicyEngine,
        *,
        guard: Decider | None = None,
    ) -> None:
        self.tracker = tracker
        self.policy = policy
        self.guard = guard

    @property
    def model_version(self) -> str:
        return self.policy.model_version

    def _builtin(self, action: Action, risk: float, reason: str, rule_id: str) -> Decision:
        return Decision(
            action=action,
            risk=risk,
            reason=reason,
            tier=Tier.POLICY,
            rule_id=rule_id,
            model_version=self.model_version,
        )

    async def decide(self, event: Event) -> Decision:
        t0 = time.perf_counter()
        decision = await self._decide(event)
        return decision.model_copy(update={"latency_ms": (time.perf_counter() - t0) * 1000})

    async def _decide(self, event: Event) -> Decision:
        sid, tool = event.session_id, event.qualified_tool

        # 1. built-in session state (applies to requests and outputs)
        if await self.tracker.is_terminated(sid):
            return self._builtin(
                Action.BLOCK, 1.0, "Session was terminated; no further tool use.", RULE_TERMINATED
            )
        if await self.tracker.is_revoked(sid, tool):
            return self._builtin(
                Action.BLOCK, 0.9, f"{tool} was revoked for this session.", RULE_REVOKED
            )

        # 2. policy
        ctx = await build_context(self.tracker, event)
        match = self.policy.evaluate(ctx)
        if match is not None and not match.escalate:
            return match.decision
        if match is not None:  # escalation: guard decides, else the rule's fallback
            if self.guard is not None:
                verdict = await self.guard.decide(event, ctx)
                if verdict is not None:
                    return verdict.model_copy(update={"rule_id": match.rule_id})
            return match.decision.model_copy(
                update={
                    "tier": Tier.FALLBACK,
                    "reason": f"{match.decision.reason} (escalated; no guard verdict)"[:500],
                }
            )

        # 3. Phase 5: cache -> fast tier -> guard go here.

        # 4. no opinion anywhere
        return Decision(
            action=Action.ALLOW,
            risk=0.0,
            reason="No rule or model objected.",
            tier=Tier.DEFAULT,
            model_version=self.model_version,
        )
