"""Enforcement of decisions, fail modes and human approvals.

``decide_with_fail_mode`` wraps the decider: any exception or timeout becomes a
``tier=fallback`` decision following the tool's fail mode (closed=BLOCK, open=ALLOW),
logged loudly.

``Enforcer.enforce`` applies a decision:

=================  ===========================================================
ALLOW              forward
BLOCK              do not forward; MCP error carrying the reason
SANITIZE           outputs: strip injected spans and forward; requests: BLOCK
REVOKE_TOOL        revoke the tool for the session, then block
TERMINATE_SESSION  mark the session terminated, then block
REQUIRE_APPROVAL   hold the call open until a human approves/denies; a timeout is
                   always a DENY (status ``expired``), whatever the fail mode
=================  ===========================================================

Shadow mode decides and logs but always forwards (no side effects, no approvals).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import uuid4

import anyio
import structlog

from gateway.approvals import Approval, ApprovalStore
from gateway.config import FailMode
from gateway.decider import PipelineDecider
from gateway.detectors import redact_deep
from gateway.models import Action, ApprovalStatus, Decision, Direction, Event, Tier
from gateway.recorder import DecisionRecorder
from gateway.sanitizer import sanitize
from gateway.session import SessionTracker

log = structlog.get_logger("sentinel.enforcer")

# progress(elapsed_s, total_s, message) -> sends an MCP progress notification if possible
ProgressFn = Callable[[float, float | None, str], Awaitable[None]]


class ApprovalCancelledError(Exception):
    """The agent went away while its call was waiting for a human decision."""


@dataclass(frozen=True)
class EnforcementResult:
    forward: bool  # call the upstream tool / return the output to the agent
    enforced: bool  # False in shadow mode
    output: str | None = None  # (possibly sanitized) output to return, for output events
    message: str | None = None  # error text for the agent when not forwarded
    approval: Approval | None = None
    sanitized: tuple[str, ...] = ()  # detector labels of removed spans


# ------------------------------------------------------------------------- fail modes


async def decide_with_fail_mode(
    decider: PipelineDecider,
    event: Event,
    *,
    fail_mode: FailMode,
    timeout_s: float,
) -> Decision:
    """Never raises (except cancellation): errors/timeouts follow the tool's fail mode."""
    try:
        return await asyncio.wait_for(decider.decide(event), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 - includes TimeoutError, redis, httpx2, bugs
        closed = fail_mode == "closed"
        log.error(
            "decision_failed_fail_mode_applied",
            fail_mode=fail_mode,
            tool=event.qualified_tool,
            session_id=event.session_id,
            event_id=event.event_id,
            error=f"{type(exc).__name__}: {exc}"[:300],
        )
        return Decision(
            action=Action.BLOCK if closed else Action.ALLOW,
            risk=1.0 if closed else 0.0,
            reason=f"Decision failed ({type(exc).__name__}); applied fail-{fail_mode}.",
            tier=Tier.FALLBACK,
            rule_id=f"builtin:fail-{fail_mode}",
            model_version=decider.model_version,
        )


# ------------------------------------------------------------------------- enforcer


# One distinct prefix per enforcement outcome, so clients (and the scripted agent's
# `expect` checks) can tell outcomes apart from the result text alone.
MSG_BLOCKED = "SentinelMCP blocked this call: "
MSG_REVOKED = "SentinelMCP revoked this tool for the session: "
MSG_TERMINATED = "SentinelMCP terminated this session: "
MSG_DENIED = "SentinelMCP denied this call: "  # approval rejected, expired or not requestable


def _block_message(decision: Decision) -> str:
    return f"{MSG_BLOCKED}{decision.reason}"


class Enforcer:
    def __init__(
        self,
        tracker: SessionTracker,
        approvals: ApprovalStore,
        *,
        shadow_mode: bool,
        recorder: DecisionRecorder | None = None,
        approval_timeout_s: float = 50.0,
        poll_interval_s: float = 0.5,
        progress_interval_s: float = 5.0,
    ) -> None:
        self.tracker = tracker
        self.approvals = approvals
        self.shadow_mode = shadow_mode
        self.recorder = recorder
        self.approval_timeout_s = approval_timeout_s
        self.poll_interval_s = poll_interval_s
        self.progress_interval_s = progress_interval_s

    async def enforce(
        self,
        event: Event,
        decision: Decision,
        *,
        progress: ProgressFn | None = None,
        full_output: str | None = None,
    ) -> EnforcementResult:
        """``full_output``: the untruncated output text (events store a truncated copy);
        SANITIZE rewrites and returns the full text."""
        action = decision.action
        if self.shadow_mode:
            if action is not Action.ALLOW:
                log.info(
                    "shadow_mode_would_enforce",
                    action=action.value,
                    tool=event.qualified_tool,
                    session_id=event.session_id,
                    rule_id=decision.rule_id,
                )
            return EnforcementResult(forward=True, enforced=False, output=event.output)

        if action is Action.ALLOW:
            return EnforcementResult(forward=True, enforced=True, output=event.output)
        if action is Action.SANITIZE:
            if event.direction is Direction.OUTPUT and event.output is not None:
                res = sanitize(full_output if full_output is not None else event.output)
                return EnforcementResult(forward=True, enforced=True, output=res.text,
                                         sanitized=tuple(res.removed))  # fmt: skip
            log.warning("sanitize_on_request_blocked", tool=event.qualified_tool)
            return EnforcementResult(forward=False, enforced=True, message=_block_message(decision))
        if action is Action.REVOKE_TOOL:
            await self.tracker.revoke_tool(event.session_id, event.qualified_tool)
            msg = f"{MSG_REVOKED}{decision.reason}"
            return EnforcementResult(forward=False, enforced=True, message=msg)
        if action is Action.TERMINATE_SESSION:
            await self.tracker.terminate(event.session_id, decision.reason)
            msg = f"{MSG_TERMINATED}{decision.reason}"
            return EnforcementResult(forward=False, enforced=True, message=msg)
        if action is Action.REQUIRE_APPROVAL:
            return await self._require_approval(event, decision, progress)
        return EnforcementResult(forward=False, enforced=True, message=_block_message(decision))

    # ------------------------------------------------------------------ approvals

    def _log_approval(self, approval: Approval | None) -> None:
        if approval is not None and self.recorder is not None:
            self.recorder.submit_approval(approval.db_row())

    async def _require_approval(
        self, event: Event, decision: Decision, progress: ProgressFn | None
    ) -> EnforcementResult:
        preview = json.dumps(redact_deep(event.args if event.args is not None else event.output))
        # Pre-allocate the id: if we are cancelled while create() is in flight, the
        # approval may already be stored, and the handler below must still close it.
        approval_id = uuid4().hex
        try:
            try:
                approval = await self.approvals.create(
                    event, decision, preview=preview, approval_id=approval_id
                )
            except Exception as exc:  # noqa: BLE001 - cannot ask a human -> deny
                log.error("approval_create_failed", error=f"{type(exc).__name__}: {exc}"[:300])
                return EnforcementResult(
                    forward=False, enforced=True,
                    message=f"{MSG_DENIED}approval could not be requested "
                    f"({type(exc).__name__}).",
                )  # fmt: skip
            self._log_approval(approval)
            log.info("approval_requested", approval_id=approval_id,
                     tool=event.qualified_tool, session_id=event.session_id)  # fmt: skip
            status = await self._wait(approval, progress)
        except BaseException:
            # Agent disconnected / request cancelled: record it, then propagate.
            with anyio.CancelScope(shield=True):
                await self._finish(approval_id, ApprovalStatus.CANCELLED,
                                   note="agent disconnected while waiting")  # fmt: skip
            raise

        final = await self.approvals.get(approval.approval_id) or approval
        if status is ApprovalStatus.APPROVED:
            return EnforcementResult(forward=True, enforced=True, output=event.output,
                                     approval=final)  # fmt: skip
        if status is ApprovalStatus.EXPIRED:
            msg = (
                f"{MSG_DENIED}approval request timed out after "
                f"{self.approval_timeout_s:g}s with no decision (treated as denied). "
                f"Original reason: {decision.reason}"
            )
        else:
            who = f" by {final.resolved_by}" if final.resolved_by else ""
            note = f" ({final.note})" if final.note else ""
            msg = f"{MSG_DENIED}rejected{who}{note}. Reason: {decision.reason}"
        return EnforcementResult(forward=False, enforced=True, message=msg, approval=final)

    async def _wait(self, approval: Approval, progress: ProgressFn | None) -> ApprovalStatus:
        start = time.monotonic()
        next_progress = 0.0
        while True:
            status = await self.approvals.status(approval.approval_id)
            if status is not None and status is not ApprovalStatus.PENDING:
                self._log_approval(await self.approvals.get(approval.approval_id))
                return status
            elapsed = time.monotonic() - start
            if elapsed >= self.approval_timeout_s:
                return await self._finish(
                    approval.approval_id, ApprovalStatus.EXPIRED, note="no decision within timeout"
                )
            if progress is not None and elapsed >= next_progress:
                next_progress = elapsed + self.progress_interval_s
                try:
                    await progress(
                        elapsed, self.approval_timeout_s,
                        f"Waiting for human approval (id {approval.approval_id[:8]}, "
                        f"{self.approval_timeout_s - elapsed:.0f}s left)",
                    )  # fmt: skip
                except Exception as exc:  # noqa: BLE001 - progress is best effort
                    log.warning("approval_progress_failed", error=type(exc).__name__)
            await asyncio.sleep(min(self.poll_interval_s, self.approval_timeout_s - elapsed))

    async def _finish(self, approval_id: str, to: ApprovalStatus, *, note: str) -> ApprovalStatus:
        """Gateway-side transition (expire/cancel). A human decision that won the race stands."""
        try:
            res = await self.approvals.transition(approval_id, to, by="sentinel", note=note)
        except Exception as exc:  # noqa: BLE001
            log.error("approval_transition_failed", approval_id=approval_id, to=to.value,
                      error=f"{type(exc).__name__}: {exc}"[:300])  # fmt: skip
            return to
        final = res.status if res.status is not None else to
        log.info("approval_finished", approval_id=approval_id, status=final.value,
                 by_gateway=res.ok)  # fmt: skip
        self._log_approval(await self.approvals.get(approval_id))
        return final
