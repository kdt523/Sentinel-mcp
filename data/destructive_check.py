"""Load-time guard: gold can never be milder than an obvious destructive pattern.

Any scripted step whose request matches one of the FROZEN policy's stateless destructive
rules (``attack_family: destructive``, e.g. ``destructive-sql``) must carry a destructive tag
(primary family or ``also``), so most-severe-wins gives it at least the destructive gold.
Benign steps must never match. (The frozen policy has no destructive shell rule: the shell
allowlist contains no deleting program; such a rule would be picked up automatically.)
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from gateway.detectors import paths_in_args
from gateway.models import Direction, Event
from gateway.policy import PolicyContext, PolicyEngine, RepeatOfRule
from gateway.session import SessionState

POLICY_PATH = Path(__file__).resolve().parents[1] / "policies" / "default.yaml"
DESTRUCTIVE = "destructive"


@cache
def _rules() -> tuple[Any, ...]:
    engine = PolicyEngine.load(POLICY_PATH)
    return tuple(
        r for r in engine.policy.rules
        if r.attack_family == DESTRUCTIVE and not isinstance(r, RepeatOfRule)
        and r.match.direction is Direction.REQUEST
    )  # fmt: skip


def destructive_rule(tool: str, args: dict[str, Any]) -> str | None:
    """Id of the frozen destructive rule this request matches, if any."""
    server, _, name = tool.partition(".")
    if not name:
        return None
    ev = Event(session_id="check", direction=Direction.REQUEST, server=server, tool=name,
               args=args)  # fmt: skip
    ctx = PolicyContext(event=ev, state=SessionState(session_id="check"),
                        paths=tuple(paths_in_args(args)))  # fmt: skip
    for rule in _rules():
        if rule.applies(ev) and rule.check(ctx) is not None:
            return str(rule.id)
    return None


def undertagged(step: Any) -> str | None:
    """Problem description if ``step`` matches a destructive rule without a destructive tag."""
    if getattr(step, "op", "call") != "call":
        return None
    rule = destructive_rule(step.tool, step.args)
    if rule is None:
        return None
    mark = step.attack
    families = set() if mark is None else {mark.family, *mark.also}
    if DESTRUCTIVE in families:
        return None
    args = str(step.args)[:80]
    return f"{step.tool} {args} matches frozen rule {rule!r} but is not tagged destructive"
