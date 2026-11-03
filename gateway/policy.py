"""YAML policy engine: hard rules evaluated in order, first match wins.

The engine is pure and synchronous: it decides from a precomputed ``PolicyContext``
(the event plus the session state *before* this event, plus which taint classes flow into
this event's arguments). ``build_context`` gathers that from the session tracker.

Outcomes:
- ``action: <ACTION>``  — a final decision (tier=policy).
- ``escalate: true``    — never auto-allow; route to the reasoning guard. Until a guard exists
  the decider uses ``fallback`` (default REQUIRE_APPROVAL).
No matching rule returns ``None``: the policy has no opinion.

CLI:
    python -m gateway.policy validate [--policy policies/default.yaml]
    python -m gateway.policy dry-run --scenario policies/scenarios/<name>.jsonl [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import re
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from gateway.config import FailMode
from gateway.detectors import (
    egress_targets,
    fingerprints_in_args,
    iter_strings,
    paths_in_args,
    redact_secrets,
)
from gateway.models import Action, Decision, Direction, Event, Tier
from gateway.session import SessionState, SessionTracker

PLACEHOLDERS = frozenset({"tool", "path", "target", "classes", "count", "limit", "risk", "match"})
TaintClass = Literal["secret", "pii"]


class PolicyError(ValueError):
    """Policy file could not be loaded or failed validation."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ----------------------------------------------------------------------------- schema


class Outcome(_Strict):
    action: Action | None = None
    escalate: bool = False
    fallback: Action = Action.REQUIRE_APPROVAL

    @model_validator(mode="after")
    def _one_outcome(self) -> Self:
        if (self.action is None) == (not self.escalate):
            raise ValueError("outcome needs exactly one of `action` or `escalate: true`")
        if self.escalate and self.fallback is Action.ALLOW:
            raise ValueError("escalate fallback must not be ALLOW (escalations never auto-allow)")
        return self


class Match(_Strict):
    tools: list[str] = Field(default_factory=lambda: ["*"], min_length=1)
    direction: Direction = Direction.REQUEST


def _compile(patterns: list[str], ignore_case: bool) -> list[re.Pattern[str]]:
    flags = re.IGNORECASE if ignore_case else 0
    out = []
    for p in patterns:
        try:
            out.append(re.compile(p, flags))
        except re.error as exc:
            raise ValueError(f"invalid regex {p!r}: {exc}") from exc
    return out


def _glob_any(value: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(value, g) for g in globs)


@dataclass(frozen=True)
class PolicyContext:
    event: Event
    state: SessionState
    paths: tuple[str, ...] = ()
    egress: frozenset[str] = frozenset()
    flow_classes: frozenset[str] = frozenset()  # taint classes whose fingerprints are in args


class _RuleBase(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$", max_length=64)
    description: str = ""
    match: Match = Field(default_factory=Match)
    outcome: Outcome
    risk: float = Field(ge=0.0, le=1.0)
    attack_family: str | None = None
    reason: str = Field(min_length=1, max_length=300)

    @field_validator("reason")
    @classmethod
    def _known_placeholders(cls, v: str) -> str:
        used = {name for _, name, _, _ in string.Formatter().parse(v) if name}
        unknown = used - PLACEHOLDERS
        if unknown:
            raise ValueError(
                f"unknown reason placeholders {sorted(unknown)}; allowed {sorted(PLACEHOLDERS)}"
            )
        return v

    def applies(self, event: Event) -> bool:
        return event.direction is self.match.direction and _glob_any(
            event.qualified_tool, self.match.tools
        )

    def check(self, ctx: PolicyContext) -> dict[str, str] | None:  # pragma: no cover
        raise NotImplementedError


class ToolMatchRule(_RuleBase):
    """Fires on every matching event (e.g. tools that always need approval)."""

    type: Literal["tool_match"]

    def check(self, ctx: PolicyContext) -> dict[str, str] | None:
        return {}


class DenyPathsRule(_RuleBase):
    type: Literal["deny_paths"]
    patterns: list[str] = Field(min_length=1)
    ignore_case: bool = True
    _rx: list[re.Pattern[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self._rx = _compile(self.patterns, self.ignore_case)

    def check(self, ctx: PolicyContext) -> dict[str, str] | None:
        for path in ctx.paths:
            norm = path.replace("\\", "/")
            if any(rx.search(norm) for rx in self._rx):
                return {"path": path[:120]}
        return None


class DenyArgsRegexRule(_RuleBase):
    type: Literal["deny_args_regex"]
    arg: str | None = None  # None = any string anywhere in the arguments
    patterns: list[str] = Field(min_length=1)
    ignore_case: bool = True
    _rx: list[re.Pattern[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self._rx = _compile(self.patterns, self.ignore_case)

    def check(self, ctx: PolicyContext) -> dict[str, str] | None:
        args = ctx.event.args or {}
        values = [args.get(self.arg)] if self.arg is not None else list(iter_strings(args))
        for value in values:
            if not isinstance(value, str):
                continue
            for rx in self._rx:
                m = rx.search(value)
                if m:
                    return {"match": redact_secrets(m.group(0).strip())[:80]}
        return None


class EgressAllowlistRule(_RuleBase):
    """Fires when the event targets a destination outside ``allowed`` (globs)."""

    type: Literal["egress_allowlist"]
    allowed: list[str]
    arg: str | None = None  # restrict target extraction to one argument (e.g. "to", "url")

    def check(self, ctx: PolicyContext) -> dict[str, str] | None:
        args = ctx.event.args or {}
        if self.arg is not None:
            targets = egress_targets({self.arg: args.get(self.arg)})
        else:
            targets = set(ctx.egress)
        for t in sorted(targets):
            if not _glob_any(t, self.allowed):
                return {"target": t}
        return None


class TaintedFlowRule(_RuleBase):
    """Fires when fingerprints of previously read secret/PII values appear in the arguments.

    ``unless_egress_within`` exempts calls whose DESTINATIONS (only the ``destination_args``,
    e.g. ``to`` / ``url`` — not addresses merely mentioned in a body) are all allowlisted.
    """

    type: Literal["tainted_flow"]
    classes: list[TaintClass] = Field(min_length=1)
    unless_egress_within: list[str] = Field(default_factory=list)
    destination_args: list[str] = Field(
        default_factory=lambda: ["to", "cc", "bcc", "recipient", "recipients", "url"]
    )

    def check(self, ctx: PolicyContext) -> dict[str, str] | None:
        hit = sorted(set(self.classes) & ctx.flow_classes)
        if not hit:
            return None
        if self.unless_egress_within:
            args = ctx.event.args or {}
            dests = egress_targets({a: args[a] for a in self.destination_args if a in args})
            if dests and all(_glob_any(t, self.unless_egress_within) for t in dests):
                return None
        return {"classes": ", ".join(hit)}


class MaxCallsRule(_RuleBase):
    type: Literal["max_calls_per_session"]
    limit: int = Field(ge=1)

    def check(self, ctx: PolicyContext) -> dict[str, str] | None:
        if ctx.state.request_count >= self.limit:
            return {"count": str(ctx.state.request_count), "limit": str(self.limit)}
        return None


class SessionRiskRule(_RuleBase):
    type: Literal["session_risk_above"]
    threshold: float = Field(gt=0.0)

    def check(self, ctx: PolicyContext) -> dict[str, str] | None:
        if ctx.state.risk >= self.threshold:
            return {"risk": f"{ctx.state.risk:.2f}"}
        return None


Rule = Annotated[
    ToolMatchRule
    | DenyPathsRule
    | DenyArgsRegexRule
    | EgressAllowlistRule
    | TaintedFlowRule
    | MaxCallsRule
    | SessionRiskRule,
    Field(discriminator="type"),
]


class ToolSettings(_Strict):
    fail_mode: FailMode | None = None


class PolicyFile(_Strict):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    revision: str = Field(min_length=1, max_length=40)
    tools: dict[str, ToolSettings] = Field(default_factory=dict)  # glob -> settings, first wins
    rules: list[Rule]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        seen: set[str] = set()
        for r in self.rules:
            if r.id in seen:
                raise ValueError(f"duplicate rule id: {r.id}")
            seen.add(r.id)
        return self


# ----------------------------------------------------------------------------- engine


class PolicyMatch(_Strict):
    rule_id: str
    escalate: bool
    decision: Decision  # final decision, or the fallback when escalate=True


class _SafeFormat(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "?"


class PolicyEngine:
    def __init__(self, policy: PolicyFile) -> None:
        self.policy = policy

    @classmethod
    def load(cls, path: Path) -> PolicyEngine:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise PolicyError(f"cannot read policy {path}: {exc}") from exc
        try:
            return cls(PolicyFile.model_validate(raw))
        except ValidationError as exc:
            raise PolicyError(f"invalid policy {path}:\n{exc}") from exc

    @property
    def model_version(self) -> str:
        return f"policy:{self.policy.name}@{self.policy.revision}"

    def fail_mode_for(self, qualified_tool: str) -> FailMode | None:
        for pattern, settings in self.policy.tools.items():
            if fnmatch.fnmatchcase(qualified_tool, pattern):
                return settings.fail_mode
        return None

    def evaluate(self, ctx: PolicyContext) -> PolicyMatch | None:
        t0 = time.perf_counter()
        for rule in self.policy.rules:
            if not rule.applies(ctx.event):
                continue
            values = rule.check(ctx)
            if values is None:
                continue
            fmt = _SafeFormat(tool=ctx.event.qualified_tool, **values)
            action = rule.outcome.fallback if rule.outcome.escalate else rule.outcome.action
            assert action is not None  # guaranteed by Outcome validation
            decision = Decision(
                action=action,
                risk=rule.risk,
                attack_family=rule.attack_family,
                reason=rule.reason.format_map(fmt)[:500],
                tier=Tier.POLICY,
                latency_ms=(time.perf_counter() - t0) * 1000,
                model_version=self.model_version,
            )
            return PolicyMatch(rule_id=rule.id, escalate=rule.outcome.escalate, decision=decision)
        return None


async def build_context(tracker: SessionTracker, event: Event) -> PolicyContext:
    """Session state BEFORE this event + taint classes flowing into its arguments."""
    state = await tracker.get(event.session_id)
    if event.direction is not Direction.REQUEST or event.args is None:
        return PolicyContext(event=event, state=state)
    flow = await tracker.match_fingerprints(event.session_id, fingerprints_in_args(event.args))
    return PolicyContext(
        event=event,
        state=state,
        paths=tuple(paths_in_args(event.args)),
        egress=frozenset(egress_targets(event.args)),
        flow_classes=frozenset(flow),
    )


# ----------------------------------------------------------------------------- CLI

DEFAULT_POLICY = Path("policies/default.yaml")


def _scenario_events(path: Path, session_id: str, max_chars: int) -> list[Event]:
    events = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        spec = json.loads(line)
        server, _, tool = spec["tool"].partition(".")
        if not tool:
            raise PolicyError(f"{path}:{n}: tool must be '<server>.<tool>'")
        if spec["direction"] == "request":
            events.append(
                Event(
                    session_id=session_id,
                    direction=Direction.REQUEST,
                    server=server,
                    tool=tool,
                    args=spec.get("args", {}),
                )  # fmt: skip
            )
        else:
            text = spec.get("output")
            if text is None:
                text = Path(spec["output_file"]).read_text(encoding="utf-8")
            events.append(
                Event.from_output(
                    session_id=session_id,
                    server=server,
                    tool=tool,
                    full_text=text,
                    max_chars=max_chars,
                )  # fmt: skip
            )
    return events


async def dry_run(engine: PolicyEngine, scenario: Path) -> list[dict[str, Any]]:
    """Replay a scenario through the policy with an in-memory session (no Redis needed)."""
    import fakeredis

    redis = fakeredis.FakeAsyncRedis()
    tracker = SessionTracker(redis)
    sid = f"dry-run:{scenario.stem}"
    rows: list[dict[str, Any]] = []
    last_request: dict[str, Event] = {}
    try:
        for i, event in enumerate(_scenario_events(scenario, sid, 16_000), 1):
            if await tracker.is_terminated(sid):
                rows.append({"step": i, "direction": event.direction.value,
                             "tool": event.qualified_tool, "action": "TERMINATED",
                             "rule": None, "escalate": False, "risk": None,
                             "reason": "session already terminated"})  # fmt: skip
                continue
            match = engine.evaluate(await build_context(tracker, event))
            decision = match.decision if match else Decision(
                action=Action.ALLOW, risk=0.0, reason="no policy opinion", tier=Tier.POLICY,
                model_version=engine.model_version,
            )  # fmt: skip
            await tracker.record(event, decision, request=last_request.get(event.qualified_tool))
            if event.direction is Direction.REQUEST:
                last_request[event.qualified_tool] = event
            if decision.action is Action.TERMINATE_SESSION:
                await tracker.terminate(sid, decision.reason)
            elif decision.action is Action.REVOKE_TOOL:
                await tracker.revoke_tool(sid, event.qualified_tool)
            rows.append({
                "step": i, "direction": event.direction.value, "tool": event.qualified_tool,
                "action": decision.action.value if match else "-",
                "rule": match.rule_id if match else None,
                "escalate": bool(match and match.escalate),
                "risk": decision.risk, "reason": decision.reason if match else "",
            })  # fmt: skip
    finally:
        await redis.aclose()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gateway.policy")
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="load and validate a policy file")
    v.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    d = sub.add_parser("dry-run", help="replay a JSONL scenario through the policy")
    d.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    d.add_argument("--scenario", type=Path, required=True)
    d.add_argument("--json", action="store_true", help="emit one JSON object per step")
    args = parser.parse_args(argv)

    try:
        engine = PolicyEngine.load(args.policy)
    except PolicyError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.cmd == "validate":
        print(f"OK {engine.model_version}: {len(engine.policy.rules)} rules")
        return 0

    rows = asyncio.run(dry_run(engine, args.scenario))
    for r in rows:
        if args.json:
            print(json.dumps(r))
        else:
            tag = " (escalate)" if r["escalate"] else ""
            print(f"{r['step']:>3} {r['direction']:<7} {r['tool']:<26} {r['action']:<18}"
                  f"{(r['rule'] or ''):<24}{r['reason']}{tag}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
