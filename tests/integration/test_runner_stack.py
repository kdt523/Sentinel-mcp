"""Step 5 on the compose stack: shell sessions are captured with per-session values, raw
capture runs only in sandbox mode, and the gateway is restored to enforcing afterwards."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from data.runner import (
    STACK_CAPTURE_ENV,
    _compose,
    _composites,
    build_samples,
    restart_stack_gateway,
    run_on_stack,
)
from data.sandbox_values import SessionValues
from data.schema import Label
from gateway.models import Action, Direction, EventKind

pytestmark = pytest.mark.integration

SHELL = [c for c in _composites(None, 42) if any(s.tool.startswith("shell.") for s in c.steps)]


@pytest.fixture(scope="module", autouse=True)
def capture_gateway() -> Iterator[None]:
    restart_stack_gateway(STACK_CAPTURE_ENV)
    yield
    restart_stack_gateway(None)
    env = _compose("exec", "-T", "gateway", "env")
    assert "SENTINEL_CAPTURE_MODE=false" in env and "SENTINEL_SHADOW_MODE=false" in env


def test_stack_gateway_captures_only_in_sandbox_mode() -> None:
    env = _compose("exec", "-T", "gateway", "env")
    assert "SENTINEL_CAPTURE_MODE=true" in env and "SENTINEL_SANDBOX_MODE=true" in env
    assert "servers.capture.yaml" in env


async def test_shell_key_read_sees_this_sessions_key_and_is_labelled() -> None:
    picks = [c for c in SHELL if c.units[c.focal_index].template_id == "dm_cat_key_shell"
             and c.units[c.focal_index].steps[0].args["cmd"].startswith("cat ")][:2]  # fmt: skip
    assert len(picks) == 2
    keys = []
    for c in picks:
        result = await run_on_stack(c)
        assert not result.step_errors
        samples = build_samples(result)
        mal = [s for s in samples if s.label is Label.MALICIOUS]
        assert [(m.event.tool, m.gold_action) for m in mal] == [("run", Action.BLOCK)]
        out = next(s for s in samples if s.step_idx > mal[0].step_idx
                   and s.event.direction is Direction.OUTPUT and s.event.tool == "run"
                   and s.event.kind is EventKind.TOOL_CALL)  # fmt: skip
        body = SessionValues.generate(c.values_seed).ssh_key_body
        assert body[:64] in (out.event.output or "")
        keys.append(body)
    assert keys[0] != keys[1]


async def test_benign_shell_sessions_capture_cleanly() -> None:
    for c in [c for c in SHELL if c.kind == "benign"][:3]:
        result = await run_on_stack(c)
        assert not result.step_errors
        samples = build_samples(result)
        assert samples and all(s.label is Label.BENIGN for s in samples)
        assert not any(s.post_termination for s in samples)
