from datetime import datetime

import pytest
from pydantic import ValidationError

from gateway.models import (
    REASON_MAX_CHARS,
    Action,
    Decision,
    Direction,
    Event,
    EventKind,
    Tier,
    sha256_text,
)


def _req(**kw: object) -> Event:
    base: dict[str, object] = {
        "session_id": "s1",
        "direction": Direction.REQUEST,
        "server": "filesystem",
        "tool": "read_file",
        "args": {"path": "docs/readme.md"},
    }
    return Event.model_validate(base | kw)


def _out(text: str = "<html>hi</html>", max_chars: int = 1000, **kw: object) -> Event:
    return Event.from_output(
        session_id="s1",
        server="http_fetch",
        tool="fetch",
        full_text=text,
        max_chars=max_chars,
        **kw,  # type: ignore[arg-type]
    )


# --- enums -----------------------------------------------------------------


def test_action_space_is_exactly_six() -> None:
    assert [a.value for a in Action] == [
        "ALLOW",
        "BLOCK",
        "SANITIZE",
        "REQUIRE_APPROVAL",
        "REVOKE_TOOL",
        "TERMINATE_SESSION",
    ]


def test_event_kinds() -> None:
    assert [k.value for k in EventKind] == ["tool_call", "tool_description"]


# --- request events ---------------------------------------------------------


def test_request_event_defaults() -> None:
    e = _req()
    assert e.kind is EventKind.TOOL_CALL
    assert e.qualified_tool == "filesystem.read_file"
    assert len(e.event_id) == 32
    assert e.ts.tzinfo is not None
    assert _req().event_id != e.event_id
    assert e.output_truncated is False


@pytest.mark.parametrize(
    "kw",
    [
        {"args": None},
        {"output": "x"},
        {"output_sha256": "0" * 64},
        {"output_chars": 3},
        {"upstream_ms": 3.0},
        {"kind": EventKind.TOOL_DESCRIPTION},  # descriptions are outputs only
        {"ts": datetime(2026, 1, 1)},  # naive timestamp
        {"session_id": ""},
        {"unexpected": 1},
    ],
)
def test_invalid_request_events(kw: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _req(**kw)


def test_events_are_immutable() -> None:
    e = _req()
    with pytest.raises(ValidationError):
        e.tool = "write_file"  # type: ignore[misc]


# --- output events ----------------------------------------------------------


def test_output_event_untruncated() -> None:
    e = _out("hello", upstream_ms=12.5)
    assert e.output == "hello"
    assert e.output_chars == 5
    assert e.output_sha256 == sha256_text("hello")
    assert e.output_truncated is False
    assert e.args is None


def test_output_event_truncated_hashes_full_text() -> None:
    full = "A" * 300 + "SECRET-TAIL"
    e = _out(full, max_chars=300)
    assert e.output == "A" * 300
    assert e.output_chars == len(full)
    assert e.output_sha256 == sha256_text(full)
    assert e.output_truncated is True


def test_tool_description_event() -> None:
    e = _out("Reads a file.", kind=EventKind.TOOL_DESCRIPTION)
    assert e.kind is EventKind.TOOL_DESCRIPTION
    assert e.direction is Direction.OUTPUT


def test_output_roundtrip_json() -> None:
    e = _out("x" * 50, max_chars=10)
    assert Event.model_validate_json(e.model_dump_json()) == e


def test_from_output_rejects_nonpositive_max() -> None:
    with pytest.raises(ValueError):
        _out("x", max_chars=0)


@pytest.mark.parametrize(
    "kw",
    [
        {"output_sha256": None},
        {"output_chars": None},
        {"output_chars": 2},  # stored text longer than claimed full length
        {"output_sha256": "f" * 64},  # untruncated but hash mismatch
        {"output_sha256": "not-a-hash"},
        {"args": {}},
    ],
)
def test_invalid_output_events(kw: dict[str, object]) -> None:
    base = _out("hello").model_dump()
    with pytest.raises(ValidationError):
        Event.model_validate(base | kw)


# --- decisions --------------------------------------------------------------


def _decision(**kw: object) -> Decision:
    base: dict[str, object] = {
        "action": "ALLOW",
        "risk": 0.0,
        "reason": "ok",
        "tier": "policy",
        "model_version": "v0",
    }
    return Decision.model_validate(base | kw)


def test_decision_roundtrip_json() -> None:
    d = Decision(
        action=Action.BLOCK,
        risk=0.9,
        attack_family="direct_malicious",
        reason="Reads a private key outside the allowed paths.",
        tier=Tier.POLICY,
        latency_ms=0.4,
        model_version="policy-only-v0",
    )
    assert d.cached is False
    assert Decision.model_validate_json(d.model_dump_json()) == d
    assert d.model_dump(mode="json")["action"] == "BLOCK"


def test_cached_decision_keeps_original_tier() -> None:
    original = _decision(action="BLOCK", risk=0.8, tier="guard", latency_ms=420.0)
    hit = original.model_copy(update={"cached": True, "latency_ms": 0.3})
    assert hit.tier is Tier.GUARD
    assert hit.cached is True
    assert original.cached is False


@pytest.mark.parametrize(
    "kw",
    [
        {"risk": 1.5},
        {"risk": -0.1},
        {"reason": ""},
        {"reason": "x" * (REASON_MAX_CHARS + 1)},
        {"action": "DENY"},
        {"tier": "cache"},
    ],
)
def test_invalid_decisions(kw: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _decision(**kw)
