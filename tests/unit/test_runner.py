"""Step 5 runner: per-session sandbox values, capture only in sandbox mode, alignment of
captured events with scripted steps, event-level labels, post_termination, clean weather
for non-poisoning sessions, neutralised injections, and gateway decisions never as labels."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from data.attacks.templates import assign_attack_splits, compose_attacks
from data.attacks.templates import expand_all as expand_attacks
from data.benign.templates import expand_all as expand_benign
from data.composer import Composite, assign_splits, compose_benign, units_from_sessions
from data.runner import (
    POLICY_VERSION,
    CaptureError,
    SessionResult,
    build_samples,
    prepare_session_sandbox,
    run_inprocess,
    weather_config,
)
from data.sandbox_values import SessionValues
from data.schema import AttackFamily, Label
from gateway.models import Action, Direction, EventKind
from tools_servers.poisoned_desc.server import FORECAST_CLEAN

SEEDS = expand_attacks(seed=42)
SPLITS = assign_attack_splits(SEEDS, seed=42)
BENIGN = units_from_sessions(expand_benign(seed=42))
BSPLITS = assign_splits(BENIGN, seed=42)
ATTACKS = compose_attacks(SEEDS, SPLITS, BENIGN, BSPLITS, seed=42)
BENIGN_C = compose_benign(BENIGN, BSPLITS, n=40, seed=42)


def _local(c: Composite) -> bool:
    return not any(st.tool.startswith("shell.") for st in c.steps)


def _attack(template_id: str, i: int = 0) -> Composite:
    found = [c for c in ATTACKS if c.units[c.focal_index].template_id == template_id and _local(c)]
    return found[i]


# --- sandbox per session ------------------------------------------------------------------------


def test_each_session_gets_its_own_secrets_and_customers(tmp_path: Path) -> None:
    a, b = SessionValues.generate("a"), SessionValues.generate("b")
    prepare_session_sandbox(tmp_path / "a" / "fs", tmp_path / "a" / "app.db", a)
    prepare_session_sandbox(tmp_path / "b" / "fs", tmp_path / "b" / "app.db", b)
    env_a = (tmp_path / "a" / "fs" / ".env").read_text("utf-8")
    env_b = (tmp_path / "b" / "fs" / ".env").read_text("utf-8")
    assert a.api_key in env_a and b.api_key in env_b and env_a != env_b
    with sqlite3.connect(tmp_path / "a" / "app.db") as conn:
        emails = {r[0] for r in conn.execute("SELECT email FROM customers")}
        order_ids = {r[0] for r in conn.execute("SELECT customer_id FROM orders")}
    conn.close()
    assert emails == {c.email for c in a.customers}
    assert order_ids and order_ids <= {c.customer_id for c in a.customers}


# --- capture only in sandbox mode ---------------------------------------------------------------


def test_raw_capture_refused_outside_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.config import Settings

    monkeypatch.delenv("SANDBOX_IN_CONTAINER", raising=False)
    with pytest.raises(ValueError, match="only allowed in the sandbox"):
        Settings(capture_mode=True, sandbox_mode=False)


async def test_runner_gateway_runs_in_sandbox_mode(tmp_path: Path) -> None:
    from data.runner import capturing_gateway

    (tmp_path / "fs").mkdir()
    prepare_session_sandbox(tmp_path / "fs", tmp_path / "db" / "app.db", SessionValues.generate(1))
    # Settings(capture_mode=True) would raise without sandbox_mode (test above); the recorder
    # creates the capture directory at startup only when capture is on.
    async with capturing_gateway(tmp_path, "clean", None):
        assert (tmp_path / "raw").is_dir()


# --- end-to-end in-process capture --------------------------------------------------------------


async def test_exfil_session_labels_post_termination_and_own_values(tmp_path: Path) -> None:
    c = _attack("ms_env_by_email")
    result = await run_inprocess(c, tmp_path / "s")
    assert not result.step_errors
    samples = build_samples(result)
    values = SessionValues.generate(c.values_seed)
    mal = [s for s in samples if s.label is Label.MALICIOUS]
    assert [(m.event.server, m.event.tool, m.gold_action) for m in mal] == [
        ("email_send", "send", Action.TERMINATE_SESSION)
    ]
    term = mal[0].step_idx
    assert all(s.post_termination == (s.step_idx > term) for s in samples)
    assert mal[0].context.taint_classes == ["secret"]  # state BEFORE the leaking request
    assert values.api_key in str(mal[0].event.args)  # this session's own fake secret
    read = next(s for s in samples if s.event.direction is Direction.OUTPUT
                and s.event.kind is EventKind.TOOL_CALL and s.event.tool == "read_file"
                and "FAKE_API_KEY" in (s.event.output or ""))  # fmt: skip
    assert values.api_key in (read.event.output or "")
    assert all(s.session_family is AttackFamily.MULTI_STEP_EXFIL for s in samples)
    assert all(s.split is c.split and s.source == "generated" for s in samples)


async def test_rugpull_labels_only_the_changed_description(tmp_path: Path) -> None:
    c = _attack("tp_rugpull")
    samples = build_samples(await run_inprocess(c, tmp_path / "s"))
    forecast = [s for s in samples if s.event.kind is EventKind.TOOL_DESCRIPTION
                and s.event.server == "weather" and s.event.tool == "get_forecast"]  # fmt: skip
    assert forecast[0].label is Label.BENIGN  # first listing is clean
    changed = [s for s in forecast if s.event.description_changed]
    assert changed and all(s.gold_action is Action.REVOKE_TOOL for s in changed)
    assert all(s.attack_family is AttackFamily.TOOL_POISONING for s in changed)


async def test_static_poisoned_description_sanitize(tmp_path: Path) -> None:
    c = _attack("tp_poisoned_description")
    assert weather_config(c)[0] == "static"
    samples = build_samples(await run_inprocess(c, tmp_path / "s"))
    poisoned = [s for s in samples if s.label is Label.MALICIOUS]
    assert poisoned and all(s.gold_action is Action.SANITIZE for s in poisoned)


async def test_benign_sessions_never_see_a_poisoned_description(tmp_path: Path) -> None:
    for i, c in enumerate([c for c in BENIGN_C if _local(c)][:4]):
        assert weather_config(c) == ("clean", None)
        samples = build_samples(await run_inprocess(c, tmp_path / f"s{i}"))
        assert all(s.label is Label.BENIGN and s.session_family is None for s in samples)
        for s in samples:
            if s.event.kind is EventKind.TOOL_DESCRIPTION and s.event.tool == "get_forecast":
                assert s.event.output == FORECAST_CLEAN


async def test_injection_stripped_by_the_tool_is_benign_and_neutralised(tmp_path: Path) -> None:
    from data.attacks.templates import resolve_content
    from data.composer import Unit

    seed = next(u for u in SEEDS if u.template_id == "ii_page_ignored")
    # Force a comment-borne injection fetched in text mode (http_fetch drops HTML comments).
    unit = resolve_content(seed, SPLITS[seed.seed_id])
    url = "http://mockweb/g/train.poison.drop_orders.html_comment.3"
    steps = tuple(replace(st, args={"url": url, "raw": False}) for st in unit.steps)
    unit = replace(unit, steps=steps)
    c = Composite("k", SPLITS[seed.seed_id], "attack", (unit,), 0, values_seed="k")
    samples = build_samples(await run_inprocess(c, tmp_path / "s"))
    out = next(s for s in samples if s.event.tool == "fetch" and s.event.kind is
               EventKind.TOOL_CALL and s.event.direction is Direction.OUTPUT)  # fmt: skip
    assert out.neutralised_by_tool and out.label is Label.BENIGN
    assert out.hiding_technique == "html_comment"  # ground truth from the generator
    assert out.gateway_decision is not None
    assert out.gateway_decision.policy_version == POLICY_VERSION
    raw = replace(c, units=(replace(unit, steps=tuple(
        replace(st, args={"url": url, "raw": True}) for st in steps)),))  # fmt: skip
    samples = build_samples(await run_inprocess(raw, tmp_path / "r"))
    out = next(s for s in samples if s.event.tool == "fetch" and s.event.kind is
               EventKind.TOOL_CALL and s.event.direction is Direction.OUTPUT)  # fmt: skip
    assert not out.neutralised_by_tool and out.gold_action is Action.SANITIZE
    assert isinstance(unit, Unit)


async def test_labels_ignore_the_gateway_decision(tmp_path: Path) -> None:
    c = _attack("ds_drop_table")
    result = await run_inprocess(c, tmp_path / "s")
    allow_all = [replace(r, decision=r.decision.model_copy(update={"action": Action.ALLOW}))
                 for r in result.records]  # fmt: skip
    a = build_samples(result)
    b = build_samples(SessionResult(c, allow_all))
    assert [s.gold_action for s in a] == [s.gold_action for s in b]
    mal = [s for s in b if s.label is Label.MALICIOUS]
    assert mal and all(s.policy_disagrees for s in mal)


async def test_misaligned_capture_is_an_error(tmp_path: Path) -> None:
    c = _attack("ds_delete_all")
    result = await run_inprocess(c, tmp_path / "s")
    with pytest.raises(CaptureError):
        first_request = next(i for i, r in enumerate(result.records)
                             if r.event.direction is Direction.REQUEST)  # fmt: skip
        build_samples(SessionResult(c, [r for i, r in enumerate(result.records)
                                        if i != first_request]))  # fmt: skip
    other = _attack("ds_drop_table")
    with pytest.raises(CaptureError):
        build_samples(SessionResult(other, result.records))


# --- build_id -----------------------------------------------------------------------------------


async def test_every_sample_carries_the_build_id(tmp_path: Path) -> None:
    from data.build_id import current_build_id

    samples = build_samples(await run_inprocess(_attack("ds_drop_table"), tmp_path / "s"))
    assert {s.build_id for s in samples} == {current_build_id()}


def test_build_id_changes_with_composition_and_mixed_ids_are_refused() -> None:
    import data.build_id as b

    assert b.current_build_id().endswith("+" + b.composition_version())
    assert b.require_single_build(["abc+123", "abc+123"]) == "abc+123"
    for bad in (["a+1", "b+2"], ["a+1", None], ["abc-dirty+1"]):
        with pytest.raises(b.MixedBuildError):
            b.require_single_build(bad)
    before = b.composition_version()
    b.composition_version.cache_clear()
    path = b.REPO / b.COMPOSITION_FILES[0]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# probe\n")
        assert b.composition_version() != before
    finally:
        path.write_bytes(original)
        b.composition_version.cache_clear()
    assert b.composition_version() == before


# --- stack configs and resume ---------------------------------------------------------------------


def test_poisoned_weather_sessions_with_shell_have_a_stack_config() -> None:
    from data.runner import (
        REPO,
        STACK_DEFAULT_SERVERS,
        _composites,
        _uses_shell,
        stack_servers_path,
    )

    odd = [c for c in _composites(None, 42) if _uses_shell(c) and weather_config(c)[0] != "clean"]
    assert odd
    for c in odd:
        path = stack_servers_path(c)
        assert path != STACK_DEFAULT_SERVERS and (REPO / path).is_file()
        mode, variant = weather_config(c)
        text = (REPO / path).read_text("utf-8")
        assert f"POISON_MODE: {mode}" in text and f'POISON_VARIANT: "{variant}"' in text


async def test_resume_skips_only_sessions_of_the_same_build(tmp_path: Path) -> None:
    import json

    from data.runner import BUILD_ID, run_all

    cs = [c for c in BENIGN_C if _local(c)][:2]
    out = tmp_path / "s.jsonl"
    first = await run_all(cs, out)
    assert first["sessions"] == 2 and first["resumed_skipped"] == 0
    assert (await run_all(cs, out))["sessions"] == 0  # same build: nothing to redo
    rows = [json.loads(x) for x in out.read_text("utf-8").splitlines()]
    for r in rows:  # pretend the file came from another build
        r["build_id"] = "old+000000000000"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    assert (await run_all(cs, out))["sessions"] == 2  # different build: redone
    assert BUILD_ID != "old+000000000000"
    again = await run_all(cs, out, resume_any_build=True)
    assert again["sessions"] == 0 and again["resumed_skipped"] == 2
