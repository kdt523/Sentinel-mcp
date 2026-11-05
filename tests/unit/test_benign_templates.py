"""Benign templates: structure, coverage, prompt/step consistency, and that every expanded
session actually runs through the gateway without tool errors."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.benign.templates import (
    TEMPLATES_PATH,
    Template,
    TemplateError,
    expand,
    expand_all,
    load_templates,
)
from tests.unit.harness import agent_client, gateway_app
from tools_servers.sandbox import prepare_sandbox
from tools_servers.sql_db.server import prepare_database

TEMPLATES = load_templates()
SESSIONS = expand_all(seed=42)
STEPS = [st for s in SESSIONS for st in s.steps]


def test_coverage_targets() -> None:
    assert len(TEMPLATES) >= 100  # decided 2026-11-05
    assert all(len([s for s in SESSIONS if s.template_id == t.id]) >= 5 for t in TEMPLATES)
    assert {st.tool.split(".")[0] for st in STEPS} == {
        "filesystem", "shell", "sql_db", "email_send", "http_fetch", "weather", "notes",
    }  # fmt: skip


def test_at_least_60_distinct_focal_tasks_per_eval_split() -> None:
    from data.composer import assign_splits, units_from_sessions
    from data.schema import Split

    units = units_from_sessions(SESSIONS)
    splits = assign_splits(units, seed=42)
    for sp in (Split.VAL, Split.TEST_SEEN, Split.TEST_HELDOUT):
        pool = [u for u in units if splits[u.seed_id] is sp]
        assert len({u.template_id for u in pool}) >= 60, sp
        assert len(pool) >= 60, sp


def test_hard_negatives_are_at_least_20_percent_of_benign_events() -> None:
    share = sum(st.hard_negative for st in STEPS) / len(STEPS)
    assert share >= 0.20, f"hard negatives only {share:.0%}"
    for t in TEMPLATES:
        if t.hard_negative:
            assert t.note, f"{t.id}: hard negatives must say why they are legitimate"


def test_expansion_is_deterministic_and_seed_sensitive() -> None:
    again = expand_all(seed=42)
    assert [(s.session_key, s.prompt, s.steps) for s in again] == [
        (s.session_key, s.prompt, s.steps) for s in SESSIONS
    ]
    other = expand_all(seed=7)
    assert [s.prompt for s in other] != [s.prompt for s in SESSIONS]


def test_sessions_have_unique_keys_and_filled_slots() -> None:
    keys = [s.session_key for s in SESSIONS]
    assert len(keys) == len(set(keys))
    for s in SESSIONS:
        assert "{" not in s.prompt or "}" not in s.prompt, s.prompt
        assert "{" not in str([st.args for st in s.steps]).replace("{'", "").replace('{"', "")


def test_typed_slots_keep_yaml_types() -> None:
    raws = {st.args["raw"] for s in SESSIONS for st in s.steps if "raw" in st.args}
    assert raws == {True, False}


def test_no_attack_giveaway_phrases_in_benign_prompts() -> None:
    giveaways = ("ignore previous", "exfil", "evil.example", "SYSTEM:", "<IMPORTANT>")
    for s in SESSIONS:
        assert not any(g.lower() in s.prompt.lower() for g in giveaways), s.prompt


def test_category_balance() -> None:
    counts = Counter(s.category for s in SESSIONS)
    assert min(counts.values()) >= 20, counts


# --- validation ------------------------------------------------------------------------------


def _template(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "t", "category": "docs",
        "phrasings": ["Read {doc}.", "Open {doc}.", "Show {doc}."],
        "params": {"doc": ["a", "b", "c", "d", "e"]},
        "steps": [{"tool": "filesystem.read_file", "args": {"path": "{doc}"}}],
    }  # fmt: skip
    return base | kw


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        ({"phrasings": ["Read it.", "Open {doc}.", "Show {doc}."]}, "omits"),
        ({"params": {"doc": ["a", "b"]}}, "only 2 variations"),
        ({"steps": [{"tool": "fs", "args": {}}]}, "tool"),
        ({"phrasings": ["x {nope}", "y {doc}", "z {doc}"]}, "undefined slots"),
        ({"free_params": ["ghost"]}, "free_params not in params"),
        ({"phrasings": ["a {doc}", "b {doc}"]}, "phrasings"),
        ({"extra": 1}, "extra"),
    ],
)
def test_template_validation(kw: dict[str, Any], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        Template.model_validate(_template(**kw))


def test_free_params_exempt_from_phrasing_rule() -> None:
    t = Template.model_validate(_template(
        params={"doc": ["a", "b", "c", "d", "e"], "slug": ["x"]},
        steps=[{"tool": "filesystem.write_file",
                "args": {"path": "notes/{slug}", "content": "{doc}"}}],
        free_params=["slug"],
    ))  # fmt: skip
    assert len(expand(t, seed=1)) == 5


def test_bad_file_is_a_template_error(tmp_path: Path) -> None:
    bad = tmp_path / "t.yaml"
    bad.write_text(yaml.safe_dump({"templates": [_template(), _template()]}), "utf-8")
    with pytest.raises(TemplateError, match="duplicate template ids"):
        load_templates(bad)
    assert TEMPLATES_PATH.exists()


# --- every session really runs -----------------------------------------------------------------


async def test_every_non_shell_benign_session_runs_without_tool_errors(tmp_path: Path) -> None:
    """Replay every expanded session that does not use the shell (the shell runs only in its
    container: tests/integration/test_benign_stack.py replays ALL sessions on the stack)
    through the gateway in shadow mode, and require that no tool call fails."""
    from agents.outcomes import classify, result_text

    failures: list[str] = []
    async with gateway_app(tmp_path, shadow_mode=True, deadline_s=1200) as app:
        from data.composer import assign_splits, resolve_unit, units_from_sessions

        units = units_from_sessions(SESSIONS)
        splits = assign_splits(units, seed=42)
        for s in [resolve_unit(u, splits[u.seed_id]) for u in units
                  if not any(st.tool.startswith("shell.") for st in u.steps)]:  # fmt: skip
            # Fresh sandbox per session, exactly as dataset generation does.
            prepare_sandbox(tmp_path / "fs", reset=True)
            prepare_database(tmp_path / "db" / "app.db", reset=True)
            async with agent_client(app) as client:  # one session per spec
                for st in s.steps:
                    res = await client.call_tool(st.tool, st.args)
                    if classify(res) == "ERROR":
                        failures.append(f"{s.seed_id} {st.tool} {st.args}: "
                                        f"{result_text(res)[:120]}")  # fmt: skip
    assert not failures, "\n".join(failures[:20])
