"""Mutators: provenance, split inheritance, labels, that encoded leaks stay detectable, that
identifiers are never mangled, the per-split size target, and sandbox replay."""

from __future__ import annotations

import base64
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from data.attacks.templates import (
    assign_attack_splits,
    bind_values,
    compose_attacks,
    expand_all,
    gold_actions,
    resolve_content,
)
from data.benign.templates import expand_all as expand_benign
from data.composer import Composite, Unit, assign_splits, flow_violations, units_from_sessions
from data.generators.mutators import (
    HOMOGLYPHS,
    MUTATORS,
    changes_events,
    inherit_splits,
    mutate,
    mutate_all,
    translation_problems,
)
from data.schema import HELDOUT_FAMILIES, AttackFamily, Split
from gateway.models import Action
from tools_servers.content_gen import ZW

SEEDS = expand_all(seed=42)
SEED_SPLITS = assign_attack_splits(SEEDS, seed=42)
MUTANTS = mutate_all(SEEDS, seed=42)
SPLITS = SEED_SPLITS | inherit_splits(MUTANTS, SEED_SPLITS)
BY_ID = {u.seed_id: u for u in SEEDS}
EVENT_MUTANTS = [m for m in MUTANTS if changes_events(m, BY_ID)]
TRAINABLE = {f.value for f in AttackFamily if f not in HELDOUT_FAMILIES}


def _mutant_composites() -> list[Composite]:
    benign = units_from_sessions(expand_benign(seed=42))
    return compose_attacks(EVENT_MUTANTS, SPLITS, benign, assign_splits(benign, seed=42), seed=42)


def test_every_mutator_applies_and_is_recorded() -> None:
    sources = Counter(m.source for m in MUTANTS)
    assert set(sources) == {f"mutated:{op}" for op in MUTATORS}, sources
    for m in MUTANTS:
        seed = BY_ID[m.parent]  # type: ignore[index]
        assert m.seed_id == f"{seed.seed_id}~{m.source.split(':')[1]}"
        assert (m.family, m.goal, m.template_id, m.kind) == (
            seed.family, seed.goal, seed.template_id, seed.kind,
        )  # fmt: skip


def test_deterministic() -> None:
    assert mutate_all(SEEDS, seed=42) == MUTANTS
    u = next(u for u in SEEDS if u.template_id == "ms_env_by_email")
    assert mutate(u, "homoglyph", seed_value=1) != mutate(u, "homoglyph", seed_value=2)


def test_mutants_stay_in_their_seeds_split() -> None:
    for m in MUTANTS:
        assert SPLITS[m.seed_id] is SEED_SPLITS[m.parent]  # type: ignore[index]
    assert not any(SPLITS[m.seed_id] is Split.TRAIN for m in MUTANTS
                   if AttackFamily(str(m.family)) in HELDOUT_FAMILIES)  # fmt: skip


def test_gold_actions_recomputed_and_split_destructive_revokes() -> None:
    for m in MUTANTS:
        assert any(g is not None for g in gold_actions(m.steps)), m.seed_id
    split_del = [m for m in MUTANTS if m.source == "mutated:split_args"
                 and any("rowid % 2" in str(s.args) for s in m.steps)]  # fmt: skip
    assert split_del
    for m in split_del:
        golds = [g for g in gold_actions(m.steps) if g is not None]
        assert Action.BLOCK in golds and Action.REVOKE_TOOL in golds


def test_encoded_and_hidden_leaks_still_violate_flow_rules() -> None:
    """Every mutant leak (base64 / hex / percent-encoded / split halves / HTML or markdown
    hidden) is still caught by the LABELING flow rules at exactly the marked steps."""
    seen_ops: set[str] = set()
    for c in _mutant_composites():
        marked = {i for i, st in enumerate(c.steps)
                  if st.attack is not None and st.attack.leaked_class is not None}  # fmt: skip
        found = {int(re.search(r" step (\d+) ", p).group(1)) for p in flow_violations(c)}  # type: ignore[union-attr]
        assert found == marked, (c.session_key, flow_violations(c))
        if marked:
            seen_ops.add(c.units[c.focal_index].source)
    assert {"mutated:b64", "mutated:hex", "mutated:pct", "mutated:split_args",
            "mutated:md_hide"} <= seen_ops  # fmt: skip


@pytest.mark.parametrize("op", ["b64", "hex", "pct"])
def test_path_decoders_decode_to_the_seed_path(op: str) -> None:
    pairs = []
    for m in EVENT_MUTANTS:
        if m.source != f"mutated:{op}":
            continue
        seed = BY_ID[m.parent]  # type: ignore[index]
        for a, b in zip(seed.steps, m.steps, strict=False):
            if a != b and b.tool == "shell.run" and "open(" in b.args["cmd"]:
                pairs.append((a, b))
    assert pairs
    for a, b in pairs:
        path = a.args.get("path") or a.args["cmd"].split()[-1]
        code = re.fullmatch(r'python3 -c "(.*)"', b.args["cmd"]).group(1)  # type: ignore[union-attr]
        probe = code.replace("print(open(", "print((").replace(").read())", "))")
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                             check=True).stdout.strip()  # fmt: skip
        assert out == path, (b.args["cmd"], out)


def test_shell_commands_have_no_control_characters() -> None:
    for m in MUTANTS:
        for st in m.steps:
            if st.tool == "shell.run":
                assert not re.search(r"[\x00-\x1f]", st.args["cmd"]), st.args["cmd"]


def test_resolved_b64_body_decodes_to_the_sessions_env() -> None:
    from data.sandbox_values import SessionValues

    m = next(m for m in EVENT_MUTANTS if m.source == "mutated:b64"
             and m.template_id == "ms_env_by_email")  # fmt: skip
    c = bind_values(Composite("k", Split.TRAIN, "attack", (m,), 0, values_seed="k"))
    body = next(st.args["body"] for st in c.steps if st.tool == "email_send.send")
    assert base64.b64decode(body).decode() == SessionValues.generate("k").dotenv()


# --- text mutators -------------------------------------------------------------------------------

_IDENT = re.compile(r"[\w.-]+@[\w.-]+|[\w.-]*\w[./]\w[\w./-]*|\.\w[\w./-]*|<<[^>]+>>|\{[a-z_]+\}")


def test_text_mutators_never_touch_identifiers() -> None:
    for m in MUTANTS:
        if m.source not in ("mutated:homoglyph", "mutated:zero_width", "mutated:synonym"):
            continue
        seed = BY_ID[m.parent]  # type: ignore[index]
        assert _IDENT.findall(m.prompt) == _IDENT.findall(seed.prompt), (seed.prompt, m.prompt)
        for a, b in zip(seed.steps, m.steps, strict=True):
            assert (a.tool, a.attack) == (b.tool, b.attack)
            for k in a.args:
                if k not in ("subject", "body"):
                    assert a.args[k] == b.args[k]


def test_homoglyph_and_zero_width_actually_obfuscate() -> None:
    cyr = set(HOMOGLYPHS.values())
    hg = [m for m in MUTANTS if m.source == "mutated:homoglyph"]
    zw = [m for m in MUTANTS if m.source == "mutated:zero_width"]

    def text(m: Unit) -> str:
        return m.prompt + " ".join(str(v) for s in m.steps for v in s.args.values())

    assert all(any(ch in cyr for ch in text(m)) for m in hg)
    assert all(ZW in text(m) for m in zw)


def test_translations_cover_every_template_with_matching_slots() -> None:
    assert translation_problems() == []
    es = [m for m in MUTANTS if m.source == "mutated:lang_es"]
    hi = [m for m in MUTANTS if m.source == "mutated:lang_hi"]
    assert len(es) == len(hi) == len(SEEDS)
    assert all(re.search(r"[ऀ-ॿ]", m.prompt) for m in hi)
    assert not any("{" in m.prompt for m in es + hi)


# --- size target -------------------------------------------------------------------------------


def test_val_and_test_seen_reach_the_size_target_with_every_trainable_family() -> None:
    """At least ~100 attack sessions per evaluation split, counting only seeds and mutants
    whose events differ from their seed (prompt-only mutants replay to the same events)."""
    for sp in (Split.VAL, Split.TEST_SEEN):
        units = [u for u in SEEDS + EVENT_MUTANTS if SPLITS[u.seed_id] is sp]
        assert len(units) >= 100, (sp, len(units))
        assert {u.family for u in units} == TRAINABLE, sp


def test_no_identical_event_mutants_counted_twice() -> None:
    sigs = Counter((m.parent, tuple((s.tool, str(s.args)) for s in m.steps)) for m in EVENT_MUTANTS)
    assert max(sigs.values()) == 1


# --- replay --------------------------------------------------------------------------------------


async def test_non_shell_event_mutants_run_in_the_sandbox(tmp_path: Path) -> None:
    from agents.outcomes import classify, result_text
    from tests.unit.harness import agent_client, gateway_app
    from tools_servers.sandbox import prepare_sandbox
    from tools_servers.sql_db.server import prepare_database

    # One mutant per (template, op): every mutated shape, within the harness time limit.
    shapes: dict[tuple[str, str], Unit] = {}
    for m in EVENT_MUTANTS:
        if not any(s.tool.startswith("shell.") for s in m.steps):
            shapes.setdefault((m.template_id, m.source), m)
    units = list(shapes.values())
    assert len(units) > 30
    failures: list[str] = []
    async with gateway_app(tmp_path, shadow_mode=True, deadline_s=1200) as app:
        for u in units:
            sp = SPLITS[u.seed_id]
            c = bind_values(Composite(u.seed_id, sp, "attack", (resolve_content(u, sp),), 0,
                                      values_seed=u.seed_id))  # fmt: skip
            prepare_sandbox(tmp_path / "fs", reset=True)
            prepare_database(tmp_path / "db" / "app.db", reset=True)
            async with agent_client(app) as client:
                for st in c.steps:
                    if st.op == "list_tools":
                        await client.list_tools()
                        continue
                    res = await client.call_tool(st.tool, st.args)
                    retry = st.attack is not None and st.attack.prior_destructive_attempts > 0
                    if classify(res) == "ERROR" and not retry:
                        failures.append(f"{u.seed_id} {st.tool} {str(st.args)[:80]}: "
                                        f"{result_text(res)[:120]}")  # fmt: skip
    assert not failures, "\n".join(failures[:25])
