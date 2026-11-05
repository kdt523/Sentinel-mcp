"""Composite sessions: chain template variations ("units") into realistic multi-task sessions.

Guardrails (decided 2026-11-05):

(a) ONE construction for both classes: a session length is drawn from ``LENGTH_DIST`` and a
    *focal* unit is placed at a uniformly random position among benign filler units. For
    attack sessions the focal unit is the attack; for benign sessions it is a benign unit
    (preferring hard negatives). Lengths and focal positions are therefore identically
    distributed across classes; ``composition_stats`` reports them for stats.json.
(b) Every unit is assigned to a split BEFORE composing; a composite only uses units of its
    own split, so no variation crosses splits.
(c) ``flow_violations`` simulates each composite's events with its own fake session values
    and applies the LABELING.md flow rules; a "benign" composite that leaks a secret, or
    sends personal data to an external recipient, is a build error.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from data.benign.templates import SessionSpec, StepCall
from data.sandbox_values import SessionValues
from data.schema import Split
from gateway.detectors import decoded_literals, egress_targets, fingerprints, fingerprints_in_args
from gateway.detectors import variants as decoded_variants


def _path_variants(text: str) -> list[str]:
    """Detector decodings plus quoted literals of any length (shared with the gateway's
    key-material shell rule)."""
    return decoded_variants(text) + decoded_literals(text)[1:]


# Units per session, shared by benign and attack composites.
LENGTH_DIST: dict[int, float] = {1: 0.20, 2: 0.35, 3: 0.30, 4: 0.15}
SPLIT_RATIOS: dict[Split, float] = {
    Split.TRAIN: 0.70, Split.VAL: 0.10, Split.TEST_SEEN: 0.10, Split.TEST_HELDOUT: 0.10,
}  # fmt: skip
INTERNAL = ("email:acme.example", "http:mockweb")
_CONNECTORS = ("Then", "After that,", "Next,", "Also,")


@dataclass(frozen=True)
class Unit:
    """One template variation: the atom that is assigned to a split and composed."""

    seed_id: str
    template_id: str
    category: str
    prompt: str
    steps: tuple[StepCall, ...]
    kind: str = "benign"  # "benign" | "attack"
    family: str | None = None  # attack units: session family
    goal: str | None = None  # attack units: what the attack tries to achieve
    # attack units: per-session upstream overrides, e.g. {"weather": {"mode": "rugpull"}}
    upstreams: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    source: str = "generated"  # "generated" | "mutated:<op>" (schema Sample.source)
    parent: str | None = None  # mutants: seed_id of the seed unit (same split)
    slots: tuple[tuple[str, Any], ...] = ()  # template slot values (language-switch mutator)

    @property
    def hard_negative(self) -> bool:
        return any(s.hard_negative for s in self.steps)


@dataclass(frozen=True)
class Composite:
    session_key: str
    split: Split
    kind: str  # class of the session: "benign" | "attack"
    units: tuple[Unit, ...]
    focal_index: int  # position of the focal unit (attack, or the benign focal)
    values_seed: str  # per-session fake secrets / PII (data/sandbox_values.py)

    @property
    def steps(self) -> tuple[StepCall, ...]:
        return tuple(s for u in self.units for s in u.steps)

    @property
    def focal_step_index(self) -> int:
        return sum(len(u.steps) for u in self.units[: self.focal_index])

    @property
    def prompt(self) -> str:
        parts = [self.units[0].prompt]
        for i, u in enumerate(self.units[1:]):
            parts.append(f"{_CONNECTORS[i % len(_CONNECTORS)]} {u.prompt[0].lower()}{u.prompt[1:]}")
        return " ".join(parts)


def units_from_sessions(sessions: Iterable[SessionSpec], kind: str = "benign") -> list[Unit]:
    return [Unit(s.seed_id, s.template_id, s.category, s.prompt, s.steps, kind) for s in sessions]


# ----------------------------------------------------------------------------- (b) splits


def assign_splits(units: Sequence[Unit], *, seed: int,
                  ratios: dict[Split, float] | None = None) -> dict[str, Split]:  # fmt: skip
    """Assign every unit (template variation) to exactly one split BEFORE any composing.

    Systematic allocation: one continuous slot pattern with the target ratios (e.g. 7x
    train, 1x val, 1x test_seen, 1x test_heldout per 10 slots, interleaved) is walked
    across all templates in turn. Global ratios are exact, and every template is spread
    over the splits. (Per-template rounding starved the small test splits: a template with
    5 variations has only 0.5 of a unit per 10% split.)"""
    ratios = ratios or SPLIT_RATIOS
    slots = _slot_pattern(ratios)
    by_template: dict[str, list[Unit]] = defaultdict(list)
    for u in units:
        by_template[u.template_id].append(u)
    rng = random.Random(f"{seed}:splits")
    order = sorted(by_template)
    rng.shuffle(order)
    cursor = rng.randrange(len(slots))
    out: dict[str, Split] = {}
    for tid in order:
        group = sorted(by_template[tid], key=lambda u: u.seed_id)
        random.Random(f"{seed}:split:{tid}").shuffle(group)
        for u in group:
            out[u.seed_id] = slots[cursor % len(slots)]
            cursor += 1
    return out


def _slot_pattern(ratios: dict[Split, float], size: int = 20) -> list[Split]:
    """Interleaved slot sequence with the given ratios (no long runs of one split)."""
    counts = {sp: round(r * size) for sp, r in ratios.items()}
    pattern: list[Split] = []
    credit = dict.fromkeys(ratios, 0.0)
    for _ in range(sum(counts.values())):
        for sp in ratios:
            credit[sp] += counts[sp]
        best = max(ratios, key=lambda sp: credit[sp])
        credit[best] -= sum(counts.values())
        pattern.append(best)
    return pattern


# ----------------------------------------------------------------------------- (a) composing


def _length(rng: random.Random) -> int:
    return rng.choices(list(LENGTH_DIST), weights=list(LENGTH_DIST.values()))[0]


def compose(focal: Unit, pool: Sequence[Unit], *, split: Split, kind: str, key: str,
            rng: random.Random) -> Composite:  # fmt: skip
    """Place ``focal`` at a uniformly random position among benign fillers from ``pool``
    (same split; fillers come from templates different from each other and the focal)."""
    k = _length(rng)
    candidates = [u for u in pool if u.template_id != focal.template_id and u.kind == "benign"]
    fillers: list[Unit] = []
    used = {focal.template_id}
    for u in rng.sample(candidates, len(candidates)):
        if len(fillers) == k - 1:
            break
        if u.template_id not in used:
            fillers.append(u)
            used.add(u.template_id)
    pos = rng.randrange(len(fillers) + 1)
    units = (*fillers[:pos], focal, *fillers[pos:])
    return Composite(key, split, kind, units, pos, values_seed=key)


BENIGN_SHARE = 0.60  # target benign share of sessions in every split (60/40)


def benign_counts_for_ratio(attack_sessions: dict[Split, int],
                            benign_share: float = BENIGN_SHARE) -> dict[Split, int]:  # fmt: skip
    """Benign composites needed per split so benign / (benign + attack) == ``benign_share``."""
    return {sp: round(n * benign_share / (1 - benign_share)) for sp, n in attack_sessions.items()}


def compose_benign(units: Sequence[Unit], splits: dict[str, Split], *, n: int = 0, seed: int,
                   n_per_split: dict[Split, int] | None = None,
                   hard_negative_share: float | None = None) -> list[Composite]:  # fmt: skip
    """Benign composites: ``n_per_split`` per split, or ``n`` allocated by ``SPLIT_RATIOS``.
    Focal units cycle through the split's units (each is focal equally often, +-1) and are
    hard negatives with the same probability hard negatives have among all units."""
    rng = random.Random(f"{seed}:benign-composites")
    by_split: dict[Split, list[Unit]] = defaultdict(list)
    for u in units:
        by_split[splits[u.seed_id]].append(u)
    share = hard_negative_share
    if share is None:
        share = sum(u.hard_negative for u in units) / max(1, len(units))
    counts = n_per_split or {sp: round(n * r) for sp, r in SPLIT_RATIOS.items()}
    out: list[Composite] = []
    for sp in SPLIT_RATIOS:
        pool = sorted(by_split[sp], key=lambda u: u.seed_id)
        hard = [u for u in pool if u.hard_negative]
        easy = [u for u in pool if not u.hard_negative]
        rng.shuffle(hard)
        rng.shuffle(easy)
        cursor = {"hard": 0, "easy": 0}
        for i in range(counts.get(sp, 0)):
            kind = "hard" if hard and (not easy or rng.random() < share) else "easy"
            group = hard if kind == "hard" else easy
            focal = group[cursor[kind] % len(group)]
            cursor[kind] += 1
            out.append(compose(focal, pool, split=sp, kind="benign",
                               key=f"benign-c:{sp.value}:{i}", rng=rng))  # fmt: skip
    return out


def split_ratio_report(composites: Sequence[Composite]) -> dict[str, dict[str, float]]:
    """Benign / attack session counts and benign share per split (for stats.json)."""
    out: dict[str, dict[str, float]] = {}
    for sp in SPLIT_RATIOS:
        b = sum(1 for c in composites if c.split is sp and c.kind == "benign")
        a = sum(1 for c in composites if c.split is sp and c.kind == "attack")
        out[sp.value] = {"benign": b, "attack": a,
                         "benign_share": round(b / (a + b), 3) if a + b else 0.0}  # fmt: skip
    return out


def composition_stats(composites: Sequence[Composite]) -> dict[str, dict[str, dict[str, int]]]:
    """Session-length and focal-position histograms per class (for stats.json)."""
    stats: dict[str, dict[str, dict[str, int]]] = {}
    for kind in sorted({c.kind for c in composites}):
        group = [c for c in composites if c.kind == kind]
        stats[kind] = {
            "units_per_session": dict(sorted(Counter(str(len(c.units)) for c in group).items())),
            "focal_position": dict(
                sorted(Counter(f"{c.focal_index + 1}/{len(c.units)}" for c in group).items())
            ),  # fmt: skip
            "steps_per_session": dict(sorted(Counter(str(len(c.steps)) for c in group).items())),
        }
    return stats


# ----------------------------------------------------------------------------- (c) flows

_SECRET_PATH = re.compile(r"(^|[\s/'\"])(\.env|\.ssh/[\w.-]+)", re.IGNORECASE)
_PII_PATH = re.compile(r"customers\.csv", re.IGNORECASE)
_PII_SQL = re.compile(r"\bfrom\s+customers\b", re.IGNORECASE)


def _sensitive_reads(step: StepCall, values: SessionValues) -> tuple[list[str], list[str]]:
    """(secret texts, pii texts) this step would bring into the session (conservative)."""
    # Decoded variants too: an encoded path (base64 / hex / percent) still reads the file.
    blob = " ".join(d for v in step.args.values() for d in _path_variants(str(v)))
    secrets, pii = [], []
    if step.tool in ("filesystem.read_file", "shell.run"):
        for m in _SECRET_PATH.finditer(blob):
            secrets.append(values.ssh_private_key() if ".ssh" in m.group(2) else values.dotenv())
        if _PII_PATH.search(blob):
            pii.append(values.customers_csv())
    if step.tool == "sql_db.query" and _PII_SQL.search(blob):
        pii.append(values.customers_csv())
    return secrets, pii


def flow_violations(c: Composite) -> list[str]:
    """LABELING.md flow rules over the composite's events: secret values leaving the session
    (any tool), or personal data sent to a non-internal destination."""
    values = SessionValues.generate(c.values_seed)
    secret_fp: set[str] = set()
    pii_fp: set[str] = set()
    problems: list[str] = []
    for i, step in enumerate(c.steps):
        args_fp = fingerprints_in_args(step.args)
        if secret_fp & args_fp:
            problems.append(
                f"{c.session_key} step {i} {step.tool}: secret values leave the session"
            )
        dests = egress_targets({k: v for k, v in step.args.items() if k in ("to", "url")})
        external = [d for d in dests if not d.startswith(INTERNAL)]
        if pii_fp & args_fp and external:
            problems.append(f"{c.session_key} step {i} {step.tool}: personal data to {external}")
        secrets, pii = _sensitive_reads(step, values)
        for text in secrets:
            secret_fp |= fingerprints(text, sensitive_lines_only=True)
        for text in pii:
            pii_fp |= fingerprints(text, sensitive_lines_only=True)
    return problems
