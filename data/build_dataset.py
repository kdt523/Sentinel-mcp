"""Build a versioned dataset from captured samples:  python -m data.build_dataset --version 1 ...

Pipeline (each step is reported in stats.json):

1. load + validate every input (schema), refuse mixed / missing / dirty ``build_id`` (no override);
2. drop events that never happen in an enforced deployment (``post_termination``);
3. write sessions flagged ``agent_misbehaviour`` to a separate analysis file (not in any split);
4. dedupe repeated clean tool descriptions, then cross-split near-duplicates (MinHash);
5. check the invariants (no leakage, held-out families/source only in test_heldout, every gold
   action present in train, ...) - a violation stops the build;
6. train only: hide past outcomes in history with probability ``--history-outcome-dropout``;
7. write the splits, per-sample train weights, stats.json, DATASET_CARD.md and a checksum.

Processed data is NOT committed (data/processed is git-ignored); eval results record the
dataset version and checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from data import analysis, dedup
from data.build_id import current_build_id, require_single_build
from data.llm_stats import fire_stats
from data.schema import HELDOUT_FAMILIES, Label, Sample, Split
from gateway.models import Action

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "data" / "processed"
SOURCES = REPO / "data" / "attacks" / "importers" / "sources.yaml"
SPLIT_ORDER = (Split.TRAIN, Split.VAL, Split.TEST_SEEN, Split.TEST_HELDOUT)
HELDOUT_SOURCES = ("agentdojo",)
WEIGHT_CLIP = (0.2, 5.0)
DATASET_SCHEMA = "1"

LIMITATIONS = (
    "Sessions come from scripted templates and a small local LLM (qwen2.5 7B/14B); real-world "
    "agent behaviour is broader. Everything runs in a sandbox with FAKE secrets only.",
    "Short sandbox events (a file path, a URL, a table name) repeat across splits by construction "
    "and are exempt from the near-duplicate check (floor: "
    f"{dedup.DEFAULT_MIN_CHARS} characters). Metrics are per session, not per event.",
    "Labels come from ground truth (what the generator or the observed event actually was), never "
    "from gateway decisions; LLM sessions are labelled event by event with data/LABELING.md.",
    "LLM sessions where the attack did not fire are labelled by what actually happened; per-family "
    "'attack fired' rates are in stats.json.",
    "Public-benchmark sessions (InjecAgent, AgentDojo) cannot be replayed in the sandbox: "
    "verified is None, so they count for detection metrics only, never ASR.",
    "AgentDojo is the held-out SOURCE; tool_poisoning and social_engineering_approval are the "
    "held-out FAMILIES. None of them appears in train or val.",
)


class BuildError(ValueError):
    """The inputs violate a dataset invariant; nothing is written."""


# --- loading ----------------------------------------------------------------------------------


def load_samples(paths: Iterable[Path]) -> list[Sample]:
    rows: list[Sample] = []
    for p in paths:
        for n, line in enumerate(p.read_text("utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(Sample.model_validate_json(line))
                except ValueError as exc:
                    raise BuildError(f"{p}:{n}: {exc}") from exc
    return rows


# --- transformations --------------------------------------------------------------------------


def history_outcome_dropout(samples: Iterable[Sample], p: float, *, seed: int) -> list[Sample]:
    """With probability ``p`` per sample, replace every past outcome by 'unknown' (train only)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("history-outcome-dropout must be in [0, 1]")
    out: list[Sample] = []
    for s in samples:
        draw = random.Random(f"{seed}:outcome-dropout:{s.sample_id}").random()
        if p > 0 and s.history and draw < p:
            hist = [h.model_copy(update={"outcome": "unknown"}) for h in s.history]
            s = s.model_copy(update={"history": hist})
        out.append(s)
    return out


def stratum(s: Sample) -> str:
    if s.label is Label.MALICIOUS:
        return f"malicious:{s.attack_family.value if s.attack_family else 'incident'}"
    return f"benign:{s.event.kind.value}{':hard' if s.hard_negative else ''}"


def train_weights(train: Sequence[Sample]) -> tuple[dict[str, float], dict[str, Any]]:
    """Per-sample weights that balance strata (malicious family / benign kind) in training.

    w = N / (K * n_stratum), clipped to ``WEIGHT_CLIP`` and rescaled to mean 1. Val and test are
    never weighted. The weights are applied at training time (sampling or loss weighting); the
    data itself keeps every session."""
    if not train:
        return {}, {"strata": {}}
    counts = Counter(stratum(s) for s in train)
    n, k = len(train), len(counts)
    raw = {key: min(max(n / (k * c), WEIGHT_CLIP[0]), WEIGHT_CLIP[1]) for key, c in counts.items()}
    mean = sum(raw[stratum(s)] for s in train) / n
    weights = {s.sample_id: round(raw[stratum(s)] / mean, 4) for s in train}
    summary = {key: {"samples": counts[key], "weight": round(raw[key] / mean, 4)}
               for key in sorted(counts)}  # fmt: skip
    return weights, {"strata": summary, "clip": list(WEIGHT_CLIP), "mean": 1.0}


# --- invariants -------------------------------------------------------------------------------


def check_invariants(splits: dict[Split, list[Sample]]) -> list[str]:
    """Violations found (empty = clean). Every one of them stops the build."""
    bad: list[str] = []
    all_rows = [s for sp in SPLIT_ORDER for s in splits[sp]]
    ids = Counter(s.sample_id for s in all_rows)
    bad += [f"duplicate sample_id {i}" for i, c in ids.items() if c > 1][:5]

    sess: dict[str, set[Split]] = defaultdict(set)
    seeds: dict[str, set[Split]] = defaultdict(set)
    for sp in SPLIT_ORDER:
        for s in splits[sp]:
            sess[s.session_id].add(sp)
            if s.label is Label.MALICIOUS:
                seeds[s.seed_id].add(sp)
    bad += [f"session {k} is in {sorted(x.value for x in v)}" for k, v in sess.items()
            if len(v) > 1][:5]  # fmt: skip
    bad += [f"attack seed {k} is in {sorted(x.value for x in v)}" for k, v in seeds.items()
            if len(v) > 1][:5]  # fmt: skip

    for sp in (Split.TRAIN, Split.VAL, Split.TEST_SEEN):
        for s in splits[sp]:
            fams = {s.attack_family, s.session_family}
            if fams & HELDOUT_FAMILIES:
                bad.append(f"held-out family in {sp.value}: {s.session_id}")
                break
        for s in splits[sp]:
            if s.source in HELDOUT_SOURCES:
                bad.append(f"held-out source {s.source} in {sp.value}: {s.session_id}")
                break
    for s in splits[Split.TEST_HELDOUT]:
        if s.source == "injecagent":
            bad.append(f"train/val source injecagent in test_heldout: {s.session_id}")
            break
    for s in all_rows:
        if s.post_termination or s.agent_misbehaviour:
            bad.append(f"{s.session_id}: post_termination/agent_misbehaviour sample in a split")
            break
        if s.label is Label.BENIGN and s.gold_action is not Action.ALLOW:
            bad.append(f"{s.sample_id}: benign sample with gold {s.gold_action.value}")
            break
    train_actions = {s.gold_action for s in splits[Split.TRAIN]}
    for sp in SPLIT_ORDER[1:]:
        missing = {s.gold_action for s in splits[sp]} - train_actions
        if missing:
            bad.append(f"gold actions in {sp.value} but never in train: "
                       f"{sorted(a.value for a in missing)}")  # fmt: skip
    return bad


# --- stats and card ---------------------------------------------------------------------------


def _ratio(rows: Sequence[Sample]) -> dict[str, Any]:
    by_session: dict[str, bool] = {}
    for s in rows:
        by_session[s.session_id] = by_session.get(s.session_id, False) or s.label is Label.MALICIOUS
    n_att = sum(by_session.values())
    mal = sum(s.label is Label.MALICIOUS for s in rows)
    return {
        "samples": len(rows), "malicious_samples": mal,
        "attack_share_samples": round(mal / len(rows), 3) if rows else 0.0,
        "sessions": len(by_session), "attack_sessions": n_att,
        "attack_share_sessions": round(n_att / len(by_session), 3) if by_session else 0.0,
    }  # fmt: skip


def compute_stats(splits: dict[Split, list[Sample]], *, meta: dict[str, Any]) -> dict[str, Any]:
    fam: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sp in SPLIT_ORDER:
        for s in splits[sp]:
            name = s.attack_family.value if s.attack_family else "benign"
            key = f"{name}|{s.event.direction.value}"
            fam[key][sp.value] += 1
    rows = [s for sp in SPLIT_ORDER for s in splits[sp]]
    return {
        **meta,
        "splits": {sp.value: _ratio(splits[sp]) for sp in SPLIT_ORDER},
        "family_direction_by_split": {k: dict(v) for k, v in sorted(fam.items())},
        "by_kind": {
            sp.value: dict(Counter(s.event.kind.value for s in splits[sp])) for sp in SPLIT_ORDER
        },  # fmt: skip
        "distinct_benign_focal_tasks": {
            sp.value: len({s.template_id for s in splits[sp] if s.label is Label.BENIGN})
            for sp in SPLIT_ORDER
        },  # fmt: skip
        "hard_negatives": {
            sp.value: sum(s.hard_negative for s in splits[sp]) for sp in SPLIT_ORDER
        },  # fmt: skip
        "sources": dict(Counter(s.source.split(":")[0] for s in rows)),
        "generators": dict(Counter(s.generator.split(":")[0] for s in rows)),
        "hiding_techniques": dict(Counter(s.hiding_technique for s in rows if s.hiding_technique)),
        "verified": {
            str(k): v
            for k, v in Counter(s.verified for s in rows if s.label is Label.MALICIOUS).items()
        },  # fmt: skip
        "policy_disagrees": {
            sp.value: sum(s.policy_disagrees for s in splits[sp]) for sp in SPLIT_ORDER
        },  # fmt: skip
        "gold_actions": {
            sp.value: dict(Counter(s.gold_action.value for s in splits[sp])) for sp in SPLIT_ORDER
        },  # fmt: skip
    }


def dataset_card(stats: dict[str, Any]) -> str:
    reg = yaml.safe_load(SOURCES.read_text("utf-8"))["sources"] if SOURCES.exists() else {}
    used = set(stats.get("sources", {}))
    lines = [
        f"# SentinelMCP dataset v{stats['version']}", "",
        f"* build_id: `{stats['build_id']}`",
        f"* dataset checksum (sha256 over the split files): `{stats['dataset_checksum']}`",
        f"* seed: {stats['seed']} - history outcome dropout (train only): "
        f"{stats['history_outcome_dropout']}",
        "", "## Splits", "",
        "| split | samples | malicious | attack share (samples) | sessions | attack sessions |",
        "|---|---|---|---|---|---|",
    ]  # fmt: skip
    for sp, r in stats["splits"].items():
        lines.append(f"| {sp} | {r['samples']} | {r['malicious_samples']} | "
                     f"{r['attack_share_samples']:.0%} | {r['sessions']} | "
                     f"{r['attack_sessions']} |")  # fmt: skip
    lines += ["", "## Sources and licences", ""]
    for name, r in reg.items():
        if name in used:
            lines.append(f"* **{r['name']}** - {r['license']}, {r['copyright']}; pinned to "
                         f"`{r['commit_short']}`; role: {r['role']}.")  # fmt: skip
            lines += [f"    * limitation: {t}" for t in r.get("limitations", [])]
    lines += ["* **Generated** (scripted templates and local LLM sessions): this repository.",
              "", "## Known limitations", ""]  # fmt: skip
    lines += [f"* {t}" for t in LIMITATIONS]
    d = stats["dedup"]
    lines += ["", "## Build steps", "",
              f"* post-termination events removed: {stats['removed']['post_termination']}",
              f"* agent-misbehaviour sessions moved to the analysis file: "
              f"{stats['removed']['agent_misbehaviour_sessions']}",
              f"* repeated clean tool descriptions removed: {stats['removed']['descriptions']}",
              f"* cross-split near-duplicates removed: {d['dropped']} "
              f"(threshold {d['threshold']}, floor {d['min_chars']} chars)",
              "", "Metrics and reporting follow `docs/evaluation-metrics.md`.", ""]  # fmt: skip
    return "\n".join(lines)


# --- build ------------------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: Iterable[Sample]) -> None:
    path.write_text("".join(s.model_dump_json() + "\n" for s in rows), "utf-8")


def _checksum(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def build(inputs: Sequence[Path], *, version: int, seed: int = 42,
          history_outcome_dropout: float = 0.2, min_chars: int = dedup.DEFAULT_MIN_CHARS,
          llm_log: Path | None = None, out_root: Path = OUT_ROOT) -> dict[str, Any]:  # fmt: skip
    samples = load_samples(inputs)
    if not samples:
        raise BuildError("no samples")
    build_id = require_single_build(s.build_id for s in samples)  # raises MixedBuildError
    unsplit = [s.session_id for s in samples if s.split is None]
    if unsplit:
        raise BuildError(f"{len(unsplit)} samples without a split, e.g. {unsplit[0]}")

    n0 = len(samples)
    samples = [s for s in samples if not s.post_termination]
    post_term = n0 - len(samples)
    samples, flagged = analysis.split_misbehaviour(samples)
    n1 = len(samples)
    samples = analysis.dedupe_descriptions(samples)
    descriptions_removed = n1 - len(samples)
    samples, dedup_report = dedup.dedupe_across_splits(samples, min_chars=min_chars)

    splits: dict[Split, list[Sample]] = {sp: [] for sp in SPLIT_ORDER}
    for s in samples:
        assert s.split is not None
        splits[s.split].append(s)
    problems = check_invariants(splits)
    if problems:
        raise BuildError("dataset invariants violated:\n  " + "\n  ".join(problems))

    splits[Split.TRAIN] = history_outcome_dropout_train(
        splits[Split.TRAIN], history_outcome_dropout, seed
    )
    for sp in SPLIT_ORDER:
        splits[sp].sort(key=lambda s: (s.session_id, s.step_idx))
    weights, weight_summary = train_weights(splits[Split.TRAIN])

    out = out_root / f"v{version}"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for sp in SPLIT_ORDER:
        path = out / f"{sp.value}.jsonl"
        _write_jsonl(path, splits[sp])
        files.append(path)
    (out / "train_weights.json").write_text(json.dumps(weights, sort_keys=True), "utf-8")
    n_flagged = analysis.write_analysis_file(flagged, out / "misbehaviour_analysis.jsonl")

    meta = {
        "version": version, "seed": seed, "build_id": build_id, "schema_version": DATASET_SCHEMA,
        "history_outcome_dropout": history_outcome_dropout,
        "dataset_checksum": _checksum(files),
        "file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "inputs": [str(p) for p in inputs],
        "removed": {"post_termination": post_term, "descriptions": descriptions_removed,
                    "agent_misbehaviour_sessions": n_flagged,
                    "agent_misbehaviour_samples": len(flagged)},  # fmt: skip
        "dedup": dedup_report, "train_weights": weight_summary,
        "misbehaviour": analysis.misbehaviour_summary(flagged),
    }  # fmt: skip
    stats = compute_stats(splits, meta=meta)
    if llm_log is not None:
        rows = [json.loads(x) for x in llm_log.read_text("utf-8").splitlines() if x.strip()]
        stats["llm_sessions"] = fire_stats(rows)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), "utf-8")
    (out / "DATASET_CARD.md").write_text(dataset_card(stats), "utf-8")
    return stats


def history_outcome_dropout_train(train: list[Sample], p: float, seed: int) -> list[Sample]:
    return history_outcome_dropout(train, p, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, nargs="+", required=True, help="sample JSONL files")
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--history-outcome-dropout", type=float, default=0.2)
    ap.add_argument("--min-chars", type=int, default=dedup.DEFAULT_MIN_CHARS)
    ap.add_argument("--llm-log", type=Path, default=None, help="LLM session log (fire rates)")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    a = ap.parse_args()
    stats = build(a.inputs, version=a.version, seed=a.seed, min_chars=a.min_chars,
                  history_outcome_dropout=a.history_outcome_dropout, llm_log=a.llm_log,
                  out_root=a.out_root)  # fmt: skip
    print(json.dumps({"version": stats["version"], "build_id": stats["build_id"],
                      "checksum": stats["dataset_checksum"], "splits": stats["splits"],
                      "current_build_id": current_build_id()}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
