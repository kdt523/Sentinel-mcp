"""Draw a stratified random sample of a built dataset for manual label review.

    python -m data.review --version 1 --n 100

Writes ``data/processed/v<N>/review.md``: each entry shows the split, family, gold action and the
event with its last history steps, plus an empty verdict line for the reviewer. Strata are the
attack families (malicious), hard negatives and ordinary benign events, so rare cases are not
drowned out. Sampling is seeded: the same dataset gives the same review set.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from data.build_dataset import OUT_ROOT, SPLIT_ORDER, load_samples, stratum
from data.schema import Sample

EXCERPT = 600


def draw(samples: list[Sample], n: int, *, seed: int = 42) -> list[Sample]:
    """About ``n`` samples spread evenly over strata (fewer if a stratum is small)."""
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[stratum(s)].append(s)
    rng = random.Random(f"{seed}:review")
    keys = sorted(groups)
    quota = {k: 0 for k in keys}
    remaining = min(n, len(samples))
    while remaining > 0 and any(quota[k] < len(groups[k]) for k in keys):
        for k in keys:
            if remaining > 0 and quota[k] < len(groups[k]):
                quota[k] += 1
                remaining -= 1
    picked: list[Sample] = []
    for k in keys:
        picked += rng.sample(sorted(groups[k], key=lambda s: s.sample_id), quota[k])
    return sorted(picked, key=lambda s: (stratum(s), s.sample_id))


def render(s: Sample, i: int) -> str:
    e = s.event
    body = json.dumps(e.args, ensure_ascii=False) if e.args is not None else (e.output or "")
    hist = "\n".join(
        f"    - {h.direction.value} {h.tool}: {h.preview[:120]}" for h in s.history[-3:]
    )
    return (
        f"### {i}. {stratum(s)} | split={s.split.value if s.split else '?'} | "
        f"gold={s.gold_action.value} | source={s.source}\n"
        f"- session `{s.session_id}` step {s.step_idx}; {e.direction.value} {e.kind.value} "
        f"`{e.server}.{e.tool}`\n"
        f"- goal: {s.attack_goal or '-'}\n"
        f"- recent history:\n{hist or '    (none)'}\n\n"
        f"```\n{body[:EXCERPT]}\n```\n\n"
        f"Verdict (ok / wrong label / unclear): \n\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    a = ap.parse_args()
    root = a.out_root / f"v{a.version}"
    rows = load_samples([root / f"{sp.value}.jsonl" for sp in SPLIT_ORDER])
    picked = draw(rows, a.n, seed=a.seed)
    text = f"# Label review - dataset v{a.version} ({len(picked)} samples)\n\n"
    text += "".join(render(s, i) for i, s in enumerate(picked, 1))
    (root / "review.md").write_text(text, "utf-8")
    print(f"wrote {root / 'review.md'} ({len(picked)} samples)")


if __name__ == "__main__":
    main()
