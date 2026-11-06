"""Cross-split near-duplicate removal (MinHash LSH) so no test event has a twin in training.

Events are normalised to text (direction, kind, tool, args or output), shingled and MinHashed.
Samples whose Jaccard similarity is above ``threshold`` form a cluster. If a cluster spans
several splits, only the copy in the MOST PROTECTED split survives (test_heldout > test_seen >
val > train): a train sample is dropped for resembling an eval sample, never the reverse.
Duplicates inside one split are kept (they are not leakage). Tool descriptions are handled by
``data.analysis.dedupe_descriptions`` (the same clean description legitimately appears in every
split), so they are skipped here.

Samples are dropped one by one: each carries its own history, so sessions stay readable.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from datasketch import MinHash, MinHashLSH

from data.schema import Sample, Split
from gateway.models import EventKind

PRIORITY = {Split.TEST_HELDOUT: 3, Split.TEST_SEEN: 2, Split.VAL: 1, Split.TRAIN: 0}
NUM_PERM = 128
DEFAULT_MIN_CHARS = 150  # measured on the real capture: see docs in dedupe_across_splits
_WS = re.compile(r"\s+")


def normalise(s: Sample) -> str:
    """Case/whitespace-insensitive text of the event; session ids and numbering do not matter."""
    e = s.event
    body = (
        json.dumps(e.args, sort_keys=True, ensure_ascii=False, default=str)
        if e.args is not None
        else (e.output or "")
    )
    text = f"{e.direction.value} {e.kind.value} {e.server}.{e.tool} {body}"
    return _WS.sub(" ", text.lower()).strip()


def shingles(text: str, *, word_n: int = 3, char_n: int = 5) -> set[str]:
    """Word n-grams; short texts fall back to character n-grams so they are not all unique."""
    words = text.split(" ")
    if len(words) >= word_n + 4:
        return {" ".join(words[i : i + word_n]) for i in range(len(words) - word_n + 1)}
    return {text[i : i + char_n] for i in range(max(1, len(text) - char_n + 1))}


def signature(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM, seed=1)
    for sh in shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def body_chars(s: Sample) -> int:
    e = s.event
    return (
        len(json.dumps(e.args, ensure_ascii=False, default=str))
        if e.args is not None
        else len(e.output or "")
    )


def dedupe_across_splits(
    samples: Iterable[Sample], *, threshold: float = 0.9, min_chars: int = DEFAULT_MIN_CHARS
) -> tuple[list[Sample], dict[str, Any]]:
    """Return (kept samples in the original order, report). Unsplit samples are rejected.

    Events whose args/output are shorter than ``min_chars`` are exempt: they come from the
    sandbox's closed vocabulary (the same few paths, URLs, tables) and repeat by construction."""
    rows = list(samples)
    if any(s.split is None for s in rows):
        raise ValueError("every sample needs a split before cross-split dedup")
    idx = [
        i
        for i, s in enumerate(rows)
        if s.event.kind is not EventKind.TOOL_DESCRIPTION and body_chars(rows[i]) >= min_chars
    ]
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    sigs: dict[int, MinHash] = {}
    for i in idx:
        sigs[i] = signature(normalise(rows[i]))
        lsh.insert(str(i), sigs[i])
    parent = list(range(len(rows)))
    for i in idx:
        for j in lsh.query(sigs[i]):
            a, b = _find(parent, i), _find(parent, int(j))
            if a != b:
                parent[b] = a
    clusters: dict[int, list[int]] = {}
    for i in idx:
        clusters.setdefault(_find(parent, i), []).append(i)

    drop: set[int] = set()
    for members in clusters.values():
        splits = {rows[i].split for i in members}
        if len(splits) < 2:
            continue
        best = max(PRIORITY[sp] for sp in splits if sp is not None)
        drop |= {i for i in members if PRIORITY[rows[i].split] < best}  # type: ignore[index]

    kept = [s for i, s in enumerate(rows) if i not in drop]
    dropped = [rows[i] for i in sorted(drop)]
    report = {
        "threshold": threshold,
        "min_chars": min_chars,
        "considered": len(idx),
        "cross_split_clusters": sum(
            1 for m in clusters.values() if len({rows[i].split for i in m}) > 1
        ),
        "dropped": len(dropped),
        "dropped_by_split": dict(Counter(s.split.value for s in dropped if s.split)),
        "dropped_by_label": dict(Counter(s.label.value for s in dropped)),
        "dropped_malicious_by_family": dict(
            Counter(s.attack_family.value for s in dropped if s.attack_family)
        ),
        "dropped_by_direction": dict(Counter(s.event.direction.value for s in dropped)),
    }
    return kept, report
