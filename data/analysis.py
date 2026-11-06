"""Build-step helpers: keep out sessions that must not enter the metrics, and dedupe descriptions.

* ``split_misbehaviour``: sessions flagged ``agent_misbehaviour`` (the agent did something
  unrequested but harmless) are excluded from attack and false-positive metrics and written to a
  separate analysis file (``write_analysis_file``), where they stay available for study.
* ``dedupe_descriptions``: every session lists the same tool descriptions again, which would make
  tool-description events about three quarters of all samples. Per session we keep the FIRST
  listing of each tool and every description event that is poisoned or changed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from data.schema import Label, Sample
from gateway.models import Direction, EventKind


def split_misbehaviour(samples: Iterable[Sample]) -> tuple[list[Sample], list[Sample]]:
    """(samples that enter the splits, samples of flagged sessions)."""
    kept: list[Sample] = []
    flagged: list[Sample] = []
    for s in samples:
        (flagged if s.agent_misbehaviour else kept).append(s)
    return kept, flagged


def write_analysis_file(flagged: Iterable[Sample], path: Path) -> int:
    """Write flagged sessions' samples (one JSON per line); returns the number of sessions."""
    rows = list(flagged)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(s.model_dump_json() + "\n" for s in rows), "utf-8")
    return len({s.session_id for s in rows})


def misbehaviour_summary(flagged: Iterable[Sample]) -> dict[str, object]:
    """Counts for stats.json: flagged sessions and the tools involved."""
    rows = list(flagged)
    by_tool: dict[str, int] = {}
    for s in rows:
        if s.event.direction is Direction.REQUEST and s.event.kind is EventKind.TOOL_CALL:
            by_tool[f"{s.event.server}.{s.event.tool}"] = by_tool.get(
                f"{s.event.server}.{s.event.tool}", 0) + 1  # fmt: skip
    return {"sessions": len({s.session_id for s in rows}), "samples": len(rows),
            "requests_by_tool": dict(sorted(by_tool.items()))}  # fmt: skip


def dedupe_descriptions(samples: Iterable[Sample]) -> list[Sample]:
    """Drop repeated, unchanged, benign tool-description events within a session."""
    seen: set[tuple[str, str]] = set()
    out: list[Sample] = []
    for s in samples:
        if s.event.kind is EventKind.TOOL_DESCRIPTION:
            key = (s.session_id, f"{s.event.server}.{s.event.tool}")
            interesting = s.label is Label.MALICIOUS or s.event.description_changed
            if key in seen and not interesting:
                continue
            seen.add(key)
        out.append(s)
    return out


def description_share(samples: Iterable[Sample]) -> float:
    rows = list(samples)
    if not rows:
        return 0.0
    return sum(s.event.kind is EventKind.TOOL_DESCRIPTION for s in rows) / len(rows)


__all__ = ["json", "dedupe_descriptions", "description_share", "misbehaviour_summary",
           "split_misbehaviour", "write_analysis_file"]  # fmt: skip
