"""Build-step helpers: misbehaviour partition and description dedupe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from data.analysis import (
    dedupe_descriptions,
    description_share,
    misbehaviour_summary,
    split_misbehaviour,
    write_analysis_file,
)
from data.schema import Sample


def sample(session: str, idx: int, **kw: Any) -> Sample:
    base: dict[str, Any] = {
        "sample_id": f"{session}-{idx:03d}-xx", "session_id": session, "step_idx": idx,
        "context": {"request_count": 0},
        "event": {"direction": "request", "kind": "tool_call", "server": "filesystem",
                  "tool": "read_file", "args": {"path": "docs/readme.md"}},
        "label": "benign", "gold_action": "ALLOW", "source": "llm_session:qwen2.5:7b",
        "generator": "llm:qwen2.5:7b", "template_id": "t", "seed_id": "t#0",
    }  # fmt: skip
    return Sample.model_validate(base | kw)


def desc(session: str, idx: int, tool: str, changed: bool = False, **kw: Any) -> Sample:
    ev = {"direction": "output", "kind": "tool_description", "server": "weather", "tool": tool,
          "output": "d", "output_sha256": "0" * 64, "output_chars": 1,
          "description_changed": changed}  # fmt: skip
    return sample(session, idx, event=ev, **kw)


def test_flagged_sessions_are_split_off_and_written(tmp_path: Path) -> None:
    rows = [
        sample("a", 0), sample("a", 1),
        sample("b", 0, agent_misbehaviour=True), sample("b", 1, agent_misbehaviour=True),
    ]  # fmt: skip
    kept, flagged = split_misbehaviour(rows)
    assert {s.session_id for s in kept} == {"a"} and {s.session_id for s in flagged} == {"b"}
    path = tmp_path / "analysis" / "agent_misbehaviour.jsonl"
    assert write_analysis_file(flagged, path) == 1
    assert len(path.read_text("utf-8").splitlines()) == 2
    assert misbehaviour_summary(flagged)["sessions"] == 1


def test_only_llm_samples_can_carry_the_flag() -> None:
    with pytest.raises(ValidationError, match="only applies to LLM"):
        sample("a", 0, generator="scripted", source="generated", agent_misbehaviour=True)


def test_dedupe_keeps_the_first_listing_and_poisoned_or_changed_ones() -> None:
    poisoned = desc("s", 5, "get_forecast", label="malicious", gold_action="SANITIZE",
                    attack_family="tool_poisoning", session_family="tool_poisoning")  # fmt: skip
    rows = [
        desc("s", 0, "get_forecast"), desc("s", 1, "convert"),
        desc("s", 2, "get_forecast"), desc("s", 3, "convert"),  # repeats: dropped
        sample("s", 4),  # not a description: kept
        poisoned,  # poisoned: kept
        desc("s", 6, "get_forecast", changed=True),  # changed: kept
        desc("other", 0, "get_forecast"),  # another session keeps its own first listing
    ]  # fmt: skip
    out = dedupe_descriptions(rows)
    assert [(s.session_id, s.step_idx) for s in out] == [
        ("s", 0),
        ("s", 1),
        ("s", 4),
        ("s", 5),
        ("s", 6),
        ("other", 0),
    ]


def test_dedupe_removes_the_bulk_of_tool_description_samples() -> None:
    rows = []
    for sess in ("a", "b", "c"):
        rows += [desc(sess, 12 * k + t, f"tool{t}") for k in range(3) for t in range(12)]
        rows += [sample(sess, 100 + k) for k in range(4)]
    assert description_share(rows) >= 0.9
    out = dedupe_descriptions(rows)
    assert description_share(out) < 0.8
    assert len(out) == 3 * (12 + 4)
