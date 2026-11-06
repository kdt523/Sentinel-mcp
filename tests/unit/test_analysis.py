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


def test_dedupe_per_session_keeps_the_first_listing_and_poisoned_or_changed_ones() -> None:
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
    out = dedupe_descriptions(rows, scope="session")
    assert [(s.session_id, s.step_idx) for s in out] == [
        ("s", 0), ("s", 1), ("s", 4), ("s", 5), ("s", 6), ("other", 0)
    ]  # fmt: skip


def test_dedupe_across_sessions_keeps_one_clean_copy_per_split_and_tool() -> None:
    """Regression: sessions list each tool once, so per-session dedupe removed nothing."""
    rows = []
    for sess in range(6):
        split = "train" if sess < 4 else "val"
        rows += [desc(f"s{sess}", t, f"tool{t}", split=split) for t in range(3)]
    out = dedupe_descriptions(rows)
    assert len(out) == 3 + 3  # one copy of each of 3 tools in train, one in val
    assert {(s.split.value, s.event.tool) for s in out if s.split} == {
        (sp, f"tool{t}") for sp in ("train", "val") for t in range(3)
    }
    assert len(dedupe_descriptions(rows, scope="session")) == len(rows)  # the old, useless way


def test_dedupe_across_sessions_never_thins_poisoned_or_changed_descriptions() -> None:
    clean = [desc(f"s{i}", 0, "get_forecast", split="train") for i in range(5)]
    poisoned = [desc(f"p{i}", 0, "get_forecast", split="train", label="malicious",
                     gold_action="SANITIZE", attack_family="tool_poisoning",
                     session_family="tool_poisoning") for i in range(3)]  # fmt: skip
    changed = [desc(f"c{i}", 1, "get_forecast", changed=True, split="train") for i in range(2)]
    out = dedupe_descriptions(clean + poisoned + changed)
    assert len(out) == 1 + 3 + 2


def test_different_description_text_is_not_a_duplicate() -> None:
    a = desc("a", 0, "get_forecast", split="train")
    b = desc("b", 0, "get_forecast", split="train")
    b = b.model_copy(update={"event": b.event.model_copy(update={"output_sha256": "1" * 64})})
    assert len(dedupe_descriptions([a, b])) == 2


def test_dedupe_removes_the_bulk_of_tool_description_samples() -> None:
    rows = []
    for sess in range(30):
        rows += [desc(f"s{sess}", t, f"tool{t}", split="train") for t in range(12)]
        rows += [sample(f"s{sess}", 100 + k, split="train") for k in range(4)]
    assert description_share(rows) > 0.7
    out = dedupe_descriptions(rows)
    assert len(out) == 12 + 30 * 4
    assert description_share(out) < 0.1
