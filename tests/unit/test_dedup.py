"""Cross-split near-duplicate removal."""

from __future__ import annotations

import pytest

from data import dedup
from data.schema import EventPayload, Label, Sample, SessionContext, Split
from gateway.models import Action, Direction, EventKind

LONG = "Quarterly report: revenue grew in every region and the team shipped the new gateway. " * 4


def _s(sid: str, split: Split | None, text: str, *, kind: EventKind = EventKind.TOOL_CALL,
       malicious: bool = False) -> Sample:  # fmt: skip
    from data.schema import AttackFamily

    return Sample(
        sample_id=sid.ljust(16, "0")[:16], session_id=f"sess-{sid}", step_idx=1, history=[],
        context=SessionContext(request_count=1),
        event=EventPayload(direction=Direction.OUTPUT, kind=kind, server="fs", tool="read",
                           output=text),  # fmt: skip
        label=Label.MALICIOUS if malicious else Label.BENIGN,
        gold_action=Action.SANITIZE if malicious else Action.ALLOW,
        attack_family=AttackFamily.INDIRECT_INJECTION if malicious else None,
        session_family=AttackFamily.INDIRECT_INJECTION if malicious else None,
        source="generated", generator="scripted", template_id="t", seed_id=sid, split=split,
    )  # fmt: skip


def test_train_copy_of_an_eval_event_is_dropped_not_the_eval_one() -> None:
    a, b = _s("a", Split.TRAIN, LONG), _s("b", Split.TEST_SEEN, LONG.upper())
    kept, rep = dedup.dedupe_across_splits([a, b])
    assert kept == [b] and rep["dropped_by_split"] == {"train": 1}


def test_priority_heldout_over_seen_over_val_over_train() -> None:
    order = (Split.TRAIN, Split.VAL, Split.TEST_SEEN, Split.TEST_HELDOUT)
    rows = [_s(sp.value[:3], sp, LONG) for sp in order]
    kept, _ = dedup.dedupe_across_splits(rows)
    assert [s.split for s in kept] == [Split.TEST_HELDOUT]


def test_duplicates_inside_one_split_are_kept() -> None:
    rows = [_s("a", Split.TRAIN, LONG), _s("b", Split.TRAIN, LONG)]
    assert len(dedup.dedupe_across_splits(rows)[0]) == 2


def test_different_text_is_kept() -> None:
    other = "Completely different notes about hiring plans and the office move schedule. " * 4
    rows = [_s("a", Split.TRAIN, LONG), _s("b", Split.TEST_SEEN, other)]
    assert len(dedup.dedupe_across_splits(rows)[0]) == 2


def test_short_closed_vocabulary_events_are_exempt() -> None:
    rows = [_s("a", Split.TRAIN, "ok"), _s("b", Split.TEST_SEEN, "ok")]
    assert len(dedup.dedupe_across_splits(rows)[0]) == 2
    assert len(dedup.dedupe_across_splits(rows, min_chars=0)[0]) == 1


def test_tool_descriptions_are_skipped() -> None:
    rows = [_s("a", Split.TRAIN, LONG, kind=EventKind.TOOL_DESCRIPTION),
            _s("b", Split.TEST_SEEN, LONG, kind=EventKind.TOOL_DESCRIPTION)]  # fmt: skip
    assert len(dedup.dedupe_across_splits(rows)[0]) == 2


def test_near_duplicate_with_a_small_edit_is_caught() -> None:
    base = " ".join(f"word{i}" for i in range(150))
    edited = base.replace("word70", "changed", 1)
    kept, _ = dedup.dedupe_across_splits([_s("a", Split.TRAIN, base), _s("b", Split.VAL, edited)])
    assert [s.split for s in kept] == [Split.VAL]


def test_report_counts_malicious_by_family_and_order_is_preserved() -> None:
    rows = [_s("a", Split.TRAIN, LONG, malicious=True), _s("x", Split.TRAIN, "other " * 40),
            _s("b", Split.TEST_SEEN, LONG, malicious=True)]  # fmt: skip
    kept, rep = dedup.dedupe_across_splits(rows)
    assert [s.sample_id[0] for s in kept] == ["x", "b"]
    assert rep["dropped_malicious_by_family"] == {"indirect_injection": 1}


def test_unsplit_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="split"):
        dedup.dedupe_across_splits([_s("a", None, LONG)])
