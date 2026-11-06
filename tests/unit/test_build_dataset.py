"""data.build_dataset: pipeline, refusals, invariants, dropout, weights, determinism."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data import build_dataset as bd
from data.build_id import MixedBuildError
from data.schema import (
    AttackFamily,
    CompactEvent,
    EventPayload,
    Label,
    Sample,
    SessionContext,
    Split,
)
from gateway.models import Action, Direction, EventKind

BID = "abc123def456+0123456789ab"
F = AttackFamily


def mk(sid: str, split: Split, *, fam: AttackFamily | None = None,
       action: Action | None = None, step: int = 1, history: int = 2, source: str = "generated",
       generator: str = "scripted", build_id: str | None = BID,
       **extra: object) -> Sample:  # fmt: skip
    mal = fam is not None
    gold = action or (Action.BLOCK if mal else Action.ALLOW)
    hist = [
        CompactEvent(direction=Direction.REQUEST, kind=EventKind.TOOL_CALL, tool="fs.read",
                     preview="{}", outcome=Action.ALLOW)
        for _ in range(history)
    ]  # fmt: skip
    return Sample(
        sample_id=f"{sid}-{step}".ljust(16, "0"), session_id=f"sess-{sid}", step_idx=step,
        history=hist, context=SessionContext(request_count=history),
        event=EventPayload(direction=Direction.REQUEST, kind=EventKind.TOOL_CALL, server="fs",
                           tool="read", args={"path": f"/data/{sid}.txt"}),  # fmt: skip
        label=Label.MALICIOUS if mal else Label.BENIGN, gold_action=gold,
        attack_family=fam, session_family=fam,
        attack_goal="goal" if mal else None, source=source, generator=generator,
        template_id=f"tmpl-{sid}", seed_id=f"seed-{sid}", build_id=build_id, split=split,
        **extra,
    )  # fmt: skip


def corpus() -> list[Sample]:
    rows: list[Sample] = []
    for i in range(40):
        rows.append(mk(f"btr{i}", Split.TRAIN))
    for i in range(3):
        rows.append(mk(f"dtr{i}", Split.TRAIN, fam=F.DIRECT_MALICIOUS))
        rows.append(mk(f"itr{i}", Split.TRAIN, fam=F.INDIRECT_INJECTION, action=Action.SANITIZE))
        rows.append(mk(f"xtr{i}", Split.TRAIN, fam=F.MULTI_STEP_EXFIL,
                       action=Action.TERMINATE_SESSION))  # fmt: skip
    rows.append(mk("etr", Split.TRAIN, fam=F.MULTI_STEP_EXFIL, action=Action.REQUIRE_APPROVAL))
    rows.append(mk("rtr", Split.TRAIN, fam=F.DESTRUCTIVE, action=Action.REVOKE_TOOL))
    rows += [mk(f"bva{i}", Split.VAL) for i in range(4)] + [mk("dva", Split.VAL, fam=F.DESTRUCTIVE)]
    rows += [mk(f"bts{i}", Split.TEST_SEEN) for i in range(4)]
    rows.append(mk("dts", Split.TEST_SEEN, fam=F.DIRECT_MALICIOUS))
    rows += [mk(f"bth{i}", Split.TEST_HELDOUT) for i in range(3)]
    rows.append(mk("tp", Split.TEST_HELDOUT, fam=F.TOOL_POISONING, action=Action.SANITIZE))
    rows.append(mk("se", Split.TEST_HELDOUT, fam=F.SOCIAL_ENGINEERING_APPROVAL,
                   action=Action.REQUIRE_APPROVAL))  # fmt: skip
    rows.append(mk("ad", Split.TEST_HELDOUT, fam=F.INDIRECT_INJECTION, action=Action.SANITIZE,
                   source="agentdojo", generator="import:agentdojo"))  # fmt: skip
    return rows


def write(tmp: Path, rows: list[Sample], name: str = "in.jsonl") -> Path:
    p = tmp / name
    p.write_text("".join(s.model_dump_json() + "\n" for s in rows), "utf-8")
    return p


def run(tmp: Path, rows: list[Sample], **kw: object) -> dict[str, Any]:
    return bd.build([write(tmp, rows)], version=1, out_root=tmp / "out", **kw)  # type: ignore[arg-type]


def test_end_to_end_writes_everything(tmp_path: Path) -> None:
    stats = run(tmp_path, corpus())
    out = tmp_path / "out" / "v1"
    for name in ("train", "val", "test_seen", "test_heldout"):
        assert (out / f"{name}.jsonl").exists()
    assert (out / "stats.json").exists() and (out / "DATASET_CARD.md").exists()
    assert (out / "train_weights.json").exists() and (out / "misbehaviour_analysis.jsonl").exists()
    assert stats["build_id"] == BID and len(stats["dataset_checksum"]) == 64
    assert stats["splits"]["test_heldout"]["attack_sessions"] == 3
    card = (out / "DATASET_CARD.md").read_text("utf-8")
    assert BID in card and "AgentDojo" in card and "Known limitations" in card
    loaded = json.loads((out / "stats.json").read_text("utf-8"))
    assert loaded["policy_disagrees"]["train"] == 0 and "family_direction_by_split" in loaded


def test_build_is_deterministic(tmp_path: Path) -> None:
    a = bd.build([write(tmp_path, corpus())], version=1, out_root=tmp_path / "a")
    b = bd.build([write(tmp_path, corpus())], version=1, out_root=tmp_path / "b")
    assert a["dataset_checksum"] == b["dataset_checksum"]


def test_mixed_missing_and_dirty_build_ids_are_refused(tmp_path: Path) -> None:
    rows = corpus()
    with pytest.raises(MixedBuildError):
        run(tmp_path, [*rows, mk("other", Split.TRAIN, build_id="zzz+1")])
    with pytest.raises(MixedBuildError):
        run(tmp_path, [*rows, mk("none", Split.TRAIN, build_id=None)])
    dirty = [s.model_copy(update={"build_id": BID + "-dirty"}) for s in rows]
    with pytest.raises(MixedBuildError):
        run(tmp_path, dirty)
    assert not (tmp_path / "out").exists()  # nothing written on refusal


def test_post_termination_and_misbehaviour_leave_the_splits(tmp_path: Path) -> None:
    rows = corpus() + [
        mk("pt", Split.TRAIN, post_termination=True),
        mk("mis", Split.TRAIN, generator="llm:qwen2.5:7b", agent_misbehaviour=True),
    ]  # fmt: skip
    stats = run(tmp_path, rows)
    out = tmp_path / "out" / "v1"
    ids = "".join((out / f"{n}.jsonl").read_text("utf-8") for n in ("train", "val", "test_seen"))
    assert "sess-pt" not in ids and "sess-mis" not in ids
    assert "sess-mis" in (out / "misbehaviour_analysis.jsonl").read_text("utf-8")
    assert stats["removed"]["post_termination"] == 1
    assert stats["removed"]["agent_misbehaviour_sessions"] == 1


def test_heldout_family_in_train_stops_the_build(tmp_path: Path) -> None:
    bad = [*corpus(), mk("leak", Split.TRAIN, fam=F.TOOL_POISONING, action=Action.SANITIZE)]
    with pytest.raises(bd.BuildError, match="held-out family in train"):
        run(tmp_path, bad)


def test_heldout_source_in_val_stops_the_build(tmp_path: Path) -> None:
    bad = [*corpus(), mk("adv", Split.VAL, source="agentdojo", generator="import:agentdojo")]
    with pytest.raises(bd.BuildError, match="held-out source agentdojo in val"):
        run(tmp_path, bad)


def test_session_in_two_splits_stops_the_build(tmp_path: Path) -> None:
    bad = [*corpus(), mk("btr0", Split.VAL, step=2)]  # same session id as a train session
    with pytest.raises(bd.BuildError, match="is in"):
        run(tmp_path, bad)


def test_every_gold_action_must_appear_in_train(tmp_path: Path) -> None:
    rows = [s for s in corpus() if s.gold_action is not Action.REVOKE_TOOL]
    rows.append(mk("rv", Split.VAL, fam=F.DESTRUCTIVE, action=Action.REVOKE_TOOL))
    with pytest.raises(bd.BuildError, match="never in train.*REVOKE_TOOL"):
        run(tmp_path, rows)


def test_unsplit_samples_are_refused(tmp_path: Path) -> None:
    bad = [*corpus(), mk("ns", Split.TRAIN).model_copy(update={"split": None})]
    with pytest.raises(bd.BuildError, match="without a split"):
        run(tmp_path, bad)


def test_outcome_dropout_hides_outcomes_in_train_only(tmp_path: Path) -> None:
    run(tmp_path, corpus(), history_outcome_dropout=1.0)
    out = tmp_path / "out" / "v1"
    train = bd.load_samples([out / "train.jsonl"])
    val = bd.load_samples([out / "val.jsonl"])
    assert all(h.outcome == "unknown" for s in train for h in s.history)
    assert all(h.outcome == Action.ALLOW for s in val for h in s.history)


def test_outcome_dropout_rate_and_determinism() -> None:
    rows = [mk(f"r{i}", Split.TRAIN) for i in range(400)]
    a = bd.history_outcome_dropout(rows, 0.2, seed=7)
    assert a == bd.history_outcome_dropout(rows, 0.2, seed=7)
    hidden = sum(s.history[0].outcome == "unknown" for s in a) / len(a)
    assert 0.14 < hidden < 0.26
    assert bd.history_outcome_dropout(rows, 0.0, seed=7) == rows
    with pytest.raises(ValueError):
        bd.history_outcome_dropout(rows, 1.5, seed=7)


def test_weights_balance_strata_and_average_one() -> None:
    train = [mk(f"b{i}", Split.TRAIN) for i in range(90)]
    train += [mk(f"a{i}", Split.TRAIN, fam=F.OBFUSCATED) for i in range(10)]
    w, summary = bd.train_weights(train)
    assert abs(sum(w.values()) / len(w) - 1.0) < 1e-3
    assert w[train[-1].sample_id] > w[train[0].sample_id]  # the rare stratum counts more
    assert set(summary["strata"]) == {"benign:tool_call", "malicious:obfuscated"}


def test_weights_are_clipped() -> None:
    train = [mk(f"b{i}", Split.TRAIN) for i in range(1000)] + [
        mk("a", Split.TRAIN, fam=F.OBFUSCATED)
    ]
    w, _ = bd.train_weights(train)
    lo, hi = bd.WEIGHT_CLIP
    assert max(w.values()) / min(w.values()) <= hi / lo + 1e-6


def test_val_and_test_get_no_weights(tmp_path: Path) -> None:
    run(tmp_path, corpus())
    weights = json.loads((tmp_path / "out" / "v1" / "train_weights.json").read_text("utf-8"))
    train = bd.load_samples([tmp_path / "out" / "v1" / "train.jsonl"])
    assert set(weights) == {s.sample_id for s in train}


def test_review_sample_is_stratified_and_deterministic(tmp_path: Path) -> None:
    from data import review

    rows = corpus()
    picked = review.draw(rows, 12)
    assert picked == review.draw(rows, 12) and len(picked) == 12
    strata = {bd.stratum(s) for s in picked}
    assert len(strata) >= 6  # spread over families, not dominated by the 40 benign samples
    assert len(review.draw(rows, 10_000)) == len(rows)
    text = review.render(picked[0], 1)
    assert "Verdict" in text and picked[0].session_id in text
