"""Recall / FPR report over stored gateway decisions."""

from __future__ import annotations

from typing import Any

from data.policy_recall import recall_report, usable
from data.schema import Sample


def smp(session: str, idx: int, *, mal: str | None, action: str, hard: bool = False,
        **kw: Any) -> Sample:  # fmt: skip
    base: dict[str, Any] = {
        "sample_id": f"{session}-{idx:03d}-xx", "session_id": session, "step_idx": idx,
        "context": {"request_count": 0},
        "event": {"direction": "request", "kind": "tool_call", "server": "sql_db",
                  "tool": "execute", "args": {"sql": "x"}},
        "label": "malicious" if mal else "benign",
        "gold_action": "BLOCK" if mal else "ALLOW",
        "attack_family": mal, "session_family": mal, "hard_negative": hard and not mal,
        "source": "generated", "generator": "scripted", "template_id": "t", "seed_id": "t#0",
        "gateway_decision": {"action": action, "tier": "policy", "risk": 0.5},
    }  # fmt: skip
    return Sample.model_validate(base | kw)


def build() -> list[Sample]:
    rows: list[Sample] = []
    for i in range(40):  # destructive: 30 of 40 sessions caught -> 75%
        rows.append(smp(f"d{i}", 0, mal="destructive", action="BLOCK" if i < 30 else "ALLOW"))
    for i in range(10):  # direct_malicious: 10 of 10 -> 100%
        rows.append(smp(f"m{i}", 0, mal="direct_malicious", action="BLOCK"))
    for i in range(50):  # benign: 5 ordinary false positives, 2 on hard negatives
        rows.append(smp(f"b{i}", 0, mal=None, action="BLOCK" if i < 5 else "ALLOW"))
    for i in range(20):
        rows.append(smp(f"h{i}", 0, mal=None, hard=True, action="BLOCK" if i < 2 else "ALLOW"))
    return rows


def test_recall_macro_overall_and_per_family() -> None:
    r = recall_report(build())
    assert r["per_family"]["destructive"]["recall"] == 0.75
    assert r["per_family"]["direct_malicious"]["recall"] == 1.0
    assert r["macro_recall"]["value"] == 0.875  # mean of 0.75 and 1.0
    assert r["overall_recall"]["value"] == 0.8  # 40 of 50 events
    lo, hi = r["macro_recall"]["ci95"]
    assert lo < 0.875 < hi and lo >= 0.0 and hi <= 1.0
    assert r["per_family"]["direct_malicious"]["low_n"] is True  # 10 sessions < 20
    assert r["per_family"]["destructive"]["low_n"] is False
    assert r["action_accuracy"] == 0.8  # BLOCK on 40 of 50 gold-BLOCK events


def test_false_positive_rate_is_split_into_ordinary_and_hard_negatives() -> None:
    fpr = recall_report(build())["false_positive_rate"]
    assert fpr["ordinary"]["fpr"] == 0.1 and fpr["ordinary"]["events"] == 50
    assert fpr["hard_negative"]["fpr"] == 0.1 and fpr["hard_negative"]["events"] == 20
    assert fpr["all"]["fpr"] == round(7 / 70, 4)
    assert fpr["ordinary"]["ci95"][0] < 0.1 < fpr["ordinary"]["ci95"][1]


def test_excluded_events_do_not_count() -> None:
    rows = build()
    extra = [
        smp("x1", 0, mal="destructive", action="ALLOW", post_termination=True),
        smp("x2", 0, mal=None, action="BLOCK", agent_misbehaviour=True,
            generator="llm:q", source="llm_session:q"),  # a flagged session
        smp("x3", 0, mal=None, action="BLOCK", neutralised_by_tool=True),
    ]  # fmt: skip
    assert not any(usable(s) for s in extra)
    assert recall_report(rows + extra) == recall_report(rows)


def test_report_is_deterministic_and_handles_no_attacks() -> None:
    assert recall_report(build()) == recall_report(build())
    benign_only = [s for s in build() if s.label.value == "benign"]
    r = recall_report(benign_only)
    assert r["malicious_events"] == 0 and "macro_recall" not in r
    assert r["false_positive_rate"]["all"]["events"] == 70
