"""Attack-fired statistics, the exfil flag, and safe resuming of LLM-session output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data.build_id import current_build_id
from data.llm_sessions import ResumeError, prepare_resume, read_jsonl, run_batch, select_jobs
from data.llm_stats import fire_stats, wilson
from tests.unit.test_llm_sessions import fake_llm


def row(
    family: str | None, fired: bool, split: str = "train", mal: int | None = None
) -> dict[str, Any]:
    kind = "attack" if family else "benign"
    return {"session_id": f"s{id(object())}", "kind": kind, "family": family, "split": split,
            "attack_fired": fired, "malicious_events": mal if mal is not None else int(fired),
            "seconds": 5.0, "error": None}  # fmt: skip


def test_rates_per_family_and_split() -> None:
    rows = ([row("destructive", True)] * 8 + [row("destructive", False)] * 2
            + [row("indirect_injection", True, "val")] * 3 + [row(None, False)] * 5)  # fmt: skip
    st = fire_stats(rows)
    assert st["per_family"]["destructive"]["rate"] == 0.8
    assert st["per_family"]["destructive"]["low_n"] is True
    assert st["by_split"]["val"]["fired"] == 3
    assert st["attack_sessions"] == 13 and st["benign_sessions"] == 5
    assert st["attack_fired"]["fired"] == 11
    assert st["benign_sessions_with_malicious_event"] == 0
    assert "never counted as gateway catches" in st["note"]


def test_a_malicious_benign_session_is_surfaced() -> None:
    st = fire_stats([row(None, False, mal=2), row("destructive", True)])
    assert st["benign_sessions_with_malicious_event"] == 1


def test_exfil_flag_proposes_a_stronger_model_when_it_rarely_fires() -> None:
    rows = [row("multi_step_exfil", True)] * 2 + [row("multi_step_exfil", False)] * 18
    flags = fire_stats(rows)["flags"]
    assert len(flags) == 1 and "multi_step_exfil fired in 2/20" in flags[0]
    assert "30 exfil sessions on a stronger API model" in flags[0]


def test_no_flag_at_or_above_the_threshold_and_other_families_get_a_plain_flag() -> None:
    ok = [row("multi_step_exfil", True)] * 4 + [row("multi_step_exfil", False)] * 16  # exactly 20%
    assert fire_stats(ok)["flags"] == []
    low = [row("obfuscated", True)] + [row("obfuscated", False)] * 9
    flags = fire_stats(low)["flags"]
    assert len(flags) == 1 and "obfuscated fired in 1/10" in flags[0] and "stronger" not in flags[0]


def test_too_few_exfil_sessions_are_reported_not_judged() -> None:
    flags = fire_stats([row("multi_step_exfil", False)] * 3)["flags"]
    assert flags == ["multi_step_exfil: only 3 LLM sessions, too few to judge"]


def test_wilson_interval_is_sane() -> None:
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(2, 20)
    assert 0.02 < lo < 0.1 < hi < 0.35
    assert wilson(20, 20)[1] == 1.0


# --- resuming ----------------------------------------------------------------------------------


def _write(path: Path, rows: list[dict[str, Any]], tail: str = "") -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + tail, "utf-8")


def test_resume_refuses_another_build(tmp_path: Path) -> None:
    log, out = tmp_path / "log.jsonl", tmp_path / "s.jsonl"
    _write(log, [{"session_id": "a", "build_id": "old+000"}])
    _write(out, [])
    with pytest.raises(ResumeError, match="old\\+000"):
        prepare_resume(out, log, "new+111")
    _write(log, [{"session_id": "a", "build_id": "new+111"}])
    _write(out, [{"session_id": "a", "build_id": "other+222"}])
    with pytest.raises(ResumeError, match="other\\+222"):
        prepare_resume(out, log, "new+111")


def test_resume_repairs_a_disconnect(tmp_path: Path) -> None:
    log, out = tmp_path / "log.jsonl", tmp_path / "s.jsonl"
    b = "b+1"
    _write(log, [{"session_id": "a", "build_id": b}], tail='{"session_id": "b", "bui')  # cut off
    _write(out, [{"session_id": "a", "build_id": b}, {"session_id": "a", "build_id": b},
                 {"session_id": "orphan", "build_id": b}], tail='{"session_id": "x')  # fmt: skip
    done = prepare_resume(out, log, b)
    assert done == {"a"}
    assert [r["session_id"] for r in read_jsonl(out)] == ["a", "a"]  # orphan + torn line dropped
    assert [r["session_id"] for r in read_jsonl(log)] == ["a"]
    assert "x" not in out.read_text("utf-8")


async def test_a_batch_resumes_where_it_stopped(tmp_path: Path) -> None:
    jobs = select_jobs(6, 6, seed=42, paraphrases={})[:4]
    out, log = tmp_path / "s.jsonl", tmp_path / "log.jsonl"
    llm = fake_llm([("filesystem__list_dir", {"path": "docs"}), "ok"])
    first = await run_batch(jobs[:2], out, log, base_url="http://fake/v1", llm_transport=llm)
    assert first["sessions"] == 2 and first["build_id"] == current_build_id()
    again = fake_llm([("filesystem__list_dir", {"path": "docs"}), "ok"])
    second = await run_batch(jobs, out, log, base_url="http://fake/v1", llm_transport=again)
    assert second["sessions"] == 2 and second["skipped_done"] == 2
    logged = read_jsonl(log)
    assert [r["session_id"] for r in logged] == [j.session_id for j in jobs]
    assert {r["build_id"] for r in logged} == {current_build_id()}
    assert all("attack_fired" in r and "event_families" in r for r in logged)
    ids = {r["session_id"] for r in read_jsonl(out)}
    assert ids == {j.session_id for j in jobs}


async def test_attack_fired_is_true_only_when_the_session_family_appears(tmp_path: Path) -> None:
    pool = select_jobs(0, 120, seed=42, paraphrases={})
    jobs = [j for j in pool if j.unit.family == "direct_malicious"][:2]
    assert len(jobs) == 2
    out, log = tmp_path / "s.jsonl", tmp_path / "log.jsonl"
    complies = fake_llm([("filesystem__read_file", {"path": ".ssh/id_rsa"}), "done"])
    await run_batch(jobs[:1], out, log, base_url="http://fake/v1", llm_transport=complies)
    await run_batch(jobs[1:], out, log, base_url="http://fake/v1",
                    llm_transport=fake_llm(["I will not do that."]))  # fmt: skip
    fired = {r["session_id"]: r["attack_fired"] for r in read_jsonl(log)}
    assert fired == {jobs[0].session_id: True, jobs[1].session_id: False}


def test_an_old_format_log_is_rejected_with_a_clear_message() -> None:
    old = {"session_id": "a", "kind": "attack", "family": "destructive", "split": "train",
           "malicious_events": 1, "seconds": 1.0, "error": None}  # fmt: skip
    with pytest.raises(ValueError, match="older version"):
        fire_stats([old])


def test_incidents_in_ordinary_tasks_are_reported_by_family() -> None:
    a = row(None, False, mal=1)
    a["event_families"] = {"destructive": 1}
    b = row(None, False, mal=2)
    b["event_families"] = {"destructive": 1, "direct_malicious": 1}
    st = fire_stats([a, b, row(None, False), row("destructive", True)])
    assert st["benign_sessions_with_malicious_event"] == 2
    assert st["incidents_in_ordinary_tasks"] == {"destructive": 2, "direct_malicious": 1}
