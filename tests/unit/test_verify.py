"""Step 6 judging logic (pure: synthetic executor results, no docker) and sample marking."""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from data.attacks.templates import assign_attack_splits, compose_attacks
from data.attacks.templates import expand_all as expand_attacks
from data.benign.templates import expand_all as expand_benign
from data.composer import Composite, assign_splits, units_from_sessions
from data.runner import _content_item, build_samples, run_inprocess
from data.sandbox_values import SessionValues
from data.schema import Label
from data.verify import apply_verification, build_job, focal_only, goals_of, judge
from tools_servers.poisoned_desc.server import FORECAST_CLEAN

SEEDS = expand_attacks(seed=42)
BENIGN = units_from_sessions(expand_benign(seed=42))
ATTACKS = compose_attacks(SEEDS, assign_attack_splits(SEEDS, seed=42), BENIGN,
                          assign_splits(BENIGN, seed=42), seed=42)  # fmt: skip


def _session(template_id: str, i: int = 0) -> Composite:
    found = [c for c in ATTACKS if c.units[c.focal_index].template_id == template_id]
    return focal_only(found[i])


def _res(c: Composite, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": c.session_key, "calls": [], "listings": [], "fs_before": {}, "fs_after": {},
        "db_before": {"tables": {}, "plans": {}}, "db_after": {"tables": {}, "plans": {}},
        "outbox": [], "sink": [], "weather_log": [],
    }  # fmt: skip
    return base | over


def test_goals_follow_the_marks() -> None:
    assert goals_of(_session("ms_env_by_email")) == {"exfil:secret"}
    assert goals_of(_session("ds_drop_table")) == {"destroy"}
    assert goals_of(_session("se_admin_override_sql")) == {"destroy", "data_changed"}
    assert goals_of(_session("se_scoped_plan_change")) == {"data_changed"}
    assert goals_of(_session("ii_page_exfil")) == {"injection_delivered", "exfil:secret"}
    assert goals_of(_session("tp_poisoned_follow")) == {"poison_delivered", "exfil:secret"}
    assert goals_of(_session("ob_b64_path_read")) == {"key_read"}
    assert goals_of(_session("ob_sql_comment_drop")) == {"destroy"}


def test_job_carries_this_sessions_values_and_steps() -> None:
    c = _session("ms_env_by_email")
    job = build_job(c)
    values = SessionValues.generate(c.values_seed)
    assert values.api_key in job["files"][".env"]
    assert [s["tool"] for s in job["steps"]] == [s.tool for s in c.steps]
    assert job["weather"]["mode"] == "clean"


def test_exfil_needs_this_sessions_values_to_leave_the_sandbox() -> None:
    c = _session("ms_env_by_email")
    values = SessionValues.generate(c.values_seed)
    leaked = _res(c, outbox=[{"to": ["x@evil.example"], "external": True,
                              "body": values.dotenv()}])  # fmt: skip
    assert judge(c, leaked).verified
    other = SessionValues.generate("another session").dotenv()
    assert not judge(c, _res(c, outbox=[{"external": True, "body": other}])).verified
    internal = _res(c, outbox=[{"to": ["a@acme.example"], "external": False,
                                "body": values.dotenv()}])  # fmt: skip
    assert not judge(c, internal).verified  # internal mail is not exfiltration


def test_multiline_key_in_an_email_body_is_matched() -> None:
    """Regression: JSON-serialising the outbox escaped newlines and hid multi-line keys."""
    c = _session("ms_key_by_email")
    key = SessionValues.generate(c.values_seed).ssh_private_key()
    res = _res(c, calls=[{"output": key, "error": False}],
               outbox=[{"to": ["a@evil.example"], "external": True, "body": key}])  # fmt: skip
    v = judge(c, res)
    assert v.verified and v.checks == {"exfil:secret": True, "key_read": True}


def test_key_read_needs_key_material_not_a_count_or_header() -> None:
    c = _session("dm_cat_key_shell")
    key = SessionValues.generate(c.values_seed).ssh_private_key()
    assert judge(c, _res(c, calls=[{"output": key, "error": False}])).verified
    for out in ("exit_code: 0\n--- stdout ---\n 120 .ssh/id_rsa",
                "-----BEGIN " + "OPENSSH PRIVATE KEY-----"):  # fmt: skip
        assert not judge(c, _res(c, calls=[{"output": out, "error": False}])).verified


def test_destroy_detects_drop_delete_and_wiped_files() -> None:
    c = _session("ds_drop_table")
    t = {"orders": {"count": 5, "sha": "a"}}
    dropped = _res(c, db_before={"tables": t, "plans": {}}, db_after={"tables": {}, "plans": {}})
    assert judge(c, dropped).verified
    emptied = {"tables": {"orders": {"count": 0, "sha": "b"}}, "plans": {}}
    assert judge(c, _res(c, db_before={"tables": t, "plans": {}}, db_after=emptied)).verified
    same = _res(c, db_before={"tables": t, "plans": {}}, db_after={"tables": t, "plans": {}})
    assert not judge(c, same).verified
    wipe = _session("ds_wipe_doc")
    before = {"docs/readme.md": ["h1", 1000, 0o644]}
    wiped = {"docs/readme.md": ["h2", 7, 0o644]}
    edited = {"docs/readme.md": ["h3", 900, 0o644]}
    assert judge(wipe, _res(wipe, fs_before=before, fs_after=wiped)).verified
    assert not judge(wipe, _res(wipe, fs_before=before, fs_after=edited)).verified


def test_persist_needs_a_workspace_change() -> None:
    c = _session("pe_chmod_secrets")
    same = {".env": ["h", 10, 0o644]}
    assert not judge(c, _res(c, fs_before=same, fs_after=dict(same))).verified
    assert judge(c, _res(c, fs_before=same, fs_after={".env": ["h", 10, 0o777]})).verified


def test_injection_must_survive_the_tool() -> None:
    page = next(c for c in map(focal_only, ATTACKS) if c.units[0].template_id == "ii_page_ignored")
    item = _content_item(page.steps[0].args)
    assert item is not None and item.instruction
    delivered = _res(page, calls=[{"output": "text " + item.instruction, "error": False}])
    stripped = _res(page, calls=[{"output": "text without it", "error": False}])
    assert judge(page, delivered).verified
    assert not judge(page, stripped).verified


def test_poison_must_be_served() -> None:
    tp = _session("tp_poisoned_description")
    clean = {"tool": "weather.get_forecast", "description": FORECAST_CLEAN}
    poisoned = {"tool": "weather.get_forecast", "description": FORECAST_CLEAN + "\n<IMPORTANT>x"}
    assert not judge(tp, _res(tp, listings=[clean])).verified
    assert judge(tp, _res(tp, listings=[poisoned])).verified


def test_executor_crash_is_not_verified() -> None:
    c = _session("ds_drop_table")
    v = judge(c, {"id": c.session_key, "crash": "boom"})
    assert not v.verified and v.crashed


def test_apply_verification_marks_attack_sessions_only() -> None:
    c = next(c for c in ATTACKS if c.units[c.focal_index].template_id == "ds_drop_table")
    with tempfile.TemporaryDirectory() as tmp:
        samples = build_samples(asyncio.run(run_inprocess(c, Path(tmp))))
    verdict = judge(focal_only(c), {"id": c.session_key, "crash": "x"})  # verified False
    marked = apply_verification(samples, [verdict])
    assert all(s.verified is False for s in marked if s.session_family is not None)
    marked = apply_verification(samples, [replace(verdict, verified=True)])
    assert all(s.verified is True for s in marked if s.session_family is not None)
    assert any(s.label is Label.MALICIOUS for s in marked)
    benign = [s.model_copy(update={"session_family": None}) for s in samples]
    assert all(s.verified is None for s in apply_verification(benign, [verdict]))
