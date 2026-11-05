"""Step 6 on the real executor: gateway-OFF replay inside the Linux image."""

from __future__ import annotations

import pytest

from data.composer import Composite
from data.runner import _composites
from data.verify import build_job, focal_only, judge, run_direct

pytestmark = pytest.mark.integration

ATTACKS = [focal_only(c) for c in _composites(None, 42) if c.kind == "attack"]


def _pick(template_id: str, n: int = 1) -> list[Composite]:
    return [c for c in ATTACKS if c.units[0].template_id == template_id][:n]


def test_direct_runner_verifies_real_attacks_and_rejects_harmless_ones() -> None:
    chosen = (_pick("ms_env_by_email", 2) + _pick("ds_drop_table", 2) + _pick("ds_wipe_doc", 2)
              + _pick("tp_rugpull") + _pick("tp_poisoned_description")
              + _pick("ii_note_customers") + _pick("dm_cat_key_shell", 8))  # fmt: skip
    results = run_direct([build_job(c) for c in chosen])
    verdicts = {c.session_key: judge(c, results[c.session_key]) for c in chosen}
    assert not any(v.crashed for v in verdicts.values())
    by_template: dict[str, list[bool]] = {}
    for c in chosen:
        by_template.setdefault(c.units[0].template_id, []).append(verdicts[c.session_key].verified)
    for tid in ("ms_env_by_email", "ds_drop_table", "ds_wipe_doc", "tp_rugpull",
                "tp_poisoned_description", "ii_note_customers"):  # fmt: skip
        assert all(by_template[tid]), (tid, by_template[tid])
    # cat / head / grep -v disclose the key; wc, grep -c and a footer grep do not
    cmds = {c.session_key: c.steps[0].args["cmd"] for c in chosen
            if c.units[0].template_id == "dm_cat_key_shell"}  # fmt: skip
    for sid, cmd in cmds.items():
        harmless = cmd.startswith(("wc", "grep -c", "grep -n END"))  # counts / footer only
        assert verdicts[sid].verified is not harmless, (cmd, verdicts[sid])


def test_two_sessions_do_not_share_state() -> None:
    a, b = _pick("ds_drop_table", 2)
    ra, rb = (run_direct([build_job(c)])[c.session_key] for c in (a, b))
    assert ra["fs_before"][".env"] != rb["fs_before"][".env"]  # per-session fake secrets
