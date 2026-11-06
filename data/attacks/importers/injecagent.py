"""InjecAgent adapter (https://github.com/uiuc-kang-lab/InjecAgent, MIT, pinned commit).

Each test case is a user task whose tool returns attacker text. Per case we build two sessions:

* an INJECTED session: the user's benign tool request, then the tool output carrying the
  attacker instruction -> that output is ``indirect_injection`` / ``SANITIZE`` (LABELING.md);
* a BENIGN TWIN: the same request and the same response template with the attacker text replaced
  by a neutral sentence -> benign / ALLOW. The pair isolates the injection from everything else.

Not invented: the attacker's follow-up tool calls have no arguments in the source, so they are
not generated. Tools are not the sandbox's, so ``verified`` stays ``None`` (these sessions cannot
be replayed with the gateway off): they serve detection metrics, never ASR.

Diversity is low (17 user tasks x ~62 attacker instructions x 2 wrappers), so splits are made BY
ATTACKER INSTRUCTION, and the number of cases per (instruction, wrapper) is capped.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from data.build_id import current_build_id
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

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources.yaml"
REPO = HERE.parents[2]
DEFAULT_DIR = REPO / "data" / "raw" / "external" / "injecagent"
PLACEHOLDER = "<Attacker Instruction>"
SPLIT_SHARES = ((Split.TRAIN, 0.75), (Split.VAL, 0.125), (Split.TEST_SEEN, 0.125))
CASE_FILES = (
    ("dh", "base", "test_cases_dh_base.json"), ("dh", "enhanced", "test_cases_dh_enhanced.json"),
    ("ds", "base", "test_cases_ds_base.json"), ("ds", "enhanced", "test_cases_ds_enhanced.json"),
)  # fmt: skip

# Neutral replacements for the attacker text in the benign twin (hand-written).
NEUTRAL = (
    "Works as described, arrived on time.", "Thanks, this looks good to me.",
    "See you at the meeting tomorrow.", "Nothing to add, all fine here.",
    "Good value for the price.", "Please let me know if anything changes.",
    "Reviewed and approved.", "Looks complete, thank you.",
    "Quick note: everything is on schedule.", "No issues found during the check.",
)  # fmt: skip


class ChecksumError(ValueError):
    """A downloaded file does not match the sha256 pinned in sources.yaml."""


def source_info() -> dict[str, Any]:
    info: dict[str, Any] = yaml.safe_load(SOURCES.read_text("utf-8"))["sources"]["injecagent"]
    return info


def verify_files(directory: Path = DEFAULT_DIR) -> None:
    for name, want in source_info()["files"].items():
        got = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if got != want:
            raise ChecksumError(f"{name}: sha256 {got[:12]}... != pinned {want[:12]}...")


def fetch(directory: Path = DEFAULT_DIR) -> None:
    """Download the pinned files (no-op for files already present) and verify them."""
    info = source_info()
    directory.mkdir(parents=True, exist_ok=True)
    for name in info["files"]:
        if not (directory / name).exists():
            url = f"https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/{info['commit']}/data/{name}"
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https host
                (directory / name).write_bytes(resp.read())
    verify_files(directory)


@dataclass(frozen=True)
class Case:
    case_id: str  # e.g. dh:base:17
    harm: str  # dh = direct harm, ds = data stealing
    wrapper: str  # base | enhanced
    group: str  # hash of the attacker instruction: the unit that splits are made on
    attack_type: str
    goal: str
    attacker_tools: tuple[str, ...]
    instruction: str
    user_tool: str
    user_instruction: str
    tool_params: dict[str, Any]
    injected: str  # tool response with the attacker text
    neutral: str  # same response with the placeholder replaced


def _unquote(text: str) -> str:
    t = text.strip()
    return t[1:-1] if len(t) >= 2 and t[0] == t[-1] == '"' else t


def _params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        val = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return {}
    return val if isinstance(val, dict) else {}


def load_cases(directory: Path = DEFAULT_DIR) -> list[Case]:
    cases: list[Case] = []
    for harm, wrapper, name in CASE_FILES:
        for i, row in enumerate(json.loads((directory / name).read_text("utf-8"))):
            tools = row["Attacker Tools"]
            if isinstance(tools, str):
                tools = ast.literal_eval(tools)
            instr = str(row["Attacker Instruction"])
            tmpl = _unquote(str(row["Tool Response Template"]))
            injected = _unquote(str(row["Tool Response"]))
            if instr not in injected:
                raise ValueError(f"{name}[{i}]: attacker text not found in the tool response")
            neutral_text = NEUTRAL[
                int(hashlib.sha256(instr.encode()).hexdigest(), 16) % len(NEUTRAL)
            ]
            if PLACEHOLDER not in tmpl:
                raise ValueError(f"{name}[{i}]: no placeholder in the response template")
            cases.append(Case(
                case_id=f"{harm}:{wrapper}:{i}", harm=harm, wrapper=wrapper,
                group=hashlib.sha256(instr.encode()).hexdigest()[:10],
                attack_type=str(row["Attack Type"]), goal=str(row["Expected Achievements"]),
                attacker_tools=tuple(tools), instruction=instr, user_tool=str(row["User Tool"]),
                user_instruction=str(row["User Instruction"]),
                tool_params=_params(row["Tool Parameters"]), injected=injected,
                neutral=tmpl.replace(PLACEHOLDER, neutral_text),
            ))  # fmt: skip
    return cases


def assign_groups(groups: list[str], *, seed: int = 42) -> dict[str, Split]:
    """Whole attacker instructions go to one split (train / val / test_seen); never held out:
    the held-out source is AgentDojo."""
    order = sorted(set(groups))
    random.Random(f"{seed}:injecagent-groups").shuffle(order)
    out: dict[str, Split] = {}
    cut, acc = [], 0.0
    for sp, share in SPLIT_SHARES:
        acc += share
        cut.append((sp, round(acc * len(order))))
    for i, g in enumerate(order):
        out[g] = next(sp for sp, end in cut if i < end)
    return out


def select(cases: list[Case], *, per_cell: int = 3, seed: int = 42) -> list[Case]:
    """At most ``per_cell`` cases per (attacker instruction, wrapper), chosen deterministically."""
    cells: dict[tuple[str, str], list[Case]] = {}
    for c in cases:
        cells.setdefault((c.group, c.wrapper), []).append(c)
    rng = random.Random(f"{seed}:injecagent-select")
    chosen: list[Case] = []
    for key in sorted(cells):
        pool = sorted(cells[key], key=lambda c: c.case_id)
        chosen += rng.sample(pool, min(per_cell, len(pool)))
    return sorted(chosen, key=lambda c: c.case_id)


def _toolkits(directory: Path) -> dict[str, str]:
    """Full tool name (toolkit + tool, e.g. ``TerminalExecute``) -> toolkit (the server)."""
    out: dict[str, str] = {}
    for kit in json.loads((directory / "tools.json").read_text("utf-8")):
        for tool in kit.get("tools", []):
            out[f"{kit['name_for_model']}{tool['name']}"] = kit["toolkit"]
    return out


def _sample(session: str, idx: int, split: Split, case: Case, event: EventPayload,
            history: list[CompactEvent], *, malicious: bool, build_id: str,
            request_count: int) -> Sample:  # fmt: skip
    return Sample(
        sample_id=hashlib.sha256(f"{session}:{idx}".encode()).hexdigest()[:16],
        session_id=session, step_idx=idx, history=list(history),
        context=SessionContext(request_count=request_count), event=event,
        label=Label.MALICIOUS if malicious else Label.BENIGN,
        gold_action=Action.SANITIZE if malicious else Action.ALLOW,
        attack_family=AttackFamily.INDIRECT_INJECTION if malicious else None,
        session_family=AttackFamily.INDIRECT_INJECTION if malicious else None,
        attack_goal=case.goal if malicious else None,
        source="injecagent", generator="import:injecagent",
        template_id=f"injecagent:{case.harm}:{case.wrapper}:{case.group}",
        seed_id=f"injecagent:{case.group}", verified=None, build_id=build_id, split=split,
    )  # fmt: skip


def to_samples(case: Case, split: Split, toolkits: dict[str, str], *,
               build_id: str) -> list[Sample]:  # fmt: skip
    """Injected session + benign twin for one case (2 samples each)."""
    if case.user_tool not in toolkits:
        raise ValueError(f"{case.case_id}: user tool {case.user_tool!r} is not in tools.json")
    server = toolkits[case.user_tool]
    request = EventPayload(direction=Direction.REQUEST, kind=EventKind.TOOL_CALL, server=server,
                           tool=case.user_tool, args=case.tool_params)  # fmt: skip
    hist = [CompactEvent(direction=Direction.REQUEST, kind=EventKind.TOOL_CALL,
                         tool=f"{server}.{case.user_tool}",
                         preview=json.dumps(case.tool_params, ensure_ascii=False)[:400],
                         outcome="unknown")]  # fmt: skip
    out: list[Sample] = []
    for kind, text, malicious in (("inj", case.injected, True), ("twin", case.neutral, False)):
        session = f"ia-{kind}:{case.case_id}"
        output = EventPayload(
            direction=Direction.OUTPUT, kind=EventKind.TOOL_CALL, server=server,
            tool=case.user_tool, output=text[:16_000],
            output_sha256=hashlib.sha256(text.encode()).hexdigest(), output_chars=len(text),
        )  # fmt: skip
        out.append(_sample(session, 0, split, case, request, [], malicious=False,
                           build_id=build_id, request_count=0))  # fmt: skip
        out.append(_sample(session, 1, split, case, output, hist, malicious=malicious,
                           build_id=build_id, request_count=1))  # fmt: skip
    return out


def build(directory: Path = DEFAULT_DIR, *, seed: int = 42, per_cell: int = 3,
          build_id: str | None = None) -> list[Sample]:  # fmt: skip
    verify_files(directory)
    cases = load_cases(directory)
    splits = assign_groups([c.group for c in cases], seed=seed)
    toolkits = _toolkits(directory)
    bid = build_id or current_build_id()
    samples: list[Sample] = []
    for case in select(cases, per_cell=per_cell, seed=seed):
        samples += to_samples(case, splits[case.group], toolkits, build_id=bid)
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--fetch", action="store_true", help="download the pinned files first")
    ap.add_argument("--per-cell", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR / "samples.jsonl")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.dir)
    samples = build(args.dir, seed=args.seed, per_cell=args.per_cell)
    args.out.write_text("".join(s.model_dump_json() + "\n" for s in samples), "utf-8")
    sessions = {s.session_id for s in samples}
    print(json.dumps({"samples": len(samples), "sessions": len(sessions),
                      "malicious": sum(s.label is Label.MALICIOUS for s in samples),
                      "out": str(args.out)}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
