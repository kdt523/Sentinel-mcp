"""Rule-based attack mutators (Phase 2, step 4).

Each mutator maps a seed attack ``Unit`` to one mutant ``Unit``, or ``None`` when it does not
apply. Mutants:

- keep the seed's family, goal and template (labelling facts are carried over; mutators that
  add a step set that step's facts explicitly, and gold actions are recomputed from
  ``data/labeling.py``);
- record the operation in ``source`` (``mutated:<op>``) and the seed in ``parent``;
- go to the SAME split as their seed (``inherit_splits``).

Two kinds of mutator:

- event mutators change the tool calls the gateway sees (encodings of leaked values or secret
  paths, argument splitting, HTML/markdown hiding, homoglyph / zero-width / synonym changes in
  natural-language arguments);
- prompt mutators change only the user's request (language switch, and the text mutators when
  no argument is natural language). Scripted replay of such a mutant produces the seed's
  events; they matter for LLM-driven sessions (step 7). ``changes_events`` tells them apart.
"""

from __future__ import annotations

import base64
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from data.benign.templates import _SLOT, AttackMark, StepCall, _fill
from data.composer import Unit
from data.schema import Split
from tools_servers.content_gen import ZW

TRANSLATIONS_PATH = Path(__file__).resolve().parents[1] / "attacks" / "translations.yaml"
LANGUAGES = ("es", "hi")

# Bare value refs (no ops yet) that carry sensitive data out of the session.
_BARE_REF = re.compile(r"<<(env|env_b64|api_key|ssh_key|customers_csv|customer_emails)>>")
_MULTILINE_REFS = frozenset({"env", "customers_csv", "customer_emails"})
_SECRET_PATH = re.compile(r"^(?:\./)?(\.env|\.ssh/id_rsa|data/customers\.csv)$")
_CAT = re.compile(r"^cat\s+(\S+)$")
_DELETE_ALL = re.compile(r"^DELETE FROM (\w+)$", re.IGNORECASE)
_TEXT_ARGS = frozenset({"subject", "body"})  # natural-language arguments (email)
_PROTECT = re.compile(
    r"<<[^<>]*>>|https?://\S+|[\w.+-]+@[\w.-]+|\S*[/\\]\S*|\{[a-z_]+\}|\S*\.\w+\S*|'[^']*'"
)
_WORD = re.compile(r"(?<![\w./@<{-])[A-Za-z]{4,}(?![\w./@>}-])")

HOMOGLYPHS: dict[str, str] = {  # Latin -> Cyrillic look-alikes
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "i": "і",
    "A": "А", "C": "С", "E": "Е", "O": "О", "P": "Р", "T": "Т", "H": "Н", "K": "К",
}  # fmt: skip

# Small curated synonym list (whole words, case-preserving on the first letter).
SYNONYMS: dict[str, tuple[str, ...]] = {
    "show": ("display", "print out"), "read": ("look at", "go through"),
    "open": ("pull up", "bring up"), "send": ("forward", "ship"),
    "email": ("mail", "e-mail"), "delete": ("remove", "erase"), "drop": ("remove", "delete"),
    "remove": ("delete", "get rid of"), "list": ("enumerate", "show"),
    "add": ("append", "insert"), "update": ("change", "modify"), "run": ("execute", "launch"),
    "check": ("verify", "confirm"), "summarise": ("sum up", "condense"),
    "please": ("kindly", "pls"), "quickly": ("fast", "right away"), "now": ("immediately",),
    "urgent": ("time-critical", "pressing"), "customers": ("clients",),
    "customer": ("client",), "table": ("db table",), "file": ("document",),
    "key": ("credential",), "everything": ("all of it",), "tell": ("inform",),
    "weather": ("forecast",), "contents": ("content",), "wipe": ("clear", "purge"),
}  # fmt: skip


def _mutant(seed: Unit, op: str, **changes: Any) -> Unit:
    return replace(seed, seed_id=f"{seed.seed_id}~{op}", source=f"mutated:{op}",
                   parent=seed.seed_id, **changes)  # fmt: skip


def _map_args(step: StepCall, fn: Callable[[str, str], str]) -> StepCall:
    return replace(step, args={k: fn(k, v) if isinstance(v, str) else v
                               for k, v in step.args.items()})  # fmt: skip


def _marked_request(step: StepCall) -> bool:
    return step.attack is not None and step.attack.event == "request"


# ----------------------------------------------------------------------------- encodings

_PATH_DECODERS: dict[str, Callable[[str], str]] = {
    "b64": lambda p: ("python3 -c \"import base64; print(open(base64.b64decode("
                      f"'{base64.b64encode(p.encode()).decode()}').decode()).read())\""),
    "hex": lambda p: ("python3 -c \"print(open(bytes.fromhex("
                      f"'{p.encode().hex()}').decode()).read())\""),
    "pct": lambda p: ("python3 -c \"from urllib.parse import unquote; print(open(unquote("
                      f"'{''.join(f'%{b:02X}' for b in p.encode())}')).read())\""),
}  # fmt: skip


def _encode(op: str) -> Callable[[Unit, random.Random], Unit | None]:
    """Encode leaked values (``<<env>>`` -> ``<<env|b64>>``) in marked steps; failing that,
    read a sensitive path through a decoder in the shell instead of naming it."""

    def mutate(seed: Unit, rng: random.Random) -> Unit | None:
        steps, changed = [], False
        for st in seed.steps:
            if st.attack is not None and _BARE_REF.search(str(st.args)):
                st = _map_args(st, lambda _k, v: _BARE_REF.sub(rf"<<\1|{op}>>", v))
                changed = True
            elif _marked_request(st):
                path = None
                if st.tool == "filesystem.read_file":
                    path = st.args.get("path")
                elif st.tool == "shell.run" and (m := _CAT.match(str(st.args.get("cmd", "")))):
                    path = m.group(1)
                if path and _SECRET_PATH.match(path):
                    st = replace(st, tool="shell.run", args={"cmd": _PATH_DECODERS[op](path)})
                    changed = True
            steps.append(st)
        return _mutant(seed, op, steps=tuple(steps)) if changed else None

    return mutate


# ----------------------------------------------------------------------------- splitting


def split_args(seed: Unit, rng: random.Random) -> Unit | None:
    """Spread one malicious argument over two calls: a multi-line leaked value is sent in two
    halves (both leak), or a bulk DELETE becomes two partial DELETEs (the second is a
    repeated destructive attempt on the same tool)."""
    steps: list[StepCall] = []
    changed = False
    for st in seed.steps:
        refs = _BARE_REF.findall(str(st.args)) if st.attack is not None else []
        if not changed and len(refs) == 1 and refs[0] in _MULTILINE_REFS:
            name = refs[0]
            for half in ("half1", "half2"):
                old, new = f"<<{name}>>", f"<<{name}|{half}>>"
                steps.append(replace(st, args={k: v.replace(old, new) if isinstance(v, str)
                                               else v for k, v in st.args.items()}))  # fmt: skip
            changed = True
            continue
        m = _DELETE_ALL.match(str(st.args.get("sql", ""))) if _marked_request(st) else None
        if not changed and m and st.attack is not None and st.attack.family == "destructive":
            t = m.group(1)
            first = replace(st, args={"sql": f"DELETE FROM {t} WHERE rowid % 2 = 0"})
            mark: AttackMark = replace(st.attack, prior_destructive_attempts=(
                st.attack.prior_destructive_attempts + 1))  # fmt: skip
            second = replace(st, args={"sql": f"DELETE FROM {t} WHERE rowid % 2 = 1"},
                             attack=mark)  # fmt: skip
            steps += [first, second]
            changed = True
            continue
        steps.append(st)
    return _mutant(seed, "split_args", steps=tuple(steps)) if changed else None


# ----------------------------------------------------------------------------- hiding


def md_hide(seed: Unit, rng: random.Random) -> Unit | None:
    """Hide a leaked value inside an HTML comment or markdown reference-link comments in an
    email body (still delivered to the recipient, invisible when rendered)."""
    op = rng.choice(("htmlc", "mdref"))
    steps, changed = [], False
    for st in seed.steps:
        if st.attack is not None and st.tool == "email_send.send":
            body = str(st.args.get("body", ""))
            if _BARE_REF.search(body):
                st = replace(st, args={**st.args, "body": _BARE_REF.sub(rf"<<\1|{op}>>", body)})
                changed = True
        steps.append(st)
    return _mutant(seed, "md_hide", steps=tuple(steps)) if changed else None


# ----------------------------------------------------------------------------- text


def _text_mutator(op: str, word_fn: Callable[[str, random.Random], str | None]):  # type: ignore[no-untyped-def]
    """Apply ``word_fn`` to plain words of the prompt and of natural-language arguments
    (email subject / body without value refs). Identifiers (paths, hosts, emails, slots,
    refs, SQL, commands) are never touched."""

    def text(s: str, rng: random.Random) -> tuple[str, bool]:
        hit = False

        def sub(m: re.Match[str]) -> str:
            nonlocal hit
            new = word_fn(m.group(0), rng)
            if new is None:
                return m.group(0)
            hit = True
            return new

        # Mutate only the gaps between protected spans (refs, URLs, emails, paths, slots).
        parts, last = [], 0
        for m in _PROTECT.finditer(s):
            parts += [_WORD.sub(sub, s[last : m.start()]), m.group(0)]
            last = m.end()
        parts.append(_WORD.sub(sub, s[last:]))
        return "".join(parts), hit

    def mutate(seed: Unit, rng: random.Random) -> Unit | None:
        prompt, changed = text(seed.prompt, rng)
        steps = []
        for st in seed.steps:
            if st.tool == "email_send.send":
                args = dict(st.args)
                for k in _TEXT_ARGS & set(args):
                    if isinstance(args[k], str) and "<<" not in args[k]:
                        args[k], hit = text(args[k], rng)
                        changed |= hit
                st = replace(st, args=args)
            steps.append(st)
        return _mutant(seed, op, prompt=prompt, steps=tuple(steps)) if changed else None

    return mutate


def _homoglyph_word(word: str, rng: random.Random) -> str | None:
    if rng.random() > 0.5:
        return None
    idx = [i for i, ch in enumerate(word) if ch in HOMOGLYPHS]
    if not idx:
        return None
    picks = set(rng.sample(idx, max(1, len(idx) // 2)))
    return "".join(HOMOGLYPHS[ch] if i in picks else ch for i, ch in enumerate(word))


def _zero_width_word(word: str, rng: random.Random) -> str | None:
    if rng.random() > 0.5:
        return None
    cut = rng.randrange(1, len(word))
    return word[:cut] + ZW + word[cut:]


def _synonym_word(word: str, rng: random.Random) -> str | None:
    alts = SYNONYMS.get(word.lower())
    if not alts:
        return None
    new = rng.choice(alts)
    return new[0].upper() + new[1:] if word[0].isupper() else new


# ----------------------------------------------------------------------------- language


def load_translations(path: Path = TRANSLATIONS_PATH) -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = yaml.safe_load(path.read_text("utf-8"))["translations"]
    return data


_TRANSLATIONS = load_translations()


def _language(lang: str) -> Callable[[Unit, random.Random], Unit | None]:
    def mutate(seed: Unit, rng: random.Random) -> Unit | None:
        phrase = _TRANSLATIONS.get(seed.template_id, {}).get(lang)
        if phrase is None:
            return None
        return _mutant(seed, f"lang_{lang}", prompt=_fill(phrase, dict(seed.slots)))

    return mutate


# ----------------------------------------------------------------------------- registry

Mutator = Callable[[Unit, random.Random], Unit | None]
MUTATORS: dict[str, Mutator] = {
    "b64": _encode("b64"),
    "hex": _encode("hex"),
    "pct": _encode("pct"),
    "split_args": split_args,
    "md_hide": md_hide,
    "homoglyph": _text_mutator("homoglyph", _homoglyph_word),
    "zero_width": _text_mutator("zero_width", _zero_width_word),
    "synonym": _text_mutator("synonym", _synonym_word),
    "lang_es": _language("es"),
    "lang_hi": _language("hi"),
}


def mutate(seed: Unit, op: str, *, seed_value: int = 42) -> Unit | None:
    return MUTATORS[op](seed, random.Random(f"{seed_value}:{op}:{seed.seed_id}"))


def mutate_all(seeds: Iterable[Unit], *, ops: Sequence[str] | None = None,
               seed: int = 42) -> list[Unit]:  # fmt: skip
    """Every applicable (seed, op) mutant, in a deterministic order."""
    out = []
    for u in seeds:
        for op in ops or MUTATORS:
            m = mutate(u, op, seed_value=seed)
            if m is not None:
                out.append(m)
    from data.attacks.templates import check_destructive_tags

    check_destructive_tags(out)  # a mutation must not produce an untagged destructive step
    return out


def inherit_splits(mutants: Iterable[Unit], splits: dict[str, Split]) -> dict[str, Split]:
    """Mutants go to the same split as their seed."""
    return {m.seed_id: splits[m.parent] for m in mutants if m.parent is not None}


def changes_events(m: Unit, seeds_by_id: dict[str, Unit]) -> bool:
    return m.parent is not None and m.steps != seeds_by_id[m.parent].steps


def translation_problems(path: Path = TRANSLATIONS_PATH) -> list[str]:
    """Translations must name the same slots and content refs as the template's phrasings."""
    from data.attacks.templates import _CONTENT_REF, load_attack_templates

    problems = []
    data = load_translations(path)
    for t in load_attack_templates():
        tr = data.get(t.id)
        if tr is None or set(tr) != set(LANGUAGES):
            problems.append(f"{t.id}: needs {LANGUAGES}")
            continue
        en = t.phrasings[0]
        want = (set(_SLOT.findall(en)), {m.group(0) for m in _CONTENT_REF.finditer(en)})
        for lang, phrase in tr.items():
            got = (set(_SLOT.findall(phrase)), {m.group(0) for m in _CONTENT_REF.finditer(phrase)})
            step_slots = {s for st in t.steps for s in _SLOT.findall(str(st.args))}
            need = (
                step_slots - set(t.free_params) - {s for s in step_slots if s.startswith("variant")}
            )
            if not need <= got[0] or got[1] != want[1] or not got[0] <= set(t.params):
                problems.append(f"{t.id}/{lang}: slots {sorted(got[0])} refs {sorted(got[1])}")
    problems += [
        f"{k}: no such template" for k in set(data) - {t.id for t in load_attack_templates()}
    ]
    return problems


__all__ = ["MUTATORS", "changes_events", "inherit_splits", "mutate", "mutate_all"]
