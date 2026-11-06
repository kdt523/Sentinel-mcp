"""Paraphrase benign user prompts for LLM-driven sessions (Phase 2 step 7).

A paraphrase must keep the task intact: every identifier-like string of the original prompt
(paths, table names, addresses, urls, numbers, content ids) and every step argument that the
prompt states verbatim must survive unchanged. Rewrites that drop one, copy the original,
change length wildly, add markup, or smuggle attack-style phrasing are rejected and the
original wording is used instead. Results are cached in ``data/benign/paraphrases.json``
(committed: builds must not depend on a live model) with the model that wrote them.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from data.composer import Unit
from data.llm_sessions import PARAPHRASES_PATH, select_jobs

DEFAULT_MODEL = "qwen2.5:3b"
BASE_URL = "http://127.0.0.1:11434/v1"
# Identifier-like: contains a digit, slash, dot-extension, underscore, @ or colon-url.
_IDENT = re.compile(r"[\w.@:/-]*[\d/_@][\w.@:/-]*|[\w-]+\.\w{2,}")
_GIVEAWAY = re.compile(
    r"ignore (all |any )?(previous|prior)|system override|do not tell the user|"
    r"without (asking|permission)|exfiltrat|bypass|jailbreak",
    re.IGNORECASE,
)
_SYSTEM = (
    "You rewrite short requests that a user sends to an office assistant. Keep the meaning "
    "and every file name, table name, e-mail address, URL, identifier and number EXACTLY as "
    "written. Change only the wording and tone. Reply with the rewritten request on one "
    "line, nothing else."
)


_ARG_TOKEN = re.compile(r"[A-Za-z0-9_./@:-]{4,}")
# SQL / argument boilerplate that is not a task detail.
_STOP = frozenset({
    "select", "from", "where", "count", "group", "order", "limit", "join", "like", "desc",
    "insert", "into", "values", "update", "delete", "true", "false", "none", "null", "name",
    "path", "content", "subject", "body", "with", "that", "this", "your", "have", "will",
})  # fmt: skip


def must_keep(prompt: str, unit: Unit) -> list[str]:
    """Strings a paraphrase has to preserve: identifier-like text in the prompt, plus every
    word of the step arguments that the prompt states (a status, product, city, ...)."""
    keep = {m.strip(".,;:!?'\"()") for m in _IDENT.findall(prompt)}
    low = prompt.lower()
    for st in unit.steps:
        for v in st.args.values():
            if not isinstance(v, str):
                continue
            if len(v) >= 3 and v in prompt:
                keep.add(v)
            for tok in _ARG_TOKEN.findall(v):
                tok = tok.strip(".,;:")
                if len(tok) < 4 or tok.lower() in _STOP:
                    continue
                if re.search(rf"(?<![\w]){re.escape(tok.lower())}(?![\w])", low):
                    keep.add(tok)
    return sorted(k for k in keep if len(k) >= 2)


def validate(original: str, candidate: str, keep: list[str]) -> str | None:
    """None if acceptable, else the reason for rejection."""
    text = candidate.strip().strip("`").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text or "\n" in text:
        return "empty or multi-line"
    if text.lower() == original.lower():
        return "identical to the original"
    # a change of case, spelling or one filler word adds no wording diversity
    if difflib.SequenceMatcher(None, original.lower(), text.lower()).ratio() > NEAR_COPY_RATIO:
        return "near-identical to the original"
    if not 0.4 * len(original) <= len(text) <= 2.5 * len(original) + 20:
        return "length out of range"
    missing = [k for k in keep if k.lower() not in text.lower()]
    if missing:
        return f"dropped {missing[:3]}"
    if _GIVEAWAY.search(text) and not _GIVEAWAY.search(original):
        return "attack-style phrasing"
    if re.search(r"[{}<>]|```|\*\*", text) and not re.search(r"[{}<>]|```|\*\*", original):
        return "markup added"
    return None


@dataclass
class Paraphrase:
    seed_id: str
    original: str
    text: str | None
    reason: str | None


class ModelUnavailableError(RuntimeError):
    """The model server did not answer repeatedly: nothing useful can be generated."""


TIMEOUT = httpx.Timeout(connect=10, read=300, write=30, pool=30)
WARMUP_TIMEOUT = httpx.Timeout(connect=10, read=900, write=30, pool=30)  # first load of a model
SAVE_EVERY = 10
MAX_CONSECUTIVE_ERRORS = 5
NEAR_COPY_RATIO = 0.93


async def _chat(client: httpx.AsyncClient, body: dict[str, Any], *, retries: int = 2,
                timeout: httpx.Timeout = TIMEOUT) -> str:  # fmt: skip
    """One chat completion with retries on timeouts / transient HTTP errors."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = await client.post("/chat/completions", json=body, timeout=timeout)
            resp.raise_for_status()
            return str(resp.json()["choices"][0]["message"]["content"])
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            last = exc
            await asyncio.sleep(2 * (attempt + 1))
    raise ModelUnavailableError(f"{type(last).__name__}: {last}"[:200])


async def paraphrase_one(client: httpx.AsyncClient, model: str, unit: Unit, prompt: str,
                         *, tries: int = 3) -> Paraphrase:  # fmt: skip
    keep = must_keep(prompt, unit)
    reason: str | None = "no attempt"
    for attempt in range(tries):
        body = {
            "model": model, "temperature": 0.7 if attempt else 0.4, "max_tokens": 160,
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user", "content": prompt}],
        }  # fmt: skip
        candidate = await _chat(client, body)
        reason = validate(prompt, candidate, keep)
        if reason is None:
            text = candidate.strip().strip("`").strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
                text = text[1:-1].strip()
            return Paraphrase(unit.seed_id, prompt, text, None)
    return Paraphrase(unit.seed_id, prompt, None, reason)


def _save(path: Path, data: dict[str, Any], model: str) -> None:
    data["model"] = model
    data["generated"] = datetime.now(UTC).strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n", "utf-8")
    tmp.replace(path)  # atomic: a crash never leaves a half-written file


async def generate(
    units: list[tuple[Unit, str]],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = BASE_URL,
    path: Path = PARAPHRASES_PATH,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Paraphrase (unit, prompt) pairs and merge the results into the cache file.

    Progress is saved every ``SAVE_EVERY`` prompts and on any failure, so a crash or a
    disconnect loses little. A model that cannot be reached at all raises
    ``ModelUnavailableError`` instead of silently rejecting every prompt."""
    data: dict[str, Any] = {"prompts": {}, "rejected": {}}
    if path.exists():
        data = json.loads(path.read_text("utf-8"))
    # cache hygiene: entries that fail the CURRENT rules are dropped and regenerated
    for unit, prompt in units:
        cached = data["prompts"].get(unit.seed_id)
        if cached is not None:
            why = validate(prompt, cached, must_keep(prompt, unit))
            if why is not None:
                data["prompts"].pop(unit.seed_id)
                data["rejected"][unit.seed_id] = f"cache re-check: {why}"
    todo = [(u, p) for u, p in units if u.seed_id not in data["prompts"]]
    errors = 0
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=TIMEOUT, transport=transport) as c:
            if todo:  # absorbs the one-off model load, which can take minutes on a cold server
                warm = {"model": model, "max_tokens": 1, "temperature": 0,
                        "messages": [{"role": "user", "content": "hi"}]}  # fmt: skip
                await _chat(c, warm, retries=1, timeout=WARMUP_TIMEOUT)
            for i, (unit, prompt) in enumerate(todo, 1):
                try:
                    p = await paraphrase_one(c, model, unit, prompt)
                    errors = 0
                except ModelUnavailableError as exc:
                    errors += 1
                    data["rejected"][unit.seed_id] = f"model error: {exc}"
                    if errors >= MAX_CONSECUTIVE_ERRORS:
                        raise
                    continue
                if p.text is not None:
                    data["prompts"][unit.seed_id] = p.text
                    data["rejected"].pop(unit.seed_id, None)
                else:
                    data["rejected"][unit.seed_id] = p.reason
                if i % SAVE_EVERY == 0:
                    _save(path, data, model)
    finally:
        _save(path, data, model)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--benign", type=int, default=270)
    ap.add_argument("--attack", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    jobs = [j for j in select_jobs(args.benign, args.attack, seed=args.seed, paraphrases={})
            if j.unit.kind != "attack"]  # fmt: skip
    data = asyncio.run(generate([(j.unit, j.unit.prompt) for j in jobs], model=args.model))
    print(json.dumps({"accepted": len(data["prompts"]), "rejected": len(data["rejected"]),
                      "model": data["model"]}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
