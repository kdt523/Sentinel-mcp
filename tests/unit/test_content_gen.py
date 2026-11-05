"""Generated poisoned / decoy content: ground truth by construction, split partitions,
held-out technique, decoys, and the disjoint tool-poisoning pool."""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp import Client
from mcp.types import TextContent

from tools_servers.content_gen import (
    DECOYS,
    GOALS,
    HELDOUT_ONLY_TECHNIQUES,
    INSTRUCTIONS,
    NOTE_COVERS,
    SPLITS,
    TECHNIQUES,
    TOOL_POISON_PHRASINGS,
    WEB_COVERS,
    ZW,
    ContentIdError,
    generate_note,
    generate_page,
    instruction_pool,
    make_id,
    poisoned_description,
    split_of,
)

TRAINABLE_TECHNIQUES = [t for t in TECHNIQUES if t not in HELDOUT_ONLY_TECHNIQUES]
ALL_PHRASES = [p for ps in INSTRUCTIONS.values() for p in ps]


def _plain(text: str) -> str:
    """Undo the hiding techniques so the payload can be found in the text."""
    text = html.unescape(text).replace(ZW, "")
    for blob in re.findall(r"<code>([A-Za-z0-9+/=]{16,})</code>", text):
        text += " " + base64.b64decode(blob).decode("utf-8", errors="replace")
    return text


# --- ids and determinism ---------------------------------------------------------------------


def test_generation_is_deterministic_and_carries_ground_truth() -> None:
    iid = make_id("train", "poison", "exfil_env_url", "html_comment", 3)
    a, b = generate_page(iid), generate_page(iid)
    assert a == b
    assert (a.split, a.kind, a.goal, a.technique, a.malicious) == (
        "train", "poison", "exfil_env_url", "html_comment", True,
    )  # fmt: skip
    assert generate_page(make_id("train", "poison", "exfil_env_url", "html_comment", 4)) != a


@pytest.mark.parametrize(
    "bad",
    [
        "train.poison.exfil_env_url.html_comment",  # too few parts
        "dev.poison.exfil_env_url.html_comment.1",  # unknown split
        "train.poison.none.html_comment.1",  # poison needs a goal
        "train.decoy.drop_orders.css_hidden.1",  # decoys have no goal
        "train.poison.steal_all.html_comment.1",  # unknown goal
        "train.poison.exfil_env_url.attribute_injection.1",  # held-out technique outside heldout
        "val.decoy.none.attribute_injection.1",
        "train.poison.exfil_env_url.html_comment.-1",
    ],
)
def test_bad_ids_rejected(bad: str) -> None:
    with pytest.raises(ContentIdError):
        generate_page(bad)


def test_heldout_technique_only_in_test_heldout() -> None:
    assert {"attribute_injection"} == HELDOUT_ONLY_TECHNIQUES
    item = generate_page(make_id("test_heldout", "poison", "drop_orders", "attribute_injection", 0))
    assert item.malicious and re.search(r'(alt|title|data-note)="[^"]*DROP', item.text, re.I)


# --- guardrail (b): partitions by split ---------------------------------------------------------


def test_instruction_pools_partitioned_by_split() -> None:
    for goal in GOALS:
        pools = {sp: set(instruction_pool(goal, sp)) for sp in SPLITS}
        assert all(pools.values()), (goal, {k: len(v) for k, v in pools.items()})
        for a in SPLITS:
            for b in SPLITS:
                if a != b:
                    assert not pools[a] & pools[b]
        assert set().union(*pools.values()) == set(INSTRUCTIONS[goal])


@pytest.mark.parametrize("split", SPLITS)
def test_items_use_only_their_splits_covers_and_phrasings(split: str) -> None:
    web_titles = {t for i, (t, _) in enumerate(WEB_COVERS) if split_of(i) == split}
    note_titles = {t for i, (t, _) in enumerate(NOTE_COVERS) if split_of(i) == split}
    techniques = TECHNIQUES if split == "test_heldout" else TRAINABLE_TECHNIQUES
    for goal in GOALS:
        for tech in techniques:
            for n in range(3):
                page = generate_page(make_id(split, "poison", goal, tech, n))
                note = generate_note(make_id(split, "poison", goal, tech, n))
                assert page.cover_title in web_titles and note.cover_title in note_titles
                assert page.instruction in instruction_pool(goal, split)
                assert note.instruction in instruction_pool(goal, split)


# --- ground truth: poison carries its instruction, decoys carry none ---------------------------


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_poison_contains_its_instruction(technique: str) -> None:
    split = "test_heldout" if technique in HELDOUT_ONLY_TECHNIQUES else "train"
    for goal in GOALS:
        item = generate_page(make_id(split, "poison", goal, technique, 1))
        assert item.instruction and item.instruction in _plain(item.text), (technique, goal)


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_decoys_use_the_technique_but_carry_no_instruction(technique: str) -> None:
    """Guardrail (a): every technique also appears harmlessly."""
    split = "test_heldout" if technique in HELDOUT_ONLY_TECHNIQUES else "train"
    for n in range(8):
        for gen in (generate_page, generate_note):
            item = gen(make_id(split, "decoy", None, technique, n))
            assert not item.malicious and item.goal is None and item.instruction is None
            plain = _plain(item.text)
            assert not any(p in plain for p in ALL_PHRASES)
            assert not re.search(r"exfil-sink|\.env|id_rsa|DROP TABLE|DELETE FROM", plain)
            assert any(d in plain for d in DECOYS[technique]) or technique == "base64_blob"


def test_some_decoys_trip_the_rule_sanitizer_so_hidden_is_not_malicious() -> None:
    """Hidden-but-harmless content that a naive detector flags: the hard negatives that keep
    'hidden' from being learned as a proxy for 'malicious'."""
    from gateway.sanitizer import detect

    flagged = {t for t in TRAINABLE_TECHNIQUES
               if detect(generate_page(make_id("train", "decoy", None, t, 0)).text)}  # fmt: skip
    assert {"html_comment", "css_hidden", "zero_width"} <= flagged


# --- guardrail (c): tool-poisoning phrasings disjoint from injection phrasings -----------------


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def test_tool_poison_pool_shares_no_5gram_with_injection_pool() -> None:
    injection = set().union(*(_ngrams(p) for p in ALL_PHRASES))
    for phrase in TOOL_POISON_PHRASINGS:
        overlap = _ngrams(phrase) & injection
        assert not overlap, (phrase, overlap)
    assert not set(TOOL_POISON_PHRASINGS) & set(ALL_PHRASES)


def test_poisoned_description_variants() -> None:
    from tools_servers.content_gen import CLEAN_FORECAST_DESCRIPTION

    variants = {poisoned_description(i) for i in range(len(TOOL_POISON_PHRASINGS))}
    assert len(variants) == len(TOOL_POISON_PHRASINGS)
    for v in variants:
        assert v.startswith(CLEAN_FORECAST_DESCRIPTION) and v != CLEAN_FORECAST_DESCRIPTION
        assert any(p in v.replace(ZW, "") for p in TOOL_POISON_PHRASINGS)


# --- served by the sandbox ---------------------------------------------------------------------


def test_mockweb_serves_generated_pages() -> None:
    from tools_servers.mockweb.app import create_app

    client = TestClient(create_app())
    iid = make_id("val", "poison", "drop_orders", "css_hidden", 2)
    assert client.get(f"/g/{iid}").text == generate_page(iid).text
    assert client.get("/g/train.poison.drop_orders.attribute_injection.0").status_code == 404
    assert client.get("/g/garbage").status_code == 404
    assert client.get("/docs/getting-started").status_code == 200  # static pages unchanged


async def test_notes_server_serves_generated_notes() -> None:
    from tools_servers.poisoned_output.server import build_server

    iid = make_id("train", "poison", "customers_external_email", "addressed_note", 5)
    async with Client(build_server()) as c:
        ok = await c.call_tool("read_shared_note", {"note_id": iid})
        bad = await c.call_tool("read_shared_note", {"note_id": "nope"})
    text = "".join(x.text for x in ok.content if isinstance(x, TextContent))
    assert text == generate_note(iid).text and bad.is_error


async def test_weather_variant_description(tmp_path: Path) -> None:
    from tools_servers.poisoned_desc.server import build_server

    async with Client(build_server("static", tmp_path / "l", variant=3)) as c:
        tools = {t.name: t.description for t in (await c.list_tools()).tools}
    assert tools["get_forecast"] == poisoned_description(3)
