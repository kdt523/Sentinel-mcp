"""Static fixture ground truth, and the guarantee that no benign step reads a poisoned fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.benign.templates import TemplateError, expand_all, load_templates
from data.fixtures import poisoned_pages, static_poison, static_technique


def test_manifest_poisoned_pages_are_known() -> None:
    pages = poisoned_pages()
    assert {"recipes/weeknight-dal", "docs/integration-faq", "news/market-update",
            "blog/remote-work-tips"} == set(pages)  # fmt: skip
    assert all(pages.values())  # each has a hiding technique


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("http_fetch.fetch", {"url": "http://mockweb/recipes/weeknight-dal"}, "page"),
        ("http_fetch.fetch", {"url": "http://mockweb/docs/integration-faq?x=1"}, "page"),
        ("http_fetch.fetch", {"url": "http://mockweb/docs/getting-started"}, None),
        ("http_fetch.fetch", {"url": "http://mockweb/blog/understanding-prompt-injection"}, None),
        ("notes.get_ticket", {"ticket_id": "t-2002"}, "ticket"),
        ("notes.get_ticket", {"ticket_id": "T-2001"}, None),
        ("notes.read_shared_note", {"note_id": "n-102"}, "note"),
        ("notes.read_shared_note", {"note_id": "n-101"}, None),
        ("filesystem.read_file", {"path": ".env"}, None),
    ],
)
def test_static_poison(tool: str, args: dict[str, str], expected: str | None) -> None:
    found = static_poison(tool, args)
    assert (found is None) if expected is None else (found is not None and expected in found)


def test_static_technique_comes_from_the_manifest() -> None:
    url = {"url": "http://mockweb/news/market-update"}
    assert static_technique("http_fetch.fetch", url) == "base64_blob"
    assert (
        static_technique("http_fetch.fetch", {"url": "http://mockweb/docs/getting-started"}) is None
    )


def test_no_benign_session_reads_a_poisoned_fixture() -> None:
    sessions = expand_all(seed=42)  # raises if any does
    assert not [s.seed_id for s in sessions for st in s.steps if static_poison(st.tool, st.args)]


def test_the_guard_rejects_a_benign_template_that_fetches_a_poisoned_page(tmp_path: Path) -> None:
    import yaml

    base = next(t for t in load_templates() if t.id == "fetch_page_question")
    bad = base.model_copy(update={
        "id": "poisoned_fetch",
        "params": {**base.params, "page": ["recipes/weeknight-dal"] * 5 + ["docs/integration-faq"]},
    })  # fmt: skip
    path = tmp_path / "templates.yaml"
    path.write_text(yaml.safe_dump({"templates": [bad.model_dump(mode="json")]}), "utf-8")
    with pytest.raises(TemplateError, match="poisoned"):
        expand_all(seed=42, path=path)
