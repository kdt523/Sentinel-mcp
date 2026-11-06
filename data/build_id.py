"""build_id: ties every sample to the code and templates that produced it.

``build_id = <git commit>[-dirty]+<composition version>`` where the composition version is a
sha256 over everything that decides WHICH sessions exist and how they are labelled (templates,
translations, composer, mutators, labelling rules, content generators, destructive check).
Splits and metrics must come from ONE capture: ``require_single_build`` is what
``data.build_dataset`` calls, and it refuses mixed build_ids. Step-6 checks may merge captures.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPOSITION_FILES = (
    "data/benign/templates.yaml",
    "data/benign/templates.py",
    "data/attacks/templates.yaml",
    "data/attacks/templates.py",
    "data/attacks/translations.yaml",
    "data/composer.py",
    "data/generators/mutators.py",
    "data/labeling.py",
    "data/destructive_check.py",
    "data/sandbox_values.py",
    "data/fixtures.py",
    "data/llm_label.py",
    "data/llm_sessions.py",
    "data/benign/paraphrases.json",  # optional: absent until paraphrases are generated
    "tools_servers/content_gen.py",
)


class MixedBuildError(ValueError):
    """Samples from different captures must never be combined into one dataset."""


@cache
def composition_version() -> str:
    h = hashlib.sha256()
    for rel in COMPOSITION_FILES:
        path = REPO / rel
        if not path.exists():  # optional inputs (paraphrases) may not exist yet
            continue
        h.update(rel.encode() + b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)  # noqa: S603, S607
    return out.stdout.strip() if out.returncode == 0 else ""


@cache
def git_commit() -> str:
    commit = _git("rev-parse", "--short=12", "HEAD") or "nogit"
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return f"{commit}-dirty" if dirty else commit


def current_build_id() -> str:
    return f"{git_commit()}+{composition_version()}"


def require_single_build(build_ids: Iterable[str | None]) -> str:
    """The one build_id of a clean capture; refuses mixed or missing ids."""
    ids = set(build_ids)
    if None in ids:
        raise MixedBuildError("samples without a build_id cannot be used to build a dataset")
    if len(ids) != 1:
        raise MixedBuildError(f"mixed build_ids {sorted(i for i in ids if i)}: recapture")
    (only,) = ids
    assert only is not None
    if "-dirty" in only:
        raise MixedBuildError(f"build_id {only} comes from uncommitted code: commit and recapture")
    return only
