"""Benign task templates: load, validate and expand into concrete sessions.

Each template expands into 5..20 variations (seeded shuffle of the cartesian product of
its params). A session carries:
- ``steps``  : the tool calls the scripted agent replays (all ground-truth benign/ALLOW)
- ``prompt`` : one of the template's phrasings with its slots filled, for LLM sessions
"""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TEMPLATES_PATH = Path(__file__).resolve().parent / "templates.yaml"
_SLOT = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
MIN_VARIATIONS, MAX_VARIATIONS = 5, 20


class TemplateError(ValueError):
    pass


class StepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    args: dict[str, Any] = Field(default_factory=dict)
    hard_negative: bool | None = None  # None -> inherit the template's flag


class Template(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    category: str
    hard_negative: bool = False
    note: str | None = None  # why a hard negative is legitimate (for reviewers)
    phrasings: list[str] = Field(min_length=3)
    # Params the user would not state (file slugs, ids, flags): the agent may pick them
    # freely, so phrasings need not mention them.
    free_params: list[str] = Field(default_factory=list)
    params: dict[str, list[Any]] = Field(default_factory=dict)
    steps: list[StepSpec] = Field(min_length=1)
    max_variations: int = Field(default=12, ge=MIN_VARIATIONS, le=MAX_VARIATIONS)

    @field_validator("params")
    @classmethod
    def _non_empty(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for k, vals in v.items():
            if not vals:
                raise ValueError(f"param {k!r} has no values")
        return v

    @model_validator(mode="after")
    def _slots_defined(self) -> Self:
        used = {s for p in self.phrasings for s in _SLOT.findall(p)}
        used |= {s for st in self.steps for s in _SLOT.findall(str(st.args))}
        missing = used - set(self.params)
        if missing:
            raise ValueError(f"template {self.id}: undefined slots {sorted(missing)}")
        # Prompt and steps must describe the SAME task: every param that changes the steps
        # must appear in every phrasing (pure "variant*" counters excepted).
        step_slots = {s for st in self.steps for s in _SLOT.findall(str(st.args))}
        step_slots = {
            s for s in step_slots if not s.startswith("variant") and s not in self.free_params
        }
        unknown_free = set(self.free_params) - set(self.params)
        if unknown_free:
            raise ValueError(
                f"template {self.id}: free_params not in params {sorted(unknown_free)}"
            )
        for p in self.phrasings:
            absent = step_slots - set(_SLOT.findall(p))
            if absent:
                raise ValueError(
                    f"template {self.id}: phrasing {p!r} omits {sorted(absent)} used by steps"
                )
        if self.n_combinations < MIN_VARIATIONS:
            raise ValueError(f"template {self.id}: only {self.n_combinations} variations")
        return self

    @property
    def n_combinations(self) -> int:
        n = 1
        for vals in self.params.values():
            n *= len(vals)
        return n


class TemplatesFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    templates: list[Template]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [t.id for t in self.templates]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate template ids: {dupes}")
        return self


@dataclass(frozen=True)
class AttackMark:
    """Labelling facts for a malicious event produced by a step (data/labeling.py)."""

    event: str  # "request" | "output" | "description"
    family: str
    leaked_class: str | None = None  # multi_step_exfil: "secret" | "pii"
    prior_destructive_attempts: int = 0
    description_changed: bool = False
    target_tool: str | None = None  # description events: the poisoned tool (<server>.<tool>)


@dataclass(frozen=True)
class StepCall:
    tool: str  # "<server>.<tool>"; for op="list_tools" the server whose tools are listed
    args: dict[str, Any]
    hard_negative: bool
    op: str = "call"  # "call" | "list_tools"
    attack: AttackMark | None = None


@dataclass(frozen=True)
class SessionSpec:
    session_key: str  # "<template_id>#<variation>"
    template_id: str
    seed_id: str
    category: str
    prompt: str
    steps: tuple[StepCall, ...]


def load_templates(path: Path = TEMPLATES_PATH) -> list[Template]:
    try:
        return TemplatesFile.model_validate(yaml.safe_load(path.read_text("utf-8"))).templates
    except Exception as exc:  # noqa: BLE001 - surface as a template error with the file
        raise TemplateError(f"{path}: {exc}") from exc


def _fill(value: Any, slots: dict[str, Any]) -> Any:
    if isinstance(value, str):
        whole = _SLOT.fullmatch(value)
        if whole:  # the whole value is one slot: keep the param's YAML type (bool, int...)
            return slots[whole.group(1)]
        return _SLOT.sub(lambda m: str(slots[m.group(1)]), value)
    if isinstance(value, dict):
        return {k: _fill(v, slots) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, slots) for v in value]
    return value


def expand(template: Template, *, seed: int) -> list[SessionSpec]:
    rng = random.Random(f"{seed}:{template.id}")
    keys = sorted(template.params)
    combos = list(itertools.product(*(template.params[k] for k in keys)))
    rng.shuffle(combos)
    sessions = []
    for i, combo in enumerate(combos[: template.max_variations]):
        slots = dict(zip(keys, combo, strict=True))
        steps = tuple(
            StepCall(
                tool=st.tool,
                args=_fill(st.args, slots),
                hard_negative=template.hard_negative if st.hard_negative is None
                else st.hard_negative,
            )
            for st in template.steps
        )  # fmt: skip
        sessions.append(SessionSpec(
            session_key=f"{template.id}#{i}", template_id=template.id,
            seed_id=f"{template.id}#{i}", category=template.category,
            prompt=_fill(rng.choice(template.phrasings), slots), steps=steps,
        ))  # fmt: skip
    return sessions


def expand_all(seed: int = 42, path: Path = TEMPLATES_PATH) -> list[SessionSpec]:
    return [s for t in load_templates(path) for s in expand(t, seed=seed)]
