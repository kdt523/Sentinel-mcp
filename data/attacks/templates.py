"""Attack task templates: load, validate, expand, assign splits, resolve and compose.

Templates (``templates.yaml``) carry labelling FACTS on malicious steps (``attack`` marks);
gold actions always come from ``data/labeling.py``. Placeholders resolve in two phases:

- content refs ``<<page:KIND:GOAL:TECH>>`` / ``<<note:...>>`` resolve once the unit's split is
  known, to a generated item of THAT split (``tools_servers/content_gen.py`` partitions covers
  and phrasings by split; ``*`` picks any goal / technique allowed in the split);
- value refs (``<<env>>``, ``<<api_key>>`` ...) resolve per composite session from its own
  fake ``SessionValues``.

Split rules: held-out families (tool_poisoning, social_engineering_approval) and
``technique_heldout`` templates go to test_heldout only; all others are spread over
train / val / test_seen with the composer's systematic allocation.
"""

from __future__ import annotations

import base64
import itertools
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import quote

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data.benign.templates import _SLOT, MIN_VARIATIONS, AttackMark, StepCall, TemplateError, _fill
from data.composer import Composite, Unit, assign_splits, compose
from data.labeling import EventFacts, gold_action
from data.sandbox_values import SessionValues
from data.schema import HELDOUT_FAMILIES, AttackFamily, Split
from gateway.models import Action, Direction, EventKind
from tools_servers.content_gen import (
    GOALS,
    HELDOUT_ONLY_TECHNIQUES,
    TECHNIQUES,
    TOOL_POISON_PHRASINGS,
    make_id,
)

TEMPLATES_PATH = Path(__file__).resolve().parent / "templates.yaml"
_CONTENT_REF = re.compile(r"<<(page|note):(poison|decoy):([a-z_*]+):([a-z_*]+)>>")
_VALUE_REF = re.compile(r"<<([a-z_]+)>>")
CONTENT_ITEMS_PER_CELL = 40  # n in <split>.<kind>.<goal>.<technique>.<n>
SEEN_SPLIT_RATIOS = {Split.TRAIN: 0.70, Split.VAL: 0.10, Split.TEST_SEEN: 0.10}


class MarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["request", "output", "description"]
    family: AttackFamily
    leaked_class: Literal["secret", "pii"] | None = None
    prior_destructive_attempts: int = Field(default=0, ge=0)
    description_changed: bool = False
    target_tool: str | None = None

    def facts(self) -> EventFacts:
        return EventFacts(
            family=self.family,
            direction=Direction.REQUEST if self.event == "request" else Direction.OUTPUT,
            kind=EventKind.TOOL_DESCRIPTION if self.event == "description"
            else EventKind.TOOL_CALL,
            leaked_class=self.leaked_class,
            prior_destructive_attempts=self.prior_destructive_attempts,
            description_changed=self.description_changed,
        )  # fmt: skip


class AttackStepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["call", "list_tools"] = "call"
    tool: str = Field(pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?$")
    args: dict[str, Any] = Field(default_factory=dict)
    attack: MarkSpec | None = None

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if (self.op == "call") != ("." in self.tool):
            need = "<server>.<tool>" if self.op == "call" else "<server>"
            raise ValueError(f"{self.op} step needs {need}")
        if self.attack is not None:
            gold_action(self.attack.facts())  # raises LabelingError for undefined combos
            if (self.attack.event == "description") != (self.op == "list_tools"):
                raise ValueError("description events come from list_tools steps")
        return self


class AttackTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    family: AttackFamily
    goal: str
    technique_heldout: bool = False
    phrasings: list[str] = Field(min_length=3)
    free_params: list[str] = Field(default_factory=list)
    params: dict[str, list[Any]] = Field(default_factory=dict)
    upstreams: dict[str, dict[str, str]] = Field(default_factory=dict)
    steps: list[AttackStepSpec] = Field(min_length=1)
    max_variations: int = Field(default=12, ge=MIN_VARIATIONS, le=20)

    @field_validator("params")
    @classmethod
    def _non_empty(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for k, vals in v.items():
            if not vals:
                raise ValueError(f"param {k!r} has no values")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        args_text = [str(st.args) for st in self.steps]
        used = {s for p in self.phrasings for s in _SLOT.findall(p)}
        used |= {s for a in args_text for s in _SLOT.findall(a)}
        if missing := used - set(self.params):
            raise ValueError(f"template {self.id}: undefined slots {sorted(missing)}")
        if unknown := set(self.free_params) - set(self.params):
            raise ValueError(f"template {self.id}: free_params not in params {sorted(unknown)}")
        step_slots = {s for a in args_text for s in _SLOT.findall(a)}
        step_slots = {s for s in step_slots if not s.startswith("variant")} - set(self.free_params)
        step_refs = {m.group(0) for a in args_text for m in _CONTENT_REF.finditer(a)}
        for p in self.phrasings:
            if absent := step_slots - set(_SLOT.findall(p)):
                raise ValueError(f"template {self.id}: phrasing {p!r} omits {sorted(absent)}")
            if absent_refs := step_refs - {m.group(0) for m in _CONTENT_REF.finditer(p)}:
                raise ValueError(f"template {self.id}: phrasing {p!r} omits {sorted(absent_refs)}")
        if not any(st.attack for st in self.steps):
            raise ValueError(f"template {self.id}: no step carries an attack mark")
        techs = [m.group(4) for a in args_text for m in _CONTENT_REF.finditer(a)]
        if self.technique_heldout and not (
            techs and all(t in HELDOUT_ONLY_TECHNIQUES for t in techs)
        ):  # fmt: skip
            raise ValueError(f"template {self.id}: technique_heldout needs held-out techniques")
        n = 1
        for vals in self.params.values():
            n *= len(vals)
        if n < MIN_VARIATIONS:
            raise ValueError(f"template {self.id}: only {n} variations")
        return self

    @property
    def heldout(self) -> bool:
        return self.family in HELDOUT_FAMILIES or self.technique_heldout


class AttackTemplatesFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    templates: list[AttackTemplate]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [t.id for t in self.templates]
        if dupes := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"duplicate template ids: {dupes}")
        return self


def load_attack_templates(path: Path = TEMPLATES_PATH) -> list[AttackTemplate]:
    try:
        data = yaml.safe_load(path.read_text("utf-8"))
        return AttackTemplatesFile.model_validate(data).templates
    except Exception as exc:  # noqa: BLE001 - surface as a template error with the file
        raise TemplateError(f"{path}: {exc}") from exc


# ----------------------------------------------------------------------------- expansion


def _mark(spec: MarkSpec | None) -> AttackMark | None:
    if spec is None:
        return None
    return AttackMark(event=spec.event, family=spec.family.value, leaked_class=spec.leaked_class,
                      prior_destructive_attempts=spec.prior_destructive_attempts,
                      description_changed=spec.description_changed,
                      target_tool=spec.target_tool)  # fmt: skip


def expand(template: AttackTemplate, *, seed: int) -> list[Unit]:
    """Template variations as composer units (content and value refs still unresolved)."""
    rng = random.Random(f"{seed}:{template.id}")
    keys = sorted(template.params)
    combos = list(itertools.product(*(template.params[k] for k in keys)))
    rng.shuffle(combos)
    upstreams = tuple(sorted((srv, tuple(sorted(cfg.items())))
                             for srv, cfg in template.upstreams.items()))  # fmt: skip
    units = []
    for i, combo in enumerate(combos[: template.max_variations]):
        slots = dict(zip(keys, combo, strict=True))
        steps = tuple(
            StepCall(tool=st.tool, args=_fill(st.args, slots), hard_negative=False, op=st.op,
                     attack=_mark(st.attack))
            for st in template.steps
        )  # fmt: skip
        units.append(Unit(
            seed_id=f"{template.id}#{i}", template_id=template.id, category=template.family.value,
            prompt=_fill(rng.choice(template.phrasings), slots), steps=steps, kind="attack",
            family=template.family.value, goal=template.goal, upstreams=upstreams,
        ))  # fmt: skip
    return units


def expand_all(seed: int = 42, path: Path = TEMPLATES_PATH) -> list[Unit]:
    return [u for t in load_attack_templates(path) for u in expand(t, seed=seed)]


def assign_attack_splits(units: Sequence[Unit], *, seed: int,
                         path: Path = TEMPLATES_PATH) -> dict[str, Split]:  # fmt: skip
    heldout = {t.id for t in load_attack_templates(path) if t.heldout}
    out = {u.seed_id: Split.TEST_HELDOUT for u in units if u.template_id in heldout}
    # Stratified by family, so every seen family reaches train, val and test_seen.
    by_family: dict[str | None, list[Unit]] = defaultdict(list)
    for u in units:
        if u.template_id not in heldout:
            by_family[u.family].append(u)
    for fam in sorted(by_family, key=str):
        out |= assign_splits(by_family[fam], seed=seed, ratios=SEEN_SPLIT_RATIOS)
    return out


# ----------------------------------------------------------------------------- resolution


def _resolve_text(text: str, table: dict[str, str]) -> str:
    return _CONTENT_REF.sub(lambda m: table[m.group(0)], text)


def _walk(value: Any, fn: Any) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: _walk(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk(v, fn) for v in value]
    return value


def resolve_content(unit: Unit, split: Split) -> Unit:
    """Bind every content ref (and ``*`` upstream variants) to an item of ``split``."""
    rng = random.Random(f"content:{unit.seed_id}:{split.value}")
    allowed = [t for t in TECHNIQUES
               if split is Split.TEST_HELDOUT or t not in HELDOUT_ONLY_TECHNIQUES]  # fmt: skip
    refs = sorted({m.group(0) for st in unit.steps for m in _CONTENT_REF.finditer(str(st.args))}
                  | {m.group(0) for m in _CONTENT_REF.finditer(unit.prompt)})  # fmt: skip
    table: dict[str, str] = {}
    for ref in refs:
        _, kind, goal, tech = _CONTENT_REF.fullmatch(ref).groups()  # type: ignore[union-attr]
        if goal == "*":
            goal = rng.choice(GOALS) if kind == "poison" else "none"
        if tech == "*":
            tech = rng.choice(allowed)
        table[ref] = make_id(split.value, kind, None if goal == "none" else goal, tech,
                             rng.randrange(CONTENT_ITEMS_PER_CELL))  # fmt: skip
    steps = tuple(replace(st, args=_walk(st.args, lambda s: _resolve_text(s, table)))
                  for st in unit.steps)  # fmt: skip
    upstreams = tuple(
        (srv, tuple((k, str(rng.randrange(len(TOOL_POISON_PHRASINGS))) if k == "variant"
                     and v == "*" else v) for k, v in cfg))
        for srv, cfg in unit.upstreams
    )  # fmt: skip
    return replace(unit, prompt=_resolve_text(unit.prompt, table), steps=steps,
                   upstreams=upstreams)  # fmt: skip


def value_table(values: SessionValues) -> dict[str, str]:
    env = values.dotenv()
    return {
        "env": env,
        "env_b64": base64.b64encode(env.encode()).decode(),
        "api_key": values.api_key,
        "ssh_key": values.ssh_private_key(),
        "customers_csv": values.customers_csv(),
        "customer_email": values.customers[0].email,
        "customer_emails": "\n".join(f"{c.name}, {c.email}" for c in values.customers),
    }


def resolve_values(steps: Sequence[StepCall], values: SessionValues) -> tuple[StepCall, ...]:
    """Fill value refs with this session's fake values (URL-encoded inside ``url`` args)."""
    table = value_table(values)

    def fill(args: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in args.items():
            enc = (lambda s: quote(s, safe="")) if k == "url" else (lambda s: s)
            out[k] = _walk(
                v, lambda s, enc=enc: _VALUE_REF.sub(lambda m: enc(table[m.group(1)]), s)
            )
        return out

    return tuple(replace(st, args=fill(st.args)) for st in steps)


def gold_actions(steps: Sequence[StepCall]) -> list[Action | None]:
    """Gold action of each step's marked event (``None``: unmarked, i.e. benign ALLOW)."""
    out: list[Action | None] = []
    for st in steps:
        m = st.attack
        if m is None:
            out.append(None)
            continue
        out.append(gold_action(EventFacts(
            family=AttackFamily(m.family),
            direction=Direction.REQUEST if m.event == "request" else Direction.OUTPUT,
            kind=EventKind.TOOL_DESCRIPTION if m.event == "description" else EventKind.TOOL_CALL,
            leaked_class=m.leaked_class, prior_destructive_attempts=m.prior_destructive_attempts,
            description_changed=m.description_changed,
        )))  # fmt: skip
    return out


# ----------------------------------------------------------------------------- composing


def compose_attacks(attack_units: Sequence[Unit], attack_splits: dict[str, Split],
                    benign_units: Sequence[Unit], benign_splits: dict[str, Split], *,
                    n_per_unit: int = 1, seed: int) -> list[Composite]:  # fmt: skip
    """Attack composites with the SAME construction as benign ones (guardrail (a)): each
    attack unit is the focal unit among benign fillers of its own split (guardrail (b)).
    test_heldout attacks take fillers from the benign test_heldout pool."""
    rng = random.Random(f"{seed}:attack-composites")
    pools: dict[Split, list[Unit]] = defaultdict(list)
    for u in benign_units:
        pools[benign_splits[u.seed_id]].append(u)
    for sp in pools:
        pools[sp].sort(key=lambda u: u.seed_id)
    out: list[Composite] = []
    for unit in sorted(attack_units, key=lambda u: u.seed_id):
        sp = attack_splits[unit.seed_id]
        focal = resolve_content(unit, sp)
        for j in range(n_per_unit):
            key = f"attack-c:{sp.value}:{unit.seed_id}:{j}"
            c = compose(focal, pools[sp], split=sp, kind="attack", key=key, rng=rng)
            out.append(bind_values(c))
    return out


def bind_values(c: Composite) -> Composite:
    """Resolve value refs in every unit of ``c`` with the composite's own session values."""
    values = SessionValues.generate(c.values_seed)
    units = tuple(replace(u, steps=resolve_values(u.steps, values)) for u in c.units)
    return replace(c, units=units)
