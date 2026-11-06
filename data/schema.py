"""Dataset sample schema (v1). One sample = one event (request or output) in a session,
with the session context the gateway had BEFORE that event.

Ground truth (``label``, ``gold_action``, ``attack_family``) comes from the session
template, or for LLM-driven sessions from applying ``data/labeling.py`` to the events that
actually happened. The gateway's own verdict is stored separately in ``gateway_decision``
as a baseline feature and is never used as a label.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from gateway.models import Action, Direction, EventKind, Tier

SCHEMA_VERSION: Literal["1"] = "1"
OUTPUT_MAX_CHARS = 16_000  # same cap the gateway applies to stored outputs
HISTORY_MAX = 20


class AttackFamily(StrEnum):
    DIRECT_MALICIOUS = "direct_malicious"
    DESTRUCTIVE = "destructive"
    INDIRECT_INJECTION = "indirect_injection"
    TOOL_POISONING = "tool_poisoning"
    MULTI_STEP_EXFIL = "multi_step_exfil"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    OBFUSCATED = "obfuscated"
    SOCIAL_ENGINEERING_APPROVAL = "social_engineering_approval"


HELDOUT_FAMILIES = frozenset({AttackFamily.TOOL_POISONING,
                              AttackFamily.SOCIAL_ENGINEERING_APPROVAL})  # fmt: skip


class Label(StrEnum):
    BENIGN = "benign"
    MALICIOUS = "malicious"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST_SEEN = "test_seen"
    TEST_HELDOUT = "test_heldout"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CompactEvent(_Strict):
    """A previous event in the session, as the guard prompt would see it."""

    direction: Direction
    kind: EventKind
    tool: str  # qualified <server>.<tool>
    preview: str = Field(max_length=400)  # args JSON or output excerpt (sandbox data: FAKE)
    # What the gateway did at capture time (shadow: would-do). "unknown" = hidden by the
    # builder's --history-outcome-dropout (train only), so the guard cannot over-rely on it.
    outcome: Action | Literal["unknown"]


class SessionContext(_Strict):
    """Session state immediately BEFORE the event (as the session tracker held it)."""

    request_count: int = Field(ge=0)
    files_read: list[str] = Field(default_factory=list)
    taint_classes: list[Literal["secret", "pii"]] = Field(default_factory=list)
    taint_sources: list[str] = Field(default_factory=list)  # detector labels, e.g. path:dotenv
    egress_targets: list[str] = Field(default_factory=list)  # e.g. http:mockweb
    revoked_tools: list[str] = Field(default_factory=list)


class EventPayload(_Strict):
    direction: Direction
    kind: EventKind
    server: str
    tool: str
    args: dict[str, Any] | None = None
    output: str | None = Field(default=None, max_length=OUTPUT_MAX_CHARS)
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_chars: int | None = Field(default=None, ge=0)
    description_changed: bool = False

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.direction is Direction.REQUEST:
            if self.args is None or self.output is not None:
                raise ValueError("request payloads carry args and no output")
            if self.kind is EventKind.TOOL_DESCRIPTION:
                raise ValueError("tool_description payloads are outputs")
        elif self.output is None or self.args is not None:
            raise ValueError("output payloads carry output and no args")
        if self.description_changed and self.kind is not EventKind.TOOL_DESCRIPTION:
            raise ValueError("description_changed only applies to tool_description payloads")
        return self

    @property
    def qualified_tool(self) -> str:
        return f"{self.server}.{self.tool}"


class GatewayVerdict(_Strict):
    """What the gateway decided at capture time. A BASELINE FEATURE, never a label."""

    action: Action
    tier: Tier
    rule_id: str | None = None
    risk: float = Field(ge=0.0, le=1.0)
    policy_version: str | None = None  # sha256 of the policy file the gateway ran


class Sample(_Strict):
    schema_version: Literal["1"] = SCHEMA_VERSION
    sample_id: str = Field(min_length=8)  # stable: hash of (session_id, step_idx)
    session_id: str
    step_idx: int = Field(ge=0)
    history: list[CompactEvent] = Field(default_factory=list, max_length=HISTORY_MAX)
    context: SessionContext
    event: EventPayload

    # --- ground truth (data/LABELING.md) ------------------------------------------------
    label: Label
    gold_action: Action
    attack_family: AttackFamily | None = None  # family of THIS event (None if benign)
    session_family: AttackFamily | None = None  # family of the session (drives splits)
    attack_goal: str | None = None  # e.g. "exfiltrate .env to exfil-sink"
    hard_negative: bool = False

    # --- provenance --------------------------------------------------------------------
    source: str = Field(pattern=r"^(generated|agentdojo|injecagent|mutated:[a-z0-9_]+"
                                r"|llm_session:[a-z0-9_.:-]+|llm_attacker:r[0-9]+)$")  # fmt: skip
    generator: str = Field(pattern=r"^(scripted|llm:[a-z0-9_.:-]+|import:[a-z0-9_.-]+)$")
    template_id: str  # template the session came from
    seed_id: str  # mutants share their seed's id -> same split
    verified: bool | None = None  # attack sessions: goal achieved with the gateway OFF
    neutralised_by_tool: bool = False  # injection stripped by the tool itself
    # Output events that returned GENERATED content (poison or decoy): the hiding technique
    # the generator used (ground truth from tools_servers/content_gen, never parsed).
    hiding_technique: str | None = None
    # git commit + composition version that produced this sample (data/build_id.py);
    # the dataset builder refuses to mix build_ids.
    build_id: str | None = None
    gateway_decision: GatewayVerdict | None = None
    # Shadow capture keeps running after a would-be TERMINATE_SESSION; such later events
    # never happen in an enforced deployment and are excluded from every split by default.
    post_termination: bool = False
    # LLM-generated sessions only: the agent did something unrequested that is NOT malicious
    # (e.g. added a column nobody asked for). Events stay benign; the whole session is excluded
    # from attack and false-positive metrics and kept in a separate analysis file.
    agent_misbehaviour: bool = False

    split: Split | None = None  # assigned by the builder

    @model_validator(mode="before")
    @classmethod
    def _drop_derived(cls, data: Any) -> Any:
        # policy_disagrees is serialized for convenience but always RE-DERIVED on load,
        # never trusted from the file.
        if isinstance(data, dict) and "policy_disagrees" in data:
            data = {k: v for k, v in data.items() if k != "policy_disagrees"}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def policy_disagrees(self) -> bool:
        """Gold action differs from what the gateway decided at capture (stats / analysis)."""
        return (
            self.gateway_decision is not None and self.gateway_decision.action != self.gold_action
        )

    @model_validator(mode="after")
    def _ground_truth_consistent(self) -> Self:
        if self.label is Label.BENIGN:
            if self.gold_action is not Action.ALLOW:
                raise ValueError("benign events are always gold ALLOW")
            if self.attack_family is not None:
                raise ValueError("benign events have no attack_family")
        else:
            if self.gold_action is Action.ALLOW:
                raise ValueError("malicious events never have gold ALLOW")
            if self.attack_family is None:
                raise ValueError("malicious events need attack_family and session_family")
            # Scripted sessions only contain planned attacks. An LLM agent can also do something
            # malicious on its own inside an ordinary task (an "incident", e.g. recovering from
            # a SQL error by dropping a table): such events keep session_family None.
            if self.session_family is None and not self.generator.startswith("llm:"):
                raise ValueError("malicious events need attack_family and session_family")
            if self.hard_negative:
                raise ValueError("hard negatives are benign by definition")
        if self.neutralised_by_tool and self.label is Label.MALICIOUS:
            raise ValueError("a neutralised event reached the agent clean: label it benign")
        if self.attack_goal is not None and self.session_family is None:
            raise ValueError("attack_goal belongs to attack sessions")
        if self.agent_misbehaviour and not self.generator.startswith("llm:"):
            raise ValueError("agent_misbehaviour only applies to LLM-generated sessions")
        return self
