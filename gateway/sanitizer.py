"""SANITIZE: strip suspected injected instructions from tool OUTPUTS (Phase 1, rule-based).

Each detector targets one hiding technique and returns a label, never the removed text.
Removed spans are replaced with ``[REMOVED BY SENTINEL]``. Detectors are deliberately
narrow: text that merely *talks about* prompt injection must survive unchanged; the
reasoning guard (Phase 4/5) handles subtler cases.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field

MARK = "[REMOVED BY SENTINEL]"

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HIDDEN_STYLE = (
    r"display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:\.0+)?\s*(?:;|\"|')"
    r"|font-size\s*:\s*[01](?:\.\d+)?\s*(?:px|pt)?\s*(?:;|\"|')"
)
_HIDDEN_ELEMENT = re.compile(
    rf"<(\w+)\b[^>]*\bstyle\s*=\s*[\"'][^\"']*(?:{_HIDDEN_STYLE})[^\"']*[\"']?[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_HIDDEN_ATTR_ELEMENT = re.compile(r"<(\w+)\b[^>]*\bhidden\b[^>]*>.*?</\1\s*>", re.DOTALL | re.I)
_FAKE_ROLE_BLOCK = re.compile(
    r"\[(SYSTEM|ADMIN|INST|DEVELOPER)\].*?\[/\1\]|<\|im_start\|>.*?(?:<\|im_end\|>|$)",
    re.DOTALL | re.IGNORECASE,
)
# Instruction-style pseudo-tags used in tool poisoning: <IMPORTANT>...</IMPORTANT>, etc.
_PSEUDO_TAG_BLOCK = re.compile(
    r"<(IMPORTANT|SYSTEM|INSTRUCTIONS?|ADMIN|SECRET|HIDDEN|NOTE_TO_(?:AI|ASSISTANT))>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_FAKE_ROLE_LINE = re.compile(r"(?im)^[ \t]*(?:SYSTEM|ADMIN|DEVELOPER)\s*(?:OVERRIDE)?\s*:.*$")
_MD_COMMENT = re.compile(r"(?m)^[ \t]*\[//\]:\s*#\s*\(.*?\)[ \t]*$")
_ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
# A clause explicitly ADDRESSED to the model ("Note for the assistant ...:",
# "Assistant processing this note: ...", "Automated agents reading this: ...") up to the
# end of its line/paragraph. Mentioning assistants in passing does not match.
_ADDRESSED = re.compile(
    r"(?im)(?:\b(?:note|message|instructions?|reminder|hint)\s+(?:for|to)\s+(?:the\s+)?"
    r"(?:ai\s+)?(?:assistant|agent|model|llm)s?\b[^\n:]{0,60}:"
    r"|\b(?:dear\s+)?(?:ai\s+)?(?:assistant|automated\s+agents?|ai\s+agents?|llm)s?\b"
    r"(?:\s+(?:reading|processing|summari[sz]ing|handling)\b[^\n:]{0,40})?:)"
    r"[^\n]*(?:\n(?!\s*\n)[^\n]*)*"
)
_B64_BLOB = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_INSTRUCTION_WORDS = re.compile(
    r"\b(assistant|ignore|instruction|execute|call|run|send|email|read|delete|drop|reply|"
    r"forward|upload|post)\b",
    re.IGNORECASE,
)


# Detector confidence. HIGH = the text is concealed from humans or explicitly instructs the
# model -> safe to SANITIZE automatically. WEAK = suspicious structure that also occurs in
# legitimate content -> escalate for judgement instead of rewriting automatically.
HIGH = "high"
WEAK = "weak"
CONFIDENCE: dict[str, str] = {
    "zero_width_chars": HIGH,
    "hidden_element": HIGH,
    "addressed_instruction": HIGH,
    "encoded_instruction": HIGH,
    "html_comment": WEAK,
    "markdown_comment": WEAK,
    "fake_role_block": WEAK,
    "pseudo_tag_block": WEAK,
}


@dataclass
class SanitizeResult:
    text: str
    removed: list[str] = field(default_factory=list)  # detector labels, one per removed span

    @property
    def changed(self) -> bool:
        return bool(self.removed)


def _sub(rx: re.Pattern[str], label: str, text: str, removed: list[str]) -> str:
    def repl(_: re.Match[str]) -> str:
        removed.append(label)
        return MARK

    return rx.sub(repl, text)


def _encoded_instruction(blob: str) -> bool:
    try:
        decoded = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    return decoded.isprintable() and len(_INSTRUCTION_WORDS.findall(decoded)) >= 2


def sanitize(text: str) -> SanitizeResult:
    removed: list[str] = []
    if _ZERO_WIDTH.search(text):
        text = _ZERO_WIDTH.sub("", text)
        removed.append("zero_width_chars")
    text = _sub(_HTML_COMMENT, "html_comment", text, removed)
    text = _sub(_HIDDEN_ELEMENT, "hidden_element", text, removed)
    text = _sub(_HIDDEN_ATTR_ELEMENT, "hidden_element", text, removed)
    text = _sub(_FAKE_ROLE_BLOCK, "fake_role_block", text, removed)
    text = _sub(_PSEUDO_TAG_BLOCK, "pseudo_tag_block", text, removed)
    text = _sub(_FAKE_ROLE_LINE, "fake_role_block", text, removed)
    text = _sub(_MD_COMMENT, "markdown_comment", text, removed)

    def blob_repl(m: re.Match[str]) -> str:
        if _encoded_instruction(m.group(0)):
            removed.append("encoded_instruction")
            return MARK
        return m.group(0)

    text = _B64_BLOB.sub(blob_repl, text)
    text = _sub(_ADDRESSED, "addressed_instruction", text, removed)
    return SanitizeResult(text=text, removed=removed)


def detect(text: str) -> dict[str, str]:
    """Detector labels found in ``text`` mapped to their confidence (no rewriting).

    Every detector runs independently on the ORIGINAL text, so an instruction addressed to
    the model inside an HTML comment is reported as ``addressed_instruction`` (high), not
    only as ``html_comment`` (weak).
    """
    checks: dict[str, bool] = {
        "zero_width_chars": bool(_ZERO_WIDTH.search(text)),
        "html_comment": bool(_HTML_COMMENT.search(text)),
        "hidden_element": bool(_HIDDEN_ELEMENT.search(text) or _HIDDEN_ATTR_ELEMENT.search(text)),
        "fake_role_block": bool(_FAKE_ROLE_BLOCK.search(text) or _FAKE_ROLE_LINE.search(text)),
        "pseudo_tag_block": bool(_PSEUDO_TAG_BLOCK.search(text)),
        "markdown_comment": bool(_MD_COMMENT.search(text)),
        "encoded_instruction": any(_encoded_instruction(b) for b in _B64_BLOB.findall(text)),
        "addressed_instruction": bool(_ADDRESSED.search(_ZERO_WIDTH.sub("", text))),
    }
    return {label: CONFIDENCE[label] for label, hit in checks.items() if hit}
