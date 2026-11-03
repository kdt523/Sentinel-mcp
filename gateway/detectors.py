"""Cheap, deterministic detectors shared by the session tracker, policy engine and sanitizer.

Detectors return LABELS (e.g. ``content:private_key``), never the matched secret text, so
their results are safe to log and to store in session state.
"""

from __future__ import annotations

import hashlib
import re
from base64 import b64decode, urlsafe_b64decode
from collections.abc import Iterable, Iterator, Mapping
from typing import Any
from urllib.parse import unquote_plus, urlsplit

# Paths whose mere access marks the session as having touched secrets.
SECRET_PATH_PATTERNS: dict[str, re.Pattern[str]] = {
    "dotenv": re.compile(r"(^|[/\\])\.env(\.[\w-]+)?$", re.IGNORECASE),
    "ssh_dir": re.compile(r"(^|[/\\])\.ssh([/\\]|$)", re.IGNORECASE),
    "private_key_file": re.compile(r"(^|[/\\])id_(rsa|dsa|ecdsa|ed25519)(\.pub)?$", re.I),
    "key_material": re.compile(r"\.(pem|key|p12|pfx)$", re.IGNORECASE),
    "credentials_file": re.compile(r"(^|[/\\])(credentials|secrets?)(\.\w+)?$", re.IGNORECASE),
}

# Content that looks like secret material.
SECRET_CONTENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    "stripe_style_key": re.compile(r"\b(sk|rk)_(test|live)_[A-Za-z0-9_]{8,}"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "secret_assignment": re.compile(
        r"(?im)^\s*[A-Z0-9_]*(PASSWORD|SECRET|API_KEY|TOKEN)[A-Z0-9_]*\s*[=:]\s*\S+"
    ),
}

_URL_RE = re.compile(r"\bhttps?://[^\s\"'<>]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@([A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+)\b")
_PATH_ARG_KEYS = frozenset({"path", "file", "filename", "filepath", "file_path"})
REDACTED = "[REDACTED]"


def secret_path_labels(path: str) -> set[str]:
    return {f"path:{name}" for name, rx in SECRET_PATH_PATTERNS.items() if rx.search(path)}


def secret_content_labels(text: str) -> set[str]:
    return {f"content:{name}" for name, rx in SECRET_CONTENT_PATTERNS.items() if rx.search(text)}


def redact_secrets(text: str) -> str:
    """Replace secret-looking spans with ``[REDACTED]`` (for logs, history and summaries)."""
    for name, rx in SECRET_CONTENT_PATTERNS.items():
        if name == "secret_assignment":
            text = rx.sub(lambda m: m.group(0).split("=")[0].split(":")[0] + "=" + REDACTED, text)
        else:
            text = rx.sub(REDACTED, text)
    return text


# --- PII -------------------------------------------------------------------------------

# Optional "+", then 10-13 digits with optional single separators (dates/ids are shorter).
_PHONE_RE = re.compile(r"(?<![\w+])\+?\d(?:[ .-]?\d){9,12}(?!\w)")  # never across newlines
PII_MIN_EMAILS = 3  # one contact address in a doc is not "personal data dump"
PII_MIN_PHONES = 2


def pii_labels(text: str) -> set[str]:
    labels: set[str] = set()
    emails = {m.group(0).lower() for m in _EMAIL_RE.finditer(text)}
    if len(emails) >= PII_MIN_EMAILS:
        labels.add("pii:emails")
    if len(set(_PHONE_RE.findall(text))) >= PII_MIN_PHONES:
        labels.add("pii:phones")
    return labels


def taint_class(label: str) -> str:
    """``secret`` for path:/content: labels, ``pii`` for pii: labels."""
    return "pii" if label.startswith("pii:") else "secret"


# --- content fingerprints (taint flow tracking) ------------------------------------------

FP_MIN_LINE = 12
FP_MIN_TOKEN = 8
FP_LONG_TOKEN = 16
FP_HASH_HEX = 32
_TOKEN_SPLIT = re.compile(r"[\s&=?/\\,;:\"'<>()\[\]{}|`]+")
_B64_RE = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}")


def _variants(text: str) -> list[str]:
    out = [text]
    unq = unquote_plus(text)
    if unq != text:
        out.append(unq)
    for blob in _B64_RE.findall(unq):  # one level of base64 smuggling
        padded = blob + "=" * (-len(blob) % 4)
        for dec in (b64decode, urlsafe_b64decode):
            try:
                decoded = dec(padded).decode("utf-8")
            except ValueError:  # includes binascii.Error and UnicodeDecodeError
                continue
            if decoded.isprintable() or "\n" in decoded:
                out.append(decoded)
            break
    return out


def _qualifies(token: str) -> bool:
    if len(token) < FP_MIN_TOKEN:
        return False
    return len(token) >= FP_LONG_TOKEN or "@" in token or any(c.isdigit() for c in token)


def _h(kind: str, value: str) -> str:
    return hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()[:FP_HASH_HEX]


def _tokens(text: str) -> list[str]:
    return [t for t in (raw.strip(".") for raw in _TOKEN_SPLIT.split(text)) if _qualifies(t)]


def _sensitive_line(line: str) -> bool:
    return bool(
        _tokens(line)
        or _EMAIL_RE.search(line)
        or _PHONE_RE.search(line)
        or secret_content_labels(line)
    )


def fingerprints(text: str, *, sensitive_lines_only: bool = False) -> set[str]:
    """Hashes of normalised lines and value-like tokens.

    Store side (tainted outputs): ``sensitive_lines_only=True`` so harmless lines of a
    secret file (comments, headers) never become taint. Match side (later tool args):
    default, every line is hashed and compared.
    """
    hashes: set[str] = set()
    for variant in _variants(text):
        for line in variant.splitlines():
            norm = " ".join(line.split())
            if len(norm) >= FP_MIN_LINE and (not sensitive_lines_only or _sensitive_line(norm)):
                hashes.add(_h("L", norm))
        for token in _tokens(variant):
            hashes.add(_h("T", token.lower() if "@" in token else token))
    return hashes


def fingerprints_in_args(args: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for s in iter_strings(args):
        out |= fingerprints(s)
    return out


def redact_deep(value: Any) -> Any:
    """``redact_secrets`` applied to every string nested in JSON-like data."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, Mapping):
        return {str(k): redact_deep(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact_deep(v) for v in value]
    return value


def iter_strings(value: Any) -> Iterator[str]:
    """Yield every string nested anywhere inside JSON-like tool arguments."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from iter_strings(v)
    elif isinstance(value, Iterable) and not isinstance(value, bytes | bytearray):
        for v in value:
            yield from iter_strings(v)


def paths_in_args(args: Mapping[str, Any]) -> list[str]:
    return [v for k, v in args.items() if k.lower() in _PATH_ARG_KEYS and isinstance(v, str)]


def egress_targets(args: Mapping[str, Any]) -> set[str]:
    """Destinations mentioned in arguments: ``http:<host>`` and ``email:<domain>``."""
    targets: set[str] = set()
    for s in iter_strings(args):
        for url in _URL_RE.findall(s):
            host = (urlsplit(url).hostname or "").lower()
            if host:
                targets.add(f"http:{host}")
        for domain in _EMAIL_RE.findall(s):
            targets.add(f"email:{domain.lower()}")
    return targets
