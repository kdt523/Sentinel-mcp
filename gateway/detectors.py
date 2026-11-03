"""Cheap, deterministic detectors shared by the session tracker, policy engine and sanitizer.

Detectors return LABELS (e.g. ``content:private_key``), never the matched secret text, so
their results are safe to log and to store in session state.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any
from urllib.parse import urlsplit

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
