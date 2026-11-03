"""Structured JSON logging with mandatory secret redaction.

Every string value in every log event passes through ``redact_secrets`` before rendering,
so a secret that reaches a log call by mistake is still not written out.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

from gateway.detectors import redact_deep


def redact_processor(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key in list(event_dict):
        event_dict[key] = redact_deep(event_dict[key])
    return event_dict


def configure_logging(level: str = "INFO", *, json: bool = True) -> None:
    """Configure structlog (and stdlib logging) once at process start."""
    lvl = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, stream=sys.stderr, format="%(message)s", force=True)
    renderer: Any = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            redact_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )
