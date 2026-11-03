"""Integration-test configuration."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Mapping

import pytest


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]] | None:
    # psycopg's async driver does not support the Windows Proactor loop (Linux/CI unaffected).
    if sys.platform == "win32":
        return {"selector": asyncio.SelectorEventLoop}
    return None
