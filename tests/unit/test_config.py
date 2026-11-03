import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from gateway.config import Settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Run from an empty dir so a developer's local .env cannot leak into tests.
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("SENTINEL_"):
            monkeypatch.delenv(key)


def test_safe_defaults() -> None:
    s = Settings()
    assert s.shadow_mode is True
    assert s.default_fail_mode == "closed"
    assert s.admin_api_key is None
    assert s.guard_timeout_ms == 800


def test_env_prefix_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_SHADOW_MODE", "false")
    monkeypatch.setenv("SENTINEL_DEFAULT_FAIL_MODE", "open")
    monkeypatch.setenv("SENTINEL_ADMIN_API_KEY", "fake-key")
    s = Settings()
    assert s.shadow_mode is False
    assert s.default_fail_mode == "open"
    assert s.admin_api_key is not None
    assert s.admin_api_key.get_secret_value() == "fake-key"
    assert "fake-key" not in repr(s)


def test_invalid_fail_mode_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_DEFAULT_FAIL_MODE", "maybe")
    with pytest.raises(ValidationError):
        Settings()


def test_threshold_ordering_enforced() -> None:
    with pytest.raises(ValidationError):
        Settings(fast_allow_threshold=0.9, fast_block_threshold=0.5)


def test_reads_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SENTINEL_LOG_LEVEL=DEBUG\n", encoding="utf-8")
    assert Settings().log_level == "DEBUG"
