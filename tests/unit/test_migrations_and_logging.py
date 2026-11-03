import json
import os
from argparse import Namespace
from pathlib import Path

import pytest
import structlog
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect

from gateway.config import Settings
from gateway.db.models import Base
from gateway.logging import configure_logging

REPO = Path(__file__).resolve().parents[2]


def _alembic(url: str) -> Config:
    cfg = Config(str(REPO / "alembic.ini"), cmd_opts=Namespace(x=[f"url={url}"]))
    cfg.set_main_option("script_location", str(REPO / "gateway" / "db" / "migrations"))
    return cfg


def test_migration_matches_models_and_downgrades(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    command.upgrade(_alembic(url), "head")
    engine = create_engine(url)
    with engine.connect() as conn:
        tables = set(inspect(conn).get_table_names())
        assert {"sessions", "decisions", "approvals", "feedback"} <= tables
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        assert diff == [], f"models and migration 0001 drifted: {diff}"
    command.downgrade(_alembic(url), "base")
    with engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) <= {"alembic_version"}
    engine.dispose()


# --- capture mode guard ------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("SENTINEL_") or k == "SANDBOX_IN_CONTAINER":
            monkeypatch.delenv(k)
    return monkeypatch


def test_capture_off_by_default(clean_env: pytest.MonkeyPatch) -> None:
    assert Settings().capture_mode is False


def test_capture_refused_outside_sandbox(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("SENTINEL_CAPTURE_MODE", "true")
    with pytest.raises(ValidationError, match="only allowed in the sandbox"):
        Settings()


@pytest.mark.parametrize("env", [{"SENTINEL_SANDBOX_MODE": "true"}, {"SANDBOX_IN_CONTAINER": "1"}])
def test_capture_allowed_in_sandbox(clean_env: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    clean_env.setenv("SENTINEL_CAPTURE_MODE", "true")
    for k, v in env.items():
        clean_env.setenv(k, v)
    s = Settings()
    assert s.capture_mode and s.in_sandbox


# --- logging ------------------------------------------------------------------------------


def test_logs_are_json_and_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    structlog.get_logger("t").info(
        "tool_output", text="FAKE_API_KEY=sk-fake-0000000000000000000000000000SANDBOX",
        nested={"k": ["-----BEGIN " + "OPENSSH PRIVATE KEY-----"]},
    )  # fmt: skip
    line = capsys.readouterr().err.strip().splitlines()[-1]
    data = json.loads(line)
    assert data["event"] == "tool_output" and data["level"] == "info"
    assert "sk-fake" not in line and "PRIVATE KEY" not in line
    assert "[REDACTED]" in data["text"]
    structlog.reset_defaults()
