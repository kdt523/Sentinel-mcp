"""Runtime configuration for the SentinelMCP gateway.

All tunables (thresholds, model paths, fail modes, shadow mode) live here and are
overridable via environment variables prefixed with ``SENTINEL_`` or a ``.env`` file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

FailMode = Literal["open", "closed"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Gateway settings loaded from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- storage -------------------------------------------------------------
    redis_url: str = "redis://127.0.0.1:6379/0"
    database_url: str = "postgresql+psycopg://sentinel:sentinel@127.0.0.1:5432/sentinel"

    db_connect_timeout_s: int = Field(default=5, gt=0)
    db_pool_size: int = Field(default=5, ge=1)

    # --- decision log (async, never blocks deciding) ---------------------------
    decision_log_queue_size: int = Field(default=10_000, ge=10)
    decision_log_batch_size: int = Field(default=200, ge=1)
    decision_log_flush_timeout_s: float = Field(default=10.0, gt=0)

    # --- raw capture for dataset building (Phase 2) -----------------------------
    # Writes FULL raw events (unredacted args/outputs) to capture_dir. Off by default and
    # refused unless running in the sandbox (SANDBOX_IN_CONTAINER=1 or sandbox_mode=true).
    capture_mode: bool = False
    capture_dir: Path = Path("data/raw")
    sandbox_mode: bool = Field(
        default=False, description="Explicit flag: this gateway only serves the mock sandbox."
    )

    redis_connect_timeout_s: float = Field(default=2.0, gt=0)
    redis_socket_timeout_s: float = Field(default=2.0, gt=0)

    # --- session tracker -----------------------------------------------------
    session_ttl_s: int = Field(default=86_400, gt=0, description="Idle expiry of session state.")
    session_history_len: int = Field(default=20, ge=1, le=500)
    session_risk_half_life_s: float = Field(
        default=600.0, gt=0, description="Half-life of the decayed cumulative risk."
    )
    session_max_fingerprints: int = Field(
        default=20_000, ge=100, description="Cap on stored taint fingerprints per session."
    )

    # --- policy & enforcement -----------------------------------------------
    policy_path: Path = Path("policies/default.yaml")
    shadow_mode: bool = Field(
        default=True,
        description="Decide and log but never enforce. Safe default for rollout.",
    )
    default_fail_mode: FailMode = Field(
        default="closed",
        description="Applied when deciding errors/times out and the tool has no own fail_mode.",
    )

    output_max_chars: int = Field(
        default=16_000,
        ge=256,
        description="Tool output text kept per event; full text is hashed (sha256).",
    )

    # --- reasoning guard -----------------------------------------------------
    guard_url: str = "http://localhost:8001/v1"
    guard_timeout_ms: int = Field(default=800, gt=0)

    # --- fast tier -----------------------------------------------------------
    fast_tier_model_path: Path = Path("models/fast_tier/model.onnx")
    # Placeholders only: real values are produced by fast_tier/calibrate.py on val.
    fast_allow_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    fast_block_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    # --- admin / ops ---------------------------------------------------------
    admin_api_key: SecretStr | None = Field(
        default=None,
        description="Required for admin endpoints; admin routes are disabled when unset.",
    )
    log_level: LogLevel = "INFO"
    model_version: str = "policy-only-v0"

    @property
    def in_sandbox(self) -> bool:
        return self.sandbox_mode or os.environ.get("SANDBOX_IN_CONTAINER") == "1"

    @model_validator(mode="after")
    def _capture_only_in_sandbox(self) -> Self:
        if self.capture_mode and not self.in_sandbox:
            raise ValueError(
                "capture_mode writes raw unredacted events and is only allowed in the sandbox: "
                "set SENTINEL_SANDBOX_MODE=true or run with SANDBOX_IN_CONTAINER=1"
            )
        return self

    @model_validator(mode="after")
    def _check_thresholds(self) -> Self:
        if self.fast_allow_threshold >= self.fast_block_threshold:
            raise ValueError("fast_allow_threshold must be strictly below fast_block_threshold")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()
