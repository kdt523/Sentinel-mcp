"""Runtime configuration for the SentinelMCP gateway.

All tunables (thresholds, model paths, fail modes, shadow mode) live here and are
overridable via environment variables prefixed with ``SENTINEL_`` or a ``.env`` file.
"""

from __future__ import annotations

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
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+psycopg://sentinel:sentinel@localhost:5432/sentinel"

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

    @model_validator(mode="after")
    def _check_thresholds(self) -> Self:
        if self.fast_allow_threshold >= self.fast_block_threshold:
            raise ValueError("fast_allow_threshold must be strictly below fast_block_threshold")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()
