"""Application settings, loaded from environment variables.

All runtime configuration is centralized here using ``pydantic-settings`` so that
configuration is validated once, typed everywhere, and never read ad-hoc via
``os.getenv`` scattered through the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

VectorBackend = Literal["memory", "pgvector"]
SessionBackend = Literal["memory", "redis"]


class Settings(BaseSettings):
    """Strongly-typed application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "CompareLLM"
    environment: Literal["development", "staging", "production"] = "development"

    # --- HTTP server ---
    host: str = "0.0.0.0"
    port: int = 8080

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True

    # --- Model catalogue ---
    models_config: str = Field(
        default="/config/models.yaml",
        description="Path to the models.yaml provider catalogue.",
    )

    # --- CORS ---
    # NoDecode: keep pydantic-settings from JSON-decoding the env value so the
    # ``_split_origins`` validator can accept a plain comma-separated string.
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )
    cors_allow_origin_regex: str | None = None

    # --- Persistence backends ---
    vector_backend: VectorBackend = "memory"
    session_backend: SessionBackend = "memory"
    database_url: str | None = None
    redis_url: str | None = None
    session_ttl_seconds: int = 60 * 60 * 24 * 7  # 7 days

    # --- Streaming ---
    sse_heartbeat_seconds: float = 10.0

    # --- Provider behaviour ---
    request_timeout_seconds: float = 120.0
    anthropic_default_max_tokens: int = 4096

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string in addition to a list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.upper()


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
