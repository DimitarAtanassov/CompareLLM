"""Factory that selects the configured session store backend."""

from __future__ import annotations

from app.errors import ConfigurationError
from app.infra.session.base import SessionStore
from app.settings import Settings


def build_session_store(settings: Settings) -> SessionStore:
    """Construct the session store implementation named by ``SESSION_BACKEND``."""
    if settings.session_backend == "memory":
        from app.infra.session.memory import MemorySessionStore

        return MemorySessionStore()

    if settings.session_backend == "redis":
        if not settings.redis_url:
            raise ConfigurationError("SESSION_BACKEND=redis requires REDIS_URL")
        from app.infra.session.redis import RedisSessionStore

        return RedisSessionStore(settings.redis_url, settings.session_ttl_seconds)

    raise ConfigurationError(f"Unknown SESSION_BACKEND: {settings.session_backend}")
