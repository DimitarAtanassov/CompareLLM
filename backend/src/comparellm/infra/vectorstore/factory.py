"""Factory that selects the configured vector store backend."""

from __future__ import annotations

from comparellm.errors import ConfigurationError
from comparellm.infra.vectorstore.base import VectorStore
from comparellm.settings import Settings


def build_vector_store(settings: Settings) -> VectorStore:
    """Construct the vector store implementation named by ``VECTOR_BACKEND``."""
    if settings.vector_backend == "memory":
        from comparellm.infra.vectorstore.memory import MemoryVectorStore

        return MemoryVectorStore()

    if settings.vector_backend == "pgvector":
        if not settings.database_url:
            raise ConfigurationError("VECTOR_BACKEND=pgvector requires DATABASE_URL")
        from comparellm.infra.vectorstore.pgvector import PgVectorStore

        return PgVectorStore(settings.database_url)

    raise ConfigurationError(f"Unknown VECTOR_BACKEND: {settings.vector_backend}")
