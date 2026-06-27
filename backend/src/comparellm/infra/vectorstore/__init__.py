"""Vector store abstraction with in-memory and pgvector implementations."""

from comparellm.infra.vectorstore.base import VectorStore
from comparellm.infra.vectorstore.factory import build_vector_store

__all__ = ["VectorStore", "build_vector_store"]
