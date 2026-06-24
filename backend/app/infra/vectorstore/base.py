"""Vector store protocol.

A vector store is a pure vector index: it stores document content + metadata +
embedding vectors keyed by a logical ``store_id`` and answers nearest-neighbour
queries. It is deliberately embedder-agnostic; callers compute embeddings via the
provider layer and pass vectors in. This keeps the store orthogonal to providers.
"""

from __future__ import annotations

from typing import Protocol

from app.domain.models import SearchHit


class VectorStore(Protocol):
    """Persistence interface for embeddings and similarity search."""

    async def create_store(self, store_id: str, embedding_key: str, dim: int) -> None:
        """Register a new store bound to one embedding model. Raises if it exists."""
        ...

    async def delete_store(self, store_id: str) -> None:
        """Delete a store and all of its documents (idempotent)."""
        ...

    async def store_exists(self, store_id: str) -> bool:
        ...

    async def list_stores(self) -> dict[str, str]:
        """Return ``{store_id: embedding_key}`` for all stores."""
        ...

    async def embedding_key(self, store_id: str) -> str | None:
        """Return the embedding key bound to a store, or ``None``."""
        ...

    async def add(
        self,
        store_id: str,
        ids: list[str],
        vectors: list[list[float]],
        contents: list[str],
        metadatas: list[dict[str, object]],
    ) -> list[str]:
        """Upsert documents into a store. Returns the stored ids."""
        ...

    async def search(
        self, store_id: str, query_vector: list[float], k: int
    ) -> list[SearchHit]:
        """Return the top-``k`` hits by cosine similarity (score in ``[-1, 1]``)."""
        ...

    async def search_candidates(
        self, store_id: str, query_vector: list[float], fetch_k: int
    ) -> list[tuple[SearchHit, list[float]]]:
        """Return up to ``fetch_k`` candidates with their vectors, for MMR re-ranking."""
        ...

    async def close(self) -> None:
        """Release any held resources."""
        ...
