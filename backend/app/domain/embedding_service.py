"""Embedding service: store lifecycle, indexing, search, and cross-model compare.

Computes embeddings via the provider layer and delegates persistence + similarity
search to the configured :class:`VectorStore`. Search types (``similarity``,
``mmr``, ``similarity_score_threshold``) are implemented here so they behave
identically across vector-store backends.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.domain.models import SearchHit
from app.errors import AppError, NotFoundError
from app.infra.vectorstore.base import VectorStore
from app.infra.vectorstore.mmr import maximal_marginal_relevance
from app.logging import get_logger
from app.providers.registry import ProviderRegistry

log = get_logger(__name__)

SearchType = str  # "similarity" | "mmr" | "similarity_score_threshold"


class EmbeddingService:
    """Indexing and retrieval over pluggable vector stores."""

    def __init__(self, registry: ProviderRegistry, store: VectorStore) -> None:
        self._registry = registry
        self._store = store

    # --- Inventory ---
    def list_models(self) -> list[str]:
        return self._registry.embedding_models()

    async def list_stores(self) -> dict[str, str]:
        return await self._store.list_stores()

    # --- Store lifecycle ---
    async def create_store(self, store_id: str, embedding_key: str) -> None:
        if not self._registry.has_embedding(embedding_key):
            raise NotFoundError(f"Embedding model not configured: '{embedding_key}'")
        # Probe the embedding dimension once so backends that need a fixed
        # dimension (pgvector store metadata) have it at creation time.
        embedder = self._registry.get_embedder(embedding_key)
        probe = await embedder.embed(["dimension probe"])
        dim = len(probe[0]) if probe else 0
        await self._store.create_store(store_id, embedding_key, dim)
        log.info("store_created", store_id=store_id, embedding_key=embedding_key, dim=dim)

    async def delete_store(self, store_id: str) -> None:
        await self._store.delete_store(store_id)

    # --- Indexing ---
    async def index(
        self,
        store_id: str,
        contents: list[str],
        metadatas: list[dict[str, object]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        embedding_key = await self._require_embedding_key(store_id)
        embedder = self._registry.get_embedder(embedding_key)
        vectors = await embedder.embed(contents)
        doc_ids = ids or [uuid.uuid4().hex for _ in contents]
        metas = metadatas or [{} for _ in contents]
        stored = await self._store.add(store_id, doc_ids, vectors, contents, metas)
        log.info("documents_indexed", store_id=store_id, count=len(stored))
        return stored

    # --- Retrieval ---
    async def query(
        self,
        store_id: str,
        query: str,
        *,
        k: int = 5,
        with_scores: bool = False,
        search_type: SearchType = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        embedding_key = await self._require_embedding_key(store_id)
        embedder = self._registry.get_embedder(embedding_key)
        query_vector = (await embedder.embed([query]))[0]

        if search_type == "mmr":
            candidates = await self._store.search_candidates(
                store_id, query_vector, max(fetch_k, k)
            )
            vectors = [vector for _, vector in candidates]
            selected = maximal_marginal_relevance(
                query_vector, vectors, k=k, lambda_mult=lambda_mult
            )
            hits = [candidates[i][0] for i in selected]
        elif search_type == "similarity_score_threshold":
            hits = await self._store.search(store_id, query_vector, k)
            if score_threshold is not None:
                hits = [h for h in hits if (h.score or 0.0) >= score_threshold]
        else:  # similarity (default)
            hits = await self._store.search(store_id, query_vector, k)

        if not with_scores:
            hits = [h.model_copy(update={"score": None}) for h in hits]
        return hits

    async def compare(
        self,
        dataset_id: str,
        embedding_keys: list[str],
        query: str,
        *,
        k: int = 5,
        with_scores: bool = False,
        search_type: SearchType = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        score_threshold: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run the same query across multiple embedding models on one dataset."""
        results: dict[str, dict[str, Any]] = {}
        for embedding_key in embedding_keys:
            store_id = f"{dataset_id}::{embedding_key}"
            try:
                hits = await self.query(
                    store_id,
                    query,
                    k=k,
                    with_scores=with_scores,
                    search_type=search_type,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    score_threshold=score_threshold,
                )
                results[embedding_key] = {"items": [hit.model_dump() for hit in hits]}
            except AppError as exc:
                results[embedding_key] = {"items": [], "error": exc.detail}
        return results

    async def _require_embedding_key(self, store_id: str) -> str:
        embedding_key = await self._store.embedding_key(store_id)
        if embedding_key is None:
            raise NotFoundError(f"Vector store '{store_id}' not found")
        return embedding_key
