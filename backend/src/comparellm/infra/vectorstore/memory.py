"""In-memory vector store for development and tests.

Uses NumPy for exact cosine similarity. Not durable and not shared across
processes; selected via ``VECTOR_BACKEND=memory``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np

from comparellm.domain.models import SearchHit
from comparellm.errors import NotFoundError, ValidationError


@dataclass
class _Record:
    id: str
    content: str
    metadata: dict[str, object]
    vector: np.ndarray


@dataclass
class _Store:
    embedding_key: str
    dim: int
    records: dict[str, _Record] = field(default_factory=dict)


def _cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(matrix.shape[0])
    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms[matrix_norms == 0] = 1.0
    return (matrix @ query) / (matrix_norms * query_norm)


class MemoryVectorStore:
    """Process-local vector store."""

    def __init__(self) -> None:
        self._stores: dict[str, _Store] = {}
        self._lock = asyncio.Lock()

    async def create_store(self, store_id: str, embedding_key: str, dim: int) -> None:
        async with self._lock:
            if store_id in self._stores:
                raise ValidationError(f"Store '{store_id}' already exists")
            self._stores[store_id] = _Store(embedding_key=embedding_key, dim=dim)

    async def delete_store(self, store_id: str) -> None:
        async with self._lock:
            self._stores.pop(store_id, None)

    async def store_exists(self, store_id: str) -> bool:
        return store_id in self._stores

    async def list_stores(self) -> dict[str, str]:
        return {sid: store.embedding_key for sid, store in self._stores.items()}

    async def embedding_key(self, store_id: str) -> str | None:
        store = self._stores.get(store_id)
        return store.embedding_key if store else None

    def _require(self, store_id: str) -> _Store:
        store = self._stores.get(store_id)
        if store is None:
            raise NotFoundError(f"Vector store '{store_id}' not found")
        return store

    async def add(
        self,
        store_id: str,
        ids: list[str],
        vectors: list[list[float]],
        contents: list[str],
        metadatas: list[dict[str, object]],
    ) -> list[str]:
        async with self._lock:
            store = self._require(store_id)
            for doc_id, vector, content, metadata in zip(
                ids, vectors, contents, metadatas, strict=True
            ):
                store.records[doc_id] = _Record(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    vector=np.asarray(vector, dtype=np.float64),
                )
            return ids

    def _ranked(
        self, store_id: str, query_vector: list[float], limit: int
    ) -> list[tuple[_Record, float]]:
        store = self._require(store_id)
        if not store.records:
            return []
        records = list(store.records.values())
        matrix = np.vstack([record.vector for record in records])
        scores = _cosine(np.asarray(query_vector, dtype=np.float64), matrix)
        order = np.argsort(-scores)[:limit]
        return [(records[i], float(scores[i])) for i in order]

    async def search(self, store_id: str, query_vector: list[float], k: int) -> list[SearchHit]:
        return [
            SearchHit(page_content=record.content, metadata=record.metadata, score=score)
            for record, score in self._ranked(store_id, query_vector, k)
        ]

    async def search_candidates(
        self, store_id: str, query_vector: list[float], fetch_k: int
    ) -> list[tuple[SearchHit, list[float]]]:
        return [
            (
                SearchHit(page_content=record.content, metadata=record.metadata, score=score),
                record.vector.tolist(),
            )
            for record, score in self._ranked(store_id, query_vector, fetch_k)
        ]

    async def close(self) -> None:
        self._stores.clear()
