"""PostgreSQL + pgvector vector store for production deployments.

Selected via ``VECTOR_BACKEND=pgvector`` and ``DATABASE_URL``. Uses exact cosine
KNN (sequential scan) which is correct and more than adequate for this comparison
workload; an HNSW index can be layered on later without changing the interface.

Embeddings are stored in a single dimensionless ``vector`` column so that stores
bound to different embedding models (and therefore different dimensions) coexist
in one table.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.domain.models import SearchHit
from app.errors import NotFoundError, ValidationError
from app.log import get_logger

log = get_logger(__name__)

_SCHEMA = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS vector_stores (
        store_id      text PRIMARY KEY,
        embedding_key text NOT NULL,
        dim           integer NOT NULL,
        created_at    timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS vector_documents (
        id        text PRIMARY KEY,
        store_id  text NOT NULL REFERENCES vector_stores(store_id) ON DELETE CASCADE,
        content   text NOT NULL,
        metadata  jsonb NOT NULL DEFAULT '{}'::jsonb,
        embedding vector NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_vector_documents_store ON vector_documents(store_id)",
)


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _parse_metadata(raw: Any) -> dict[str, object]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        return json.loads(raw)
    return {}


def _parse_vector(raw: Any) -> list[float]:
    if isinstance(raw, list):
        return [float(x) for x in raw]
    text_value = str(raw).strip().strip("[]")
    if not text_value:
        return []
    return [float(part) for part in text_value.split(",")]


class PgVectorStore:
    """pgvector-backed vector store."""

    def __init__(self, database_url: str) -> None:
        self._engine: AsyncEngine = create_async_engine(database_url, pool_pre_ping=True)
        self._ready = False

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        async with self._engine.begin() as conn:
            for statement in _SCHEMA:
                await conn.execute(text(statement))
        self._ready = True
        log.info("pgvector_schema_ready")

    async def create_store(self, store_id: str, embedding_key: str, dim: int) -> None:
        await self._ensure_schema()
        async with self._engine.begin() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM vector_stores WHERE store_id = :sid"), {"sid": store_id}
            )
            if exists:
                raise ValidationError(f"Store '{store_id}' already exists")
            await conn.execute(
                text(
                    "INSERT INTO vector_stores (store_id, embedding_key, dim) "
                    "VALUES (:sid, :ek, :dim)"
                ),
                {"sid": store_id, "ek": embedding_key, "dim": dim},
            )

    async def delete_store(self, store_id: str) -> None:
        await self._ensure_schema()
        async with self._engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM vector_stores WHERE store_id = :sid"), {"sid": store_id}
            )

    async def store_exists(self, store_id: str) -> bool:
        await self._ensure_schema()
        async with self._engine.connect() as conn:
            result = await conn.scalar(
                text("SELECT 1 FROM vector_stores WHERE store_id = :sid"), {"sid": store_id}
            )
            return bool(result)

    async def list_stores(self) -> dict[str, str]:
        await self._ensure_schema()
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(text("SELECT store_id, embedding_key FROM vector_stores"))
            ).all()
            return {row.store_id: row.embedding_key for row in rows}

    async def embedding_key(self, store_id: str) -> str | None:
        await self._ensure_schema()
        async with self._engine.connect() as conn:
            return await conn.scalar(
                text("SELECT embedding_key FROM vector_stores WHERE store_id = :sid"),
                {"sid": store_id},
            )

    async def _require(self, store_id: str) -> None:
        if not await self.store_exists(store_id):
            raise NotFoundError(f"Vector store '{store_id}' not found")

    async def add(
        self,
        store_id: str,
        ids: list[str],
        vectors: list[list[float]],
        contents: list[str],
        metadatas: list[dict[str, object]],
    ) -> list[str]:
        await self._require(store_id)
        rows = [
            {
                "id": doc_id,
                "sid": store_id,
                "content": content,
                "metadata": json.dumps(metadata),
                "embedding": _vector_literal(vector),
            }
            for doc_id, vector, content, metadata in zip(
                ids, vectors, contents, metadatas, strict=True
            )
        ]
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO vector_documents (id, store_id, content, metadata, embedding) "
                    "VALUES (:id, :sid, :content, CAST(:metadata AS jsonb), "
                    "CAST(:embedding AS vector)) "
                    "ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, "
                    "metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding"
                ),
                rows,
            )
        return ids

    async def search(self, store_id: str, query_vector: list[float], k: int) -> list[SearchHit]:
        await self._require(store_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT content, metadata, "
                        "1 - (embedding <=> CAST(:qvec AS vector)) AS score "
                        "FROM vector_documents WHERE store_id = :sid "
                        "ORDER BY embedding <=> CAST(:qvec AS vector) LIMIT :k"
                    ),
                    {"qvec": _vector_literal(query_vector), "sid": store_id, "k": k},
                )
            ).all()
        return [
            SearchHit(
                page_content=row.content,
                metadata=_parse_metadata(row.metadata),
                score=float(row.score),
            )
            for row in rows
        ]

    async def search_candidates(
        self, store_id: str, query_vector: list[float], fetch_k: int
    ) -> list[tuple[SearchHit, list[float]]]:
        await self._require(store_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT content, metadata, embedding::text AS vec, "
                        "1 - (embedding <=> CAST(:qvec AS vector)) AS score "
                        "FROM vector_documents WHERE store_id = :sid "
                        "ORDER BY embedding <=> CAST(:qvec AS vector) LIMIT :k"
                    ),
                    {"qvec": _vector_literal(query_vector), "sid": store_id, "k": fetch_k},
                )
            ).all()
        return [
            (
                SearchHit(
                    page_content=row.content,
                    metadata=_parse_metadata(row.metadata),
                    score=float(row.score),
                ),
                _parse_vector(row.vec),
            )
            for row in rows
        ]

    async def close(self) -> None:
        await self._engine.dispose()
