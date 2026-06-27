"""Embeddings endpoints: store lifecycle, indexing, search, and compare.

Request/response shapes are preserved from the previous service so the existing
UI keeps working unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from comparellm.api.deps import EmbeddingServiceDep

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

SearchType = Literal["similarity", "mmr", "similarity_score_threshold"]


class CreateStoreIn(BaseModel):
    store_id: str
    embedding_key: str


class IndexTextsIn(BaseModel):
    store_id: str
    texts: list[str] = Field(..., min_length=1)
    metadatas: list[dict[str, Any]] | None = None
    ids: list[str] | None = None


class DocIn(BaseModel):
    page_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexDocsIn(BaseModel):
    store_id: str
    docs: list[DocIn] = Field(..., min_length=1)


class QueryIn(BaseModel):
    store_id: str
    query: str
    k: int = Field(default=5, ge=1, le=100)
    with_scores: bool = False
    search_type: SearchType = "similarity"
    fetch_k: int = Field(default=20, ge=1, le=1000)
    lambda_mult: float = Field(default=0.5, ge=0.0, le=1.0)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class CompareIn(BaseModel):
    dataset_id: str
    embedding_models: list[str] = Field(..., min_length=1)
    query: str
    k: int = Field(default=5, ge=1, le=100)
    with_scores: bool = False
    search_type: SearchType = "similarity"
    fetch_k: int = Field(default=20, ge=1, le=1000)
    lambda_mult: float = Field(default=0.5, ge=0.0, le=1.0)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


@router.get("/models")
def list_models(service: EmbeddingServiceDep) -> dict[str, list[str]]:
    return {"embedding_models": service.list_models()}


@router.get("/stores")
async def list_stores(service: EmbeddingServiceDep) -> dict[str, dict[str, str]]:
    return {"stores": await service.list_stores()}


@router.post("/stores")
async def create_store(payload: CreateStoreIn, service: EmbeddingServiceDep) -> dict[str, bool]:
    await service.create_store(payload.store_id, payload.embedding_key)
    return {"ok": True}


@router.delete("/stores/{store_id}")
async def delete_store(store_id: str, service: EmbeddingServiceDep) -> dict[str, bool]:
    await service.delete_store(store_id)
    return {"ok": True}


@router.post("/index/texts")
async def index_texts(payload: IndexTextsIn, service: EmbeddingServiceDep) -> dict[str, Any]:
    ids = await service.index(payload.store_id, payload.texts, payload.metadatas, payload.ids)
    return {"ok": True, "ids": ids}


@router.post("/index/docs")
async def index_docs(payload: IndexDocsIn, service: EmbeddingServiceDep) -> dict[str, Any]:
    contents = [doc.page_content for doc in payload.docs]
    metadatas = [doc.metadata for doc in payload.docs]
    ids = await service.index(payload.store_id, contents, metadatas)
    return {"ok": True, "ids": ids}


@router.post("/query")
async def query(payload: QueryIn, service: EmbeddingServiceDep) -> dict[str, Any]:
    hits = await service.query(
        payload.store_id,
        payload.query,
        k=payload.k,
        with_scores=payload.with_scores,
        search_type=payload.search_type,
        fetch_k=payload.fetch_k,
        lambda_mult=payload.lambda_mult,
        score_threshold=payload.score_threshold,
    )
    return {"matches": [hit.model_dump(exclude_none=not payload.with_scores) for hit in hits]}


@router.post("/compare")
async def compare(payload: CompareIn, service: EmbeddingServiceDep) -> dict[str, Any]:
    results = await service.compare(
        payload.dataset_id,
        payload.embedding_models,
        payload.query,
        k=payload.k,
        with_scores=payload.with_scores,
        search_type=payload.search_type,
        fetch_k=payload.fetch_k,
        lambda_mult=payload.lambda_mult,
        score_threshold=payload.score_threshold,
    )
    return {
        "query": payload.query,
        "dataset_id": payload.dataset_id,
        "k": payload.k,
        "results": results,
    }
