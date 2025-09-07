from __future__ import annotations
from typing import Any, Dict, List, Optional
from time import perf_counter

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from services.embedding_service import EmbeddingService
from langchain_core.documents import Document

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

def _log(msg: str) -> None:
    print(f"[EmbeddingsRouter] {msg}")

# ---- Schemas ----
class CreateStoreIn(BaseModel):
    store_id: str = Field(..., description="Logical id for this vector store (e.g., 'default')")
    embedding_key: str = Field(..., description="Embedding key 'provider:model' (e.g., 'openai:text-embedding-3-large')")

class IndexTextsIn(BaseModel):
    store_id: str
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    ids: Optional[List[str]] = None

class IndexDocsIn(BaseModel):
    store_id: str
    docs: List[Dict[str, Any]]  # each {page_content: str, metadata?: dict}

class QueryIn(BaseModel):
    store_id: str
    query: str
    k: int = Field(default=5, ge=1, le=100)
    with_scores: bool = False

    # retriever controls
    search_type: Optional[str] = Field(default="similarity")
    fetch_k: Optional[int] = Field(default=20, ge=1, le=1000)
    lambda_mult: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class CompareIn(BaseModel):
    dataset_id: str
    embedding_models: List[str]  # e.g. ["openai:text-embedding-3-large", "cohere:embed-english-v3.0"]
    query: str
    k: int = Field(default=5, ge=1, le=100)
    with_scores: bool = False

    # retriever controls (match /query)
    search_type: Optional[str] = Field(default="similarity")
    fetch_k: Optional[int] = Field(default=20, ge=1, le=1000)
    lambda_mult: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding_service

@router.get("/models")
def list_embedding_models(service: EmbeddingService = Depends(get_embedding_service)):
    models = service.list_embedding_models()
    _log(f"GET /models -> {len(models)} models")
    return {"embedding_models": models}

@router.get("/stores")
def list_stores(service: EmbeddingService = Depends(get_embedding_service)):
    stores = service.list_stores()
    _log(f"GET /stores -> {len(stores)} stores")
    return {"stores": stores}

@router.post("/stores")
def create_store(payload: CreateStoreIn, service: EmbeddingService = Depends(get_embedding_service)):
    _log(f"POST /stores -> store_id='{payload.store_id}', embedding_key='{payload.embedding_key}'")
    service.create_store(payload.store_id, payload.embedding_key)
    return {"ok": True}

@router.delete("/stores/{store_id}")
def delete_store(store_id: str, service: EmbeddingService = Depends(get_embedding_service)):
    _log(f"DELETE /stores/{store_id}")
    service.delete_store(store_id)
    return {"ok": True}

@router.post("/index/texts")
async def index_texts(payload: IndexTextsIn, service: EmbeddingService = Depends(get_embedding_service)):
    _log(f"POST /index/texts -> store='{payload.store_id}', n_texts={len(payload.texts)}")
    ids = await service.index_texts(payload.store_id, payload.texts, payload.metadatas, payload.ids)
    _log(f"POST /index/texts -> indexed {len(ids)} ids")
    return {"ok": True, "ids": ids}

@router.post("/index/docs")
async def index_docs(payload: IndexDocsIn, service: EmbeddingService = Depends(get_embedding_service)):
    _log(f"POST /index/docs -> store='{payload.store_id}', n_docs={len(payload.docs)}")
    docs = [Document(page_content=d["page_content"], metadata=d.get("metadata") or {}) for d in payload.docs]
    ids = await service.index_docs(payload.store_id, docs)
    _log(f"POST /index/docs -> indexed {len(ids)} ids")
    return {"ok": True, "ids": ids}

@router.post("/query")
async def query(payload: QueryIn, service: EmbeddingService = Depends(get_embedding_service)):
    """
    - similarity: fast cosine top-K
    - mmr / similarity_score_threshold: retriever with optional post-scoring
    """
    t0 = perf_counter()
    stype = (payload.search_type or "similarity").strip().lower()
    _log(f"POST /query -> store='{payload.store_id}', type='{stype}', k={payload.k}, with_scores={payload.with_scores}")

    if stype == "similarity":
        results = await service.similarity_search(
            payload.store_id,
            payload.query,
            k=payload.k,
            with_scores=payload.with_scores
        )
        out = []
        if payload.with_scores:
            for doc, score in results:
                out.append({"page_content": doc.page_content, "metadata": doc.metadata, "score": float(score)})
        else:
            for doc in results:
                out.append({"page_content": doc.page_content, "metadata": doc.metadata})

        _log(f"/query similarity -> {len(out)} matches (took {(perf_counter()-t0)*1000:.1f} ms)")
        return {"matches": out}

    # Retriever paths
    emb_reg = request.app.state.embedding_registry
    try:
        vs = emb_reg.get_store(payload.store_id)
    except KeyError as e:
        _log(f"/query retriever ERROR -> {e}")
        raise HTTPException(status_code=404, detail=str(e))

    search_kwargs: Dict[str, Any] = {"k": payload.k}
    if stype == "mmr":
        if payload.fetch_k is not None:
            search_kwargs["fetch_k"] = payload.fetch_k
        if payload.lambda_mult is not None:
            search_kwargs["lambda_mult"] = payload.lambda_mult
    elif stype == "similarity_score_threshold":
        if payload.score_threshold is not None:
            search_kwargs["score_threshold"] = payload.score_threshold
    else:
        _log(f"/query ERROR unsupported search_type='{payload.search_type}'")
        raise HTTPException(status_code=400, detail=f"Unsupported search_type '{payload.search_type}'. Use 'similarity', 'mmr', or 'similarity_score_threshold'.")

    retriever = vs.as_retriever(search_type=stype, search_kwargs=search_kwargs)
    docs = await retriever.ainvoke(payload.query)
    _log(f"/query retriever -> {len(docs)} doc(s) returned by retriever")

    if payload.with_scores:
        qvec = vs.embeddings.embed_query(payload.query)
        scored = await vs.asimilarity_search_with_score_by_vector(qvec, k=max(payload.k, len(docs)))
        score_by_id: Dict[Optional[str], float] = {getattr(d, "id", None): float(s) for d, s in scored}

        out = [{
            "page_content": d.page_content,
            "metadata": d.metadata,
            "score": score_by_id.get(getattr(d, "id", None))
        } for d in docs]

        _log(f"/query retriever+score -> {len(out)} matches (took {(perf_counter()-t0)*1000:.1f} ms)")
        return {"matches": out}

    out = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    _log(f"/query retriever -> {len(out)} matches (took {(perf_counter()-t0)*1000:.1f} ms)")
    return {"matches": out}

@router.post("/compare")
async def compare_across_models(
    payload: CompareIn,
    service: EmbeddingService = Depends(get_embedding_service),
    request: Request = None,
):
    search_params = {
        "k": payload.k,
        "with_scores": payload.with_scores,
        "search_type": payload.search_type,
        "fetch_k": payload.fetch_k,
        "lambda_mult": payload.lambda_mult,
        "score_threshold": payload.score_threshold,
    }
    memory_backend = request.app.state.graph_memory if request else None
    return await service.compare_across_models(
        dataset_id=payload.dataset_id,
        embedding_models=payload.embedding_models,
        query=payload.query,
        search_params=search_params,
        memory_backend=memory_backend,
    )
