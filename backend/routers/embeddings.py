from __future__ import annotations
from typing import Any, Dict, List, Optional
from time import perf_counter

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
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

@router.get("/models")
def list_embedding_models(request: Request):
    svc = request.app.state.embedding_service
    models = svc.list_embedding_models()
    _log(f"GET /models -> {len(models)} models")
    return {"embedding_models": models}

@router.get("/stores")
def list_stores(request: Request):
    svc = request.app.state.embedding_service
    stores = svc.list_stores()
    _log(f"GET /stores -> {len(stores)} stores")
    return {"stores": stores}

@router.post("/stores")
def create_store(payload: CreateStoreIn, request: Request):
    _log(f"POST /stores -> store_id='{payload.store_id}', embedding_key='{payload.embedding_key}'")
    svc = request.app.state.embedding_service
    try:
        svc.create_store(payload.store_id, payload.embedding_key)
        return {"ok": True}
    except (KeyError, ValueError) as e:
        _log(f"POST /stores ERROR -> {type(e).__name__}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/stores/{store_id}")
def delete_store(store_id: str, request: Request):
    _log(f"DELETE /stores/{store_id}")
    svc = request.app.state.embedding_service
    svc.delete_store(store_id)
    return {"ok": True}

@router.post("/index/texts")
async def index_texts(payload: IndexTextsIn, request: Request):
    _log(f"POST /index/texts -> store='{payload.store_id}', n_texts={len(payload.texts)}")
    svc = request.app.state.embedding_service
    try:
        ids = await svc.aadd_texts(payload.store_id, payload.texts, payload.metadatas, payload.ids)
        _log(f"POST /index/texts -> indexed {len(ids)} ids")
        return {"ok": True, "ids": ids}
    except KeyError as e:
        _log(f"POST /index/texts ERROR -> {e}")
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/index/docs")
async def index_docs(payload: IndexDocsIn, request: Request):
    _log(f"POST /index/docs -> store='{payload.store_id}', n_docs={len(payload.docs)}")
    svc = request.app.state.embedding_service
    docs = [Document(page_content=d["page_content"], metadata=d.get("metadata") or {}) for d in payload.docs]
    try:
        ids = await svc.aadd_documents(payload.store_id, docs)
        _log(f"POST /index/docs -> indexed {len(ids)} ids")
        return {"ok": True, "ids": ids}
    except KeyError as e:
        _log(f"POST /index/docs ERROR -> {e}")
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/query")
async def query(payload: QueryIn, request: Request):
    """
    - similarity: fast cosine top-K
    - mmr / similarity_score_threshold: retriever with optional post-scoring
    """
    t0 = perf_counter()
    stype = (payload.search_type or "similarity").strip().lower()
    _log(f"POST /query -> store='{payload.store_id}', type='{stype}', k={payload.k}, with_scores={payload.with_scores}")

    if stype == "similarity":
        svc = request.app.state.embedding_service
        try:
            results = await svc.asimilarity_search(
                payload.store_id,
                payload.query,
                k=payload.k,
                with_scores=payload.with_scores
            )
        except KeyError as e:
            _log(f"/query similarity ERROR -> {e}")
            raise HTTPException(status_code=404, detail=str(e))

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
async def compare_across_models(payload: CompareIn, request: Request):
    """
    Compare multiple embedding models **on a single dataset** using a LangGraph.
    """
    from graphs.factory import build_embedding_comparison_graph
    import uuid

    emb_reg = request.app.state.embedding_registry
    
    # 1. Build the graph
    graph, checkpointer = build_embedding_comparison_graph(
        registry=emb_reg,
        embedding_keys=payload.embedding_models,
        dataset_id=payload.dataset_id,
        memory_backend=request.app.state.graph_memory,
    )

    # 2. Prepare search parameters from payload
    search_params = {
        "k": payload.k,
        "with_scores": payload.with_scores,
        "search_type": payload.search_type,
        "fetch_k": payload.fetch_k,
        "lambda_mult": payload.lambda_mult,
        "score_threshold": payload.score_threshold,
    }

    # 3. Invoke the graph
    thread_id = str(uuid.uuid4())
    final_state = await graph.ainvoke(
        {
            "query": payload.query,
            "targets": payload.embedding_models,
            "search_params": search_params,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    # 4. Format results
    results = final_state.get("results", {})
    errors = final_state.get("errors", {})
    
    # Merge errors into the results dict for the response
    for emb_key, err_msg in errors.items():
        if emb_key not in results:
            results[emb_key] = {"items": [], "error": err_msg}
        else:
            results[emb_key]["error"] = err_msg

    return {
        "query": payload.query,
        "dataset_id": payload.dataset_id,
        "k": payload.k,
        "results": results,
    }
