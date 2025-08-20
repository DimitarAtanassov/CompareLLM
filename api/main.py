import os, asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from registry import ModelRegistry, chat_call, embedding_call, host_lock, find_similar_documents
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import time

# ------------------------ logging (added) ------------------------
import logging

LOG = logging.getLogger("askmanyllms")
if not LOG.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO))

def log_event(kind: str, **fields):
    """
    Single-line JSON logs for easy parsing in Docker/Cloud logs.
    """
    try:
        LOG.info(json.dumps({"event": kind, **fields}, ensure_ascii=False))
    except Exception as _:
        # Fallback to a simple message if something isn't JSON-serializable
        LOG.info(f"{kind} | {fields}")
# ---------------------------------------------------------------

CFG_PATH = os.getenv("MODELS_CONFIG", "/config/models.yaml")
REGISTRY = ModelRegistry.from_path(CFG_PATH)

app = FastAPI(title="Ask Many Models (Pluggable)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for document embeddings
# In production, you'd want to use a vector database like Pinecone, Weaviate, or Chroma
document_store: Dict[str, List[Dict[str, Any]]] = {}

class ChatRequest(BaseModel):
    prompt: str
    models: List[str] | None = None
    # global defaults
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 8192
    min_tokens: Optional[int] = None
    # per-model overrides
    model_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    model_config = ConfigDict(extra="ignore")

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: str
    model_config = ConfigDict(extra="ignore")

class SearchRequest(BaseModel):
    query: str
    embedding_model: str
    dataset_id: str
    top_k: Optional[int] = 5
    model_config = ConfigDict(extra="ignore")

class DatasetUploadRequest(BaseModel):
    dataset_id: str
    documents: List[Dict[str, Any]]  # Each doc should have a 'text' field and any other metadata
    embedding_model: str
    text_field: Optional[str] = "text"  # Field containing the text to embed
    model_config = ConfigDict(extra="ignore")

@app.get("/providers")
def providers():
    return {
        "providers": [
            {
                "name": p.name,
                "type": p.type,
                "base_url": p.base_url,
                "models": p.models,
                "embedding_models": getattr(p, 'embedding_models', []),
                "auth_required": bool(p.api_key_env),
            } for p in REGISTRY.providers.values()
        ]
    }

@app.get("/embedding-models")
def embedding_models():
    """Get all available embedding models."""
    return {
        "embedding_models": list(getattr(REGISTRY, 'embedding_map', {}).keys())
    }

@app.get("/datasets")
def list_datasets():
    """List all uploaded datasets."""
    return {
        "datasets": [
            {
                "dataset_id": dataset_id,
                "document_count": len(docs),
                "sample_fields": list(docs[0].keys()) if docs else []
            }
            for dataset_id, docs in document_store.items()
        ]
    }

@app.post("/embeddings")
async def generate_embeddings(req: EmbeddingRequest):
    """Generate embeddings for given texts."""
    embedding_map = getattr(REGISTRY, 'embedding_map', {})
    if req.model not in embedding_map:
        raise HTTPException(400, f"Unknown embedding model: {req.model}")
    
    provider, model = embedding_map[req.model]
    
    log_event(
        "embedding.start",
        provider=provider.name,
        model=req.model,
        text_count=len(req.texts)
    )
    start = time.perf_counter()
    
    try:
        async with host_lock(provider):
            embeddings = await embedding_call(provider, model, req.texts)
        
        dur = int((time.perf_counter() - start) * 1000)
        log_event(
            "embedding.end",
            provider=provider.name,
            model=req.model,
            ok=True,
            duration_ms=dur,
            text_count=len(req.texts)
        )
        
        return {
            "model": req.model,
            "embeddings": embeddings,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in req.texts),  # Rough estimate
                "total_tokens": sum(len(text.split()) for text in req.texts)
            }
        }
    except Exception as e:
        dur = int((time.perf_counter() - start) * 1000)
        log_event(
            "embedding.end",
            provider=provider.name,
            model=req.model,
            ok=False,
            duration_ms=dur,
            error=str(e)
        )
        raise HTTPException(500, f"Embedding generation failed: {str(e)}")

@app.post("/upload-dataset")
async def upload_dataset(req: DatasetUploadRequest):
    """Upload a dataset and generate embeddings for it."""
    embedding_map = getattr(REGISTRY, 'embedding_map', {})
    if req.embedding_model not in embedding_map:
        raise HTTPException(400, f"Unknown embedding model: {req.embedding_model}")
    
    if not req.documents:
        raise HTTPException(400, "No documents provided")
    
    # Extract texts to embed
    texts = []
    for doc in req.documents:
        if req.text_field not in doc:
            raise HTTPException(400, f"Document missing required field: {req.text_field}")
        texts.append(str(doc[req.text_field]))
    
    provider, model = embedding_map[req.embedding_model]
    
    log_event(
        "dataset.upload.start",
        dataset_id=req.dataset_id,
        provider=provider.name,
        model=req.embedding_model,
        document_count=len(req.documents)
    )
    start = time.perf_counter()
    
    try:
        # Generate embeddings for all texts
        async with host_lock(provider):
            embeddings = await embedding_call(provider, model, texts)
        
        # Create unique dataset ID for this model combination
        dataset_key = f"{req.dataset_id}_{req.embedding_model}"
        
        # Store documents with their embeddings
        enhanced_docs = []
        for i, doc in enumerate(req.documents):
            enhanced_doc = doc.copy()
            enhanced_doc['embedding'] = embeddings[i]
            enhanced_doc['_text_field'] = req.text_field
            enhanced_doc['_embedding_model'] = req.embedding_model
            enhanced_docs.append(enhanced_doc)
        
        document_store[dataset_key] = enhanced_docs
        
        dur = int((time.perf_counter() - start) * 1000)
        log_event(
            "dataset.upload.end",
            dataset_id=req.dataset_id,
            provider=provider.name,
            model=req.embedding_model,
            ok=True,
            duration_ms=dur,
            document_count=len(req.documents)
        )
        
        return {
            "dataset_id": dataset_key,
            "document_count": len(enhanced_docs),
            "embedding_model": req.embedding_model,
            "message": "Dataset uploaded and embeddings generated successfully"
        }
    except Exception as e:
        dur = int((time.perf_counter() - start) * 1000)
        log_event(
            "dataset.upload.end",
            dataset_id=req.dataset_id,
            provider=provider.name,
            model=req.embedding_model,
            ok=False,
            duration_ms=dur,
            error=str(e)
        )
        raise HTTPException(500, f"Dataset upload failed: {str(e)}")

@app.post("/search")
async def semantic_search(req: SearchRequest):
    """Perform semantic search against a dataset."""
    embedding_map = getattr(REGISTRY, 'embedding_map', {})
    if req.embedding_model not in embedding_map:
        raise HTTPException(400, f"Unknown embedding model: {req.embedding_model}")
    
    if req.dataset_id not in document_store:
        raise HTTPException(404, f"Dataset not found: {req.dataset_id}")
    
    provider, model = embedding_map[req.embedding_model]
    
    log_event(
        "search.start",
        dataset_id=req.dataset_id,
        provider=provider.name,
        model=req.embedding_model,
        query=req.query[:100]  # Log first 100 chars
    )
    start = time.perf_counter()
    
    try:
        # Generate embedding for query
        async with host_lock(provider):
            query_embeddings = await embedding_call(provider, model, [req.query])
        
        query_embedding = query_embeddings[0]
        
        # Find similar documents
        documents = document_store[req.dataset_id]
        similar_docs = find_similar_documents(query_embedding, documents, req.top_k)
        
        # Remove embedding from response to reduce payload size
        for doc in similar_docs:
            if 'embedding' in doc:
                del doc['embedding']
        
        dur = int((time.perf_counter() - start) * 1000)
        log_event(
            "search.end",
            dataset_id=req.dataset_id,
            provider=provider.name,
            model=req.embedding_model,
            ok=True,
            duration_ms=dur,
            results_count=len(similar_docs)
        )
        
        return {
            "query": req.query,
            "dataset_id": req.dataset_id,
            "embedding_model": req.embedding_model,
            "results": similar_docs,
            "total_documents": len(documents)
        }
    except Exception as e:
        dur = int((time.perf_counter() - start) * 1000)
        log_event(
            "search.end",
            dataset_id=req.dataset_id,
            provider=provider.name,
            model=req.embedding_model,
            ok=False,
            duration_ms=dur,
            error=str(e)
        )
        raise HTTPException(500, f"Search failed: {str(e)}")

@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    if dataset_id not in document_store:
        raise HTTPException(404, f"Dataset not found: {dataset_id}")
    
    doc_count = len(document_store[dataset_id])
    del document_store[dataset_id]
    
    log_event("dataset.deleted", dataset_id=dataset_id, document_count=doc_count)
    
    return {
        "dataset_id": dataset_id,
        "message": f"Dataset deleted successfully ({doc_count} documents removed)"
    }

@app.post("/ask")
async def ask(req: ChatRequest):
    messages = [
        {"role": "system", "content": "Answer clearly and concisely."},
        {"role": "user", "content": req.prompt},
    ]
    chosen = req.models or list(REGISTRY.model_map.keys())
    unknown = [m for m in chosen if m not in REGISTRY.model_map]
    if unknown:
        raise HTTPException(400, f"Unknown models: {unknown}")

    async def one(model_name: str):
        provider, model = REGISTRY.model_map[model_name]

        # ----- logging: record params passed to this model -----
        log_event(
            "call.start",
            route="/ask",
            provider=provider.name,
            provider_type=provider.type,
            model=model_name,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            min_tokens=req.min_tokens,
        )
        start = time.perf_counter()
        # -------------------------------------------------------

        # serialize per-host (helps on Mac/CPU + Ollama)
        try:
            async with host_lock(provider):
                result = await chat_call(provider, model, messages, req.temperature, req.max_tokens)
            dur = int((time.perf_counter() - start) * 1000)
            log_event(
                "call.end",
                route="/ask",
                provider=provider.name,
                provider_type=provider.type,
                model=model_name,
                ok=True,
                duration_ms=dur,
                answer_chars=len(result or ""),
            )
            return result
        except Exception as e:
            dur = int((time.perf_counter() - start) * 1000)
            log_event(
                "call.end",
                route="/ask",
                provider=provider.name,
                provider_type=provider.type,
                model=model_name,
                ok=False,
                duration_ms=dur,
                error=str(e),
            )
            raise

    tasks = [asyncio.create_task(one(m)) for m in chosen]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    answers: Dict[str, Any] = {}
    for m, res in zip(chosen, results):
        answers[m] = {"error": str(res)} if isinstance(res, Exception) else {"answer": res}
    return {"prompt": req.prompt, "models": chosen, "answers": answers}

# /ask/ndjson
@app.post("/ask/ndjson")
async def ask_ndjson(req: ChatRequest):
    messages = [
        {"role": "system", "content": "Answer clearly and concisely."},
        {"role": "user", "content": req.prompt},
    ]
    chosen = req.models or list(REGISTRY.model_map.keys())
    unknown = [m for m in chosen if m not in REGISTRY.model_map]
    if unknown:
        raise HTTPException(400, f"Unknown models: {unknown}")

    def merged_params(m: str):
        mp = req.model_params.get(m, {})
        temperature = float(mp.get("temperature", req.temperature if req.temperature is not None else 0.7))
        max_tokens = int(mp.get("max_tokens", req.max_tokens if req.max_tokens is not None else 8192))
        min_tokens = mp.get("min_tokens", req.min_tokens)
        min_tokens = int(min_tokens) if (min_tokens is not None) else None
        return temperature, max_tokens, min_tokens

    async def gen(): 
        yield json.dumps({"type": "meta", "models": chosen}) + "\n"
        out_q: asyncio.Queue[str] = asyncio.Queue()

        async def run_one(model_name: str):
            provider, model = REGISTRY.model_map[model_name]
            start = time.perf_counter()
            try:
                t, mx, mn = merged_params(model_name)

                # ----- logging: record per-model params -----
                log_event(
                    "call.start",
                    route="/ask/ndjson",
                    provider=provider.name,
                    provider_type=provider.type,
                    model=model_name,
                    temperature=t,
                    max_tokens=mx,
                    min_tokens=mn,
                )
                # -------------------------------------------

                async with host_lock(provider):
                    final_text = await chat_call(provider, model, messages, t, mx, mn)
                ms = int((time.perf_counter() - start) * 1000)

                log_event(
                    "call.end",
                    route="/ask/ndjson",
                    provider=provider.name,
                    provider_type=provider.type,
                    model=model_name,
                    ok=True,
                    duration_ms=ms,
                    answer_chars=len(final_text or ""),
                )

                out_q.put_nowait(json.dumps({
                    "type": "chunk",
                    "model": model_name,
                    "answer": final_text,
                    "latency_ms": ms
                }) + "\n")
            except Exception as e:
                ms = int((time.perf_counter() - start) * 1000)

                log_event(
                    "call.end",
                    route="/ask/ndjson",
                    provider=provider.name,
                    provider_type=provider.type,
                    model=model_name,
                    ok=False,
                    duration_ms=ms,
                    error=str(e),
                )

                out_q.put_nowait(json.dumps({
                    "type": "chunk",
                    "model": model_name,
                    "error": str(e),
                    "latency_ms": ms
                }) + "\n")

        tasks = [asyncio.create_task(run_one(m)) for m in chosen]
        pending = set(tasks)
        while pending or not out_q.empty():
            try:
                item = await asyncio.wait_for(out_q.get(), timeout=0.1)
                yield item
            except asyncio.TimeoutError:
                pending = {t for t in tasks if not t.done()}
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")