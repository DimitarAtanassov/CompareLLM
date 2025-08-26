
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from models.enhanced_requests import EmbeddingRequest, DatasetUploadRequest, SearchRequest
from models.responses import EmbeddingResponse, DatasetUploadResponse, DatasetListResponse, SearchResponse

from services.embedding_service import EmbeddingService
from services.dataset_service import DatasetService
from services.search_services import SearchService  # matches your main.py import

from fastapi import Request, HTTPException

def _get_service(request: Request, key: str, fallback_attr: str = None):
    services = getattr(request.app.state, "services", None)
    if isinstance(services, dict) and key in services:
        return services[key]
    if fallback_attr and hasattr(request.app.state, fallback_attr):
        return getattr(request.app.state, fallback_attr)
    raise HTTPException(status_code=500, detail=f"Service '{key}' not initialized")

def get_embedding_service(request: Request) -> EmbeddingService:
    return _get_service(request, "embedding", "embedding_service")

def get_dataset_service(request: Request) -> DatasetService:
    return _get_service(request, "dataset")

def get_search_service(request: Request) -> SearchService:
    return _get_service(request, "search", "search_service")

router = APIRouter()

# ----------------- Embeddings -----------------
@router.post("/embeddings", response_model=EmbeddingResponse, summary="Generate embeddings")
async def create_embeddings(req: EmbeddingRequest, request: Request) -> EmbeddingResponse:
    svc: EmbeddingService = get_embedding_service(request)
    return await svc.generate_embeddings(req)

# ----------------- Datasets -----------------
@router.post("/datasets/upload", response_model=DatasetUploadResponse, summary="Upload a dataset and pre-embed")
async def upload_dataset(req: DatasetUploadRequest, request: Request) -> DatasetUploadResponse:
    svc: DatasetService = get_dataset_service(request)
    return await svc.upload_dataset(req)

@router.get("/datasets", response_model=DatasetListResponse, summary="List embedded datasets")
async def list_datasets(request: Request) -> DatasetListResponse:
    svc: DatasetService = get_dataset_service(request)
    return await svc.list_datasets()

@router.delete("/datasets/{dataset_id}", summary="Delete embedded dataset")
async def delete_dataset(dataset_id: str, request: Request) -> Dict[str, Any]:
    svc: DatasetService = get_dataset_service(request)
    return await svc.delete_dataset(dataset_id)

# ----------------- Semantic Search -----------------
@router.post("/search/semantic", response_model=SearchResponse, summary="Semantic search against one dataset")
async def semantic_search(req: SearchRequest, request: Request) -> SearchResponse:
    svc: SearchService = get_search_service(request)
    return await svc.semantic_search(req)

# ---------- Multi-search models ----------
class MultiBucket(BaseModel):
    error: Optional[str] = None
    items: List[Dict[str, Any]] = Field(default_factory=list)
    dataset_id: Optional[str] = None
    total_documents: Optional[int] = None

class SelfDatasetCompareRequest(BaseModel):
    query: str
    embedding_models: List[str]
    top_k: int = 5
    dataset_base: Optional[str] = None  # compose per-model as f"{base}_{model}"

class MultiSearchRequest(BaseModel):
    base_dataset_id: str
    embedding_models: List[str]
    query: str
    top_k: int = 5

class MultiSearchResponse(BaseModel):
    query: str
    results: Dict[str, MultiBucket]
    duration_ms: Optional[int] = None

# ----------------- Compare results across models for a single base -----------------
@router.post("/search/multi", summary="Compare search results across multiple embedding models")
async def search_multi(req: MultiSearchRequest, request: Request):
    svc: SearchService = get_search_service(request)
    return await svc.semantic_search_multi(
        base_dataset_id=req.base_dataset_id,
        embedding_models=req.embedding_models,
        query=req.query,
        top_k=req.top_k or 5,
    )

# ----------------- Self-dataset compare (auto-detect base) -----------------
@router.post("/search/self-dataset-compare", response_model=MultiSearchResponse, summary="Compare results side-by-side within a single dataset base")
async def self_dataset_compare(
    payload: SelfDatasetCompareRequest,
    request: Request,
):
    # Resolve services
    embed_svc: EmbeddingService = get_embedding_service(request)
    search_svc: SearchService = get_search_service(request)

    # Collect all dataset IDs (try memory_store then search service, but don't hard-fail)
    all_dataset_ids: List[str] = []
    try:
        memory_store = getattr(request.app.state, "memory_store", None)
        if memory_store:
            lister = getattr(memory_store, "list_datasets", None)
            if callable(lister):
                maybe = lister()
                got = await maybe if hasattr(maybe, "__await__") else maybe
                if isinstance(got, dict):
                    all_dataset_ids = list(got.get("datasets") or [])
                elif isinstance(got, (list, tuple)):
                    all_dataset_ids = list(got)
            elif hasattr(memory_store, "datasets"):
                all_dataset_ids = list(getattr(memory_store, "datasets") or [])
    except Exception:
        all_dataset_ids = []

    if not all_dataset_ids:
        try:
            lister2 = getattr(search_svc, "list_datasets", None)
            if callable(lister2):
                maybe2 = lister2()
                got2 = await maybe2 if hasattr(maybe2, "__await__") else maybe2
                if isinstance(got2, dict):
                    all_dataset_ids = list(got2.get("datasets") or [])
                elif isinstance(got2, (list, tuple)):
                    all_dataset_ids = list(got2)
        except Exception:
            pass

    # Helper: resolve dataset for a specific model
    def resolve_dataset_for(model: str) -> Optional[str]:
        if payload.dataset_base:
            return f"{payload.dataset_base}_{model}"
        # find dataset that endswith _{model}
        for ds in all_dataset_ids:
            if ds.endswith(f"_{model}"):
                return ds
        return None

    # Try to auto-detect a base that covers all requested models
    chosen_base: Optional[str] = None
    if not payload.dataset_base and all_dataset_ids:
        pairs = []
        for d in all_dataset_ids:
            for m in payload.embedding_models:
                sfx = f"_{m}"
                if d.endswith(sfx):
                    base = d[: -len(sfx)]
                    pairs.append((base, m))
        from collections import defaultdict
        cover = defaultdict(set)
        for base, m in pairs:
            cover[base].add(m)
        common = [b for b, ms in cover.items() if set(payload.embedding_models).issubset(ms)]
        if common:
            chosen_base = sorted(common)[0]

    # Perform search per model
    out: Dict[str, MultiBucket] = {{}}
    for model in payload.embedding_models:
        bucket = MultiBucket(items=[])
        try:
            if chosen_base:
                dataset_id = f"{chosen_base}_{model}"
                bucket.dataset_id = dataset_id
            else:
                resolved = resolve_dataset_for(model)
                if not resolved:
                    bucket.error = f"No dataset found for model '{model}'"
                    out[model] = bucket
                    continue
                bucket.dataset_id = resolved

            # Embed query using this model
            e_req = EmbeddingRequest(texts=[payload.query], model=model)
            e_resp = await embed_svc.generate_embeddings(e_req)
            qvec = e_resp.embeddings[0]

            # Use search service's generic search
            items = await search_svc.search(bucket.dataset_id, qvec, payload.top_k or 5)
            bucket.items = items
            # Optionally retrieve total count via storage
            try:
                storage = getattr(request.app.state, "memory_store", None)
                if storage:
                    docs = await storage.get_dataset(bucket.dataset_id)
                    bucket.total_documents = len(docs or [])
            except Exception:
                pass

        except Exception as e:
            bucket.error = str(e)
        out[model] = bucket

    return MultiSearchResponse(query=payload.query, results=out)

# ----------------- Provider feature matrix (mounted at /providers/features and /v2/providers/features) -----------------
@router.get("/providers/features")
async def get_provider_features(request: Request):
    reg = getattr(request.app.state, "registry", None)
    if not reg:
        raise HTTPException(status_code=500, detail="Model registry not initialized")
    features: Dict[str, Any] = {{}}
    for pname, provider in reg.providers.items():
        pf = {{
            "type": provider.type,
            "models": provider.models,
            "embedding_models": provider.embedding_models,
            "features": {{}}
        }}
        if provider.type == "anthropic":
            pf["features"] = {{
                "thinking": True, "tools": True, "streaming": True, "system_messages": True,
                "stop_sequences": True, "service_tiers": True, "large_context": True
            }}
        elif provider.type == "openai":
            pf["features"] = {{
                "tools": True, "streaming": True, "json_mode": True, "seed": True,
                "logit_bias": True, "system_messages": True
            }}
        elif provider.type == "gemini":
            pf["features"] = {{
                "multimodal": True, "safety_settings": True, "tools": True, "streaming": True,
                "system_messages": True, "large_context": True
            }}
        elif provider.type == "ollama":
            pf["features"] = {{
                "local": True, "custom_models": True, "mirostat": True, "format_json": True
            }}
        features[pname] = pf
    return {{"providers": features}}
