import time
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import asyncio
import json

from models.enhanced_requests import EnhancedChatRequest, EnhancedOpenAIChatRequest, AnthropicProviderParams
from models.responses import ChatResponse, ModelAnswer
from services.enhanced_chat_service import EnhancedChatService
from providers.registry import ModelRegistry
from providers.adapters.enhanced_chat_adapter import EnhancedChatAdapter
from core.exceptions import AskManyLLMsException

router = APIRouter(prefix="/v2", tags=["Enhanced API"])

class MultiBucket(BaseModel):
    error: Optional[str] = None
    items: List[Dict[str, Any]] = Field(default_factory=list)
    dataset_id: Optional[str] = None
    total_documents: Optional[int] = None

class SelfDatasetCompareRequest(BaseModel):
    query: str
    embedding_models: List[str]
    top_k: int = 5
    # Optional base to compose per-model dataset_id == f"{dataset_base}_{model}"
    dataset_base: Optional[str] = None

class MultiSearchResponse(BaseModel):
    query: str
    results: Dict[str, MultiBucket]
    duration_ms: Optional[int] = None

# ---------- Response models (OpenAI-compatible) ----------

class OpenAIChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

# ---------- Model capabilities / validation ----------

class ModelCapabilities(BaseModel):
    model_name: str
    provider_name: str
    provider_type: str
    supports_thinking: Optional[bool] = None
    supports_tools: Optional[bool] = None
    supports_streaming: Optional[bool] = None
    max_context_tokens: Optional[int] = None
    default_rpm: Optional[int] = None

class ValidationResult(BaseModel):
    valid: bool
    warnings: List[str]
    errors: List[str]

class ParameterExampleResponse(BaseModel):
    anthropic_example: Dict[str, Any]
    openai_example: Dict[str, Any]
    gemini_example: Dict[str, Any]
    ollama_example: Dict[str, Any]

# ---------- NEW: Multi-provider embeddings search request ----------

class MultiSearchRequest(BaseModel):
    base_dataset_id: str
    embedding_models: List[str]
    query: str
    top_k: int = 5


# --- Helpers to access services on app.state ---------------------------------
# Adjust these if you wire services differently.

def get_embedding_service(request: Request):
    svc = getattr(request.app.state, "embedding_service", None)
    if svc is None:
        raise HTTPException(status_code=500, detail="Embedding service not initialized")
    return svc

def get_search_service(request: Request):
    svc = getattr(request.app.state, "search_service", None)
    if svc is None:
        raise HTTPException(status_code=500, detail="Search service not initialized")
    return svc

def get_memory_store(request: Request):
    store = getattr(request.app.state, "memory_store", None)
    if store is None:
        # Optional; only needed if you want to list datasets for auto-detect
        raise HTTPException(status_code=500, detail="Memory store not initialized")
    return store


# --- Dataset resolution logic -------------------------------------------------
def _find_dataset_for_model(
    model: str,
    dataset_base: Optional[str],
    all_dataset_ids: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Returns {'dataset_id': str} or {'error': str}.
    If dataset_base is provided -> use f"{dataset_base}_{model}".
    Else, auto-detect by suffix match on all_dataset_ids.
    """
    if dataset_base:
        return {"dataset_id": f"{dataset_base}_{model}"}

    if not all_dataset_ids:
        return {"error": "No datasets available and no dataset_base provided"}

    suffix = f"_{model}"
    matches = [d for d in all_dataset_ids if d.endswith(suffix)]
    if len(matches) == 1:
        return {"dataset_id": matches[0]}
    if len(matches) == 0:
        return {"error": f"No dataset found ending with '{suffix}'"}
    return {"error": f"Multiple datasets match '{suffix}': {matches}"}

def setup_enhanced_chat_routes(chat_service: EnhancedChatService, registry: ModelRegistry):
    """
    Setup enhanced chat and search routes with dependency injection.
    Expects the following singletons to be set at app startup:
      - request.app.state.search_service : SearchService
      - request.app.state.registry       : ModelRegistry
      - request.app.state.embedding_service / storage handled elsewhere
    """

    # --- Dependency to fetch SearchService from app.state ---
    def get_search_service(request: Request):
        svc = getattr(request.app.state, "search_service", None)
        if svc is None:
            raise HTTPException(status_code=500, detail="SearchService not initialized")
        return svc

    # ---------- Enhanced OpenAI-compatible chat completions ----------
    @router.post("/chat/completions", response_model=OpenAIChatResponse)
    async def enhanced_openai_chat_completions(request: EnhancedOpenAIChatRequest):
        """Enhanced OpenAI-compatible chat completions with provider parameters."""
        try:
            # Validate model exists
            if request.model not in registry.model_map:
                available_models = list(registry.model_map.keys())
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {request.model} not found. Available: {available_models}"
                )

            # Convert to internal enhanced format
            internal_request = request.to_enhanced_request()

            # Validate request for the specific model
            validation = await chat_service.validate_request_for_model(
                request.model, internal_request
            )

            if not validation["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Request validation failed: {', '.join(validation['errors'])}"
                )

            # Log warnings if any
            if validation["warnings"]:
                print(f"⚠️  Request warnings: {', '.join(validation['warnings'])}")

            # Process the request
            response = await chat_service.chat_completion(internal_request)

            # Convert to OpenAI format
            model_answer = response.answers[request.model]

            if model_answer.error:
                raise HTTPException(status_code=500, detail=model_answer.error)

            choice = OpenAIChoice(
                index=0,
                message={"role": "assistant", "content": model_answer.answer or ""},
                finish_reason="stop"
            )

            # Very rough token usage estimation (word count proxy)
            prompt_tokens = sum(len(msg.get("content", "").split()) for msg in request.messages)
            completion_tokens = len((model_answer.answer or "").split())

            return OpenAIChatResponse(
                id=f"enhanced-{int(asyncio.get_event_loop().time())}",
                created=int(asyncio.get_event_loop().time()),
                model=request.model,
                choices=[choice],
                usage=OpenAIUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )

        except AskManyLLMsException as e:
            raise HTTPException(status_code=400, detail=e.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---------- NEW: Multi-provider embeddings search endpoint ----------
    @router.post("/search/multi")
    async def search_multi(
        req: MultiSearchRequest,
        search_service = Depends(get_search_service),
    ) -> Dict[str, Any]:
        """
        Compare cosine-similarity search results across multiple embedding models
        using the SAME query. Expects datasets saved as {base_dataset_id}_{model}.
        """
        return await search_service.semantic_search_multi(
            base_dataset_id=req.base_dataset_id,
            embedding_models=req.embedding_models,
            query=req.query,
            top_k=req.top_k,
        )

    def _infer_dataset_base_for_models(
        all_dataset_ids: List[str],
        embedding_models: List[str],
    ) -> Optional[str]:
        """
        Given a flat list of dataset ids like:
        ["stock_news_voyage-2", "stock_news_nomic-embed-text", "faq_voyage-2"]
        and requested embedding_models:
        ["voyage-2", "nomic-embed-text"]
        return the dataset_base that covers ALL requested models, if exactly one exists.
        If multiple bases fully cover, pick the one with the largest total number of shards.
        If none fully cover, return None (caller can error gracefully).
        """
        # Build: base -> {models it has}
        base_to_models = {}
        for ds in all_dataset_ids:
            for m in embedding_models:
                suffix = f"_{m}"
                if ds.endswith(suffix):
                    base = ds[: -len(suffix)]
                    base_to_models.setdefault(base, set()).add(m)

        # Candidates that fully cover requested models
        required = set(embedding_models)
        full_coverage = [b for b, ms in base_to_models.items() if required.issubset(ms)]
        if not full_coverage:
            return None
        if len(full_coverage) == 1:
            return full_coverage[0]

        # Tie-breaker: choose the base with the most shards overall (most models embedded)
        full_coverage.sort(key=lambda b: len(base_to_models[b]), reverse=True)
        return full_coverage[0]


    # ---------- Enhanced multi-model chat (native enhanced format) ----------
    @router.post("/chat/completions/enhanced", response_model=ChatResponse)
    async def enhanced_multi_model_chat(request: EnhancedChatRequest):
        """Enhanced multi-model chat with full provider parameter support."""
        try:
            # Validate all requested models
            chosen_models = request.models or list(registry.model_map.keys())
            unknown_models = [m for m in chosen_models if m not in registry.model_map]
            if unknown_models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown models: {unknown_models}. Available: {list(registry.model_map.keys())}"
                )

            # Validate request for each model
            validation_results = {}
            for model in chosen_models:
                validation = await chat_service.validate_request_for_model(model, request)
                validation_results[model] = validation

                if not validation["valid"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model {model} validation failed: {', '.join(validation['errors'])}"
                    )

            # Log any warnings
            for model, validation in validation_results.items():
                if validation["warnings"]:
                    print(f"⚠️  Model {model} warnings: {', '.join(validation['warnings'])}")

            # Process the request
            response = await chat_service.chat_completion(request)
            return response

        except AskManyLLMsException as e:
            raise HTTPException(status_code=400, detail=e.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---------- Model capability and validation helpers ----------
    @router.get("/models/{model_name}/capabilities", response_model=ModelCapabilities)
    async def get_model_capabilities(model_name: str):
        """Get capabilities and configuration for a specific model."""
        try:
            capabilities = chat_service.get_model_capabilities(model_name)
            return ModelCapabilities(**capabilities)
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/models/{model_name}/validate", response_model=ValidationResult)
    async def validate_request_for_model(model_name: str, request: EnhancedChatRequest):
        """Validate that a request is compatible with a specific model."""
        try:
            validation = await chat_service.validate_request_for_model(model_name, request)
            return ValidationResult(**validation)
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # ---------- Parameter example helper ----------
    @router.get("/parameters/examples", response_model=ParameterExampleResponse)
    async def get_parameter_examples():
        """Get examples of provider-specific parameters."""
        return ParameterExampleResponse(
            anthropic_example={
                "thinking_enabled": True,
                "thinking_budget_tokens": 2048,
                "top_k": 40,
                "top_p": 0.9,
                "stop_sequences": ["Human:", "Assistant:"],
                "service_tier": "auto",
                "tool_choice_type": "auto",
                "user_id": "user-123"
            },
            openai_example={
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "stop": ["Human:", "AI:"],
                "seed": 42,
                "response_format": {"type": "json_object"},
                "user": "user-123"
            },
            gemini_example={
                "top_k": 40,
                "top_p": 0.9,
                "candidate_count": 1,
                "stop_sequences": ["Human:", "AI:"],
                "safety_settings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            },
            ollama_example={
                "mirostat": 1,
                "mirostat_eta": 0.1,
                "mirostat_tau": 5.0,
                "num_ctx": 4096,
                "repeat_last_n": 64,
                "repeat_penalty": 1.1,
                "seed": 42,
                "top_k": 40,
                "top_p": 0.9,
                "format": "json"
            }
        )

    # ---------- Anthropic-optimized endpoint ----------
    @router.post("/chat/anthropic", response_model=ChatResponse)
    async def anthropic_optimized_chat(
        messages: List[Dict[str, str]],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
        temperature: float = 1.0,
        thinking_enabled: bool = False,
        thinking_budget: Optional[int] = None,
        service_tier: str = "auto",
        stop_sequences: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ):
        """Optimized endpoint for Anthropic Claude models with thinking support."""
        try:
            # Validate model is Anthropic
            if model not in registry.model_map:
                raise HTTPException(status_code=404, detail=f"Model {model} not found")

            provider, _ = registry.model_map[model]
            if provider.type != "anthropic":
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {model} is not an Anthropic model"
                )

            # Create enhanced request with Anthropic parameters
            from models.DEPRECATED_requests import create_anthropic_request

            request = create_anthropic_request(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_enabled=thinking_enabled,
                thinking_budget=thinking_budget,
                service_tier=service_tier,
                stop_sequences=stop_sequences,
                top_k=top_k,
                top_p=top_p
            )

            # Process the request
            response = await chat_service.chat_completion(request)
            return response

        except AskManyLLMsException as e:
            raise HTTPException(status_code=400, detail=e.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---------- Provider feature matrix ----------
    @router.get("/providers/features")
    async def get_provider_features():
        """Get feature matrix for all providers."""
        features = {}

        for provider_name, provider in registry.providers.items():
            provider_features = {
                "type": provider.type,
                "models": provider.models,
                "embedding_models": provider.embedding_models,
                "features": {}
            }

            if provider.type == "anthropic":
                provider_features["features"] = {
                    "thinking": True,
                    "tools": True,
                    "streaming": True,
                    "system_messages": True,
                    "stop_sequences": True,
                    "service_tiers": True,
                    "large_context": True
                }
            elif provider.type == "openai":
                provider_features["features"] = {
                    "function_calling": True,
                    "tools": True,
                    "streaming": True,
                    "json_mode": True,
                    "seed": True,
                    "logit_bias": True,
                    "system_messages": True
                }
            elif provider.type == "gemini":
                provider_features["features"] = {
                    "multimodal": True,
                    "safety_settings": True,
                    "tools": True,
                    "streaming": True,
                    "system_messages": True,
                    "large_context": True
                }
            elif provider.type == "ollama":
                provider_features["features"] = {
                    "local_deployment": True,
                    "custom_models": True,
                    "mirostat": True,
                    "streaming": True,
                    "system_messages": True,
                    "json_mode": True
                }

            features[provider_name] = provider_features

        return {"providers": features}

    return router


@router.post("/search/self-dataset-compare", response_model=MultiSearchResponse)
async def self_dataset_compare(
    payload: SelfDatasetCompareRequest,
    request: Request,
    embedding_service = Depends(get_embedding_service),
    search_service = Depends(get_search_service),
):
    """
    For each embedding model:
      - embed the user query with that model
      - search that model's dataset (auto-detected)
      - return comparable top_k results per model

    Auto-detect rules:
      1) If payload.dataset_base is provided -> use f"{dataset_base}_{model}".
      2) Else, list datasets and prefer a base that is present for *all* requested models.
      3) Else, fall back to the first dataset that ends with _{model} per model.
    """
    start = time.time()
    results: Dict[str, MultiBucket] = {}

    # --- Try to obtain all dataset ids without hard-failing ---
    all_dataset_ids: List[str] = []

    # (A) Try memory_store if present
    memory_store = getattr(request.app.state, "memory_store", None)
    if memory_store:
        try:
            lister = getattr(memory_store, "list_datasets", None)
            if callable(lister):
                maybe = lister()
                got = await maybe if asyncio.iscoroutine(maybe) else maybe
                # normalize shapes
                if isinstance(got, dict) and "datasets" in got:
                    all_dataset_ids = list(got["datasets"] or [])
                elif isinstance(got, (list, tuple)):
                    all_dataset_ids = list(got)
            elif hasattr(memory_store, "datasets"):
                all_dataset_ids = list(getattr(memory_store, "datasets") or [])
        except Exception:
            all_dataset_ids = []

    # (B) Fallback to search_service if nothing found
    if not all_dataset_ids:
        try:
            lister2 = getattr(search_service, "list_datasets", None)
            if callable(lister2):
                maybe2 = lister2()
                got2 = await maybe2 if asyncio.iscoroutine(maybe2) else maybe2
                if isinstance(got2, dict) and "datasets" in got2:
                    all_dataset_ids = list(got2["datasets"] or [])
                elif isinstance(got2, (list, tuple)):
                    all_dataset_ids = list(got2)
        except Exception:
            pass

    # --- Helper: resolve a dataset_id for a given model ---
    def resolve_dataset_for(model: str) -> Dict[str, str]:
        # If caller gave a base, use it directly
        if payload.dataset_base:
            return {"dataset_id": f"{payload.dataset_base}_{model}"}

        # No base provided; try to find a common base across all models
        # Build mapping: model -> [datasets ending with _{model}]
        suffix = f"_{model}"
        candidates = [d for d in all_dataset_ids if d.endswith(suffix)]
        if not candidates:
            return {"error": f"No dataset found ending with '{suffix}'"}

        return {"dataset_id": candidates[0]}  # simple pick; upgraded below

    # If no dataset_base, try to pick a common base across all models first
    chosen_base: Optional[str] = None
    if not payload.dataset_base and all_dataset_ids:
        # build reverse index: dataset_id -> base, model
        pairs = []
        for d in all_dataset_ids:
            # look for trailing _{model}
            for m in payload.embedding_models:
                sfx = f"_{m}"
                if d.endswith(sfx):
                    base = d[: -len(sfx)]
                    pairs.append((base, m))
        # find bases that cover all selected models
        from collections import defaultdict
        cover = defaultdict(set)
        for base, m in pairs:
            cover[base].add(m)
        common = [b for b, ms in cover.items() if set(payload.embedding_models).issubset(ms)]
        if common:
            # choose the first common base deterministically (alphabetical)
            chosen_base = sorted(common)[0]

    # --- Embed + search per model (do not fail whole request) ---
    for model in payload.embedding_models:
        bucket = MultiBucket(items=[])
        try:
            # dataset resolution
            if chosen_base:
                dataset_id = f"{chosen_base}_{model}"
                bucket.dataset_id = dataset_id
            else:
                resolved = resolve_dataset_for(model)
                if "error" in resolved:
                    bucket.error = resolved["error"]
                    results[model] = bucket
                    continue
                dataset_id = resolved["dataset_id"]
                bucket.dataset_id = dataset_id

            # optional: total doc count if available
            try:
                if memory_store and hasattr(memory_store, "get_dataset_meta") and callable(memory_store.get_dataset_meta):
                    meta = memory_store.get_dataset_meta(dataset_id)
                    if isinstance(meta, dict) and "document_count" in meta:
                        bucket.total_documents = int(meta["document_count"])
            except Exception:
                pass

            # embed the query with this model
            embed = getattr(embedding_service, "embed_texts", None)
            if not callable(embed):
                raise RuntimeError("embedding_service.embed_texts() missing")
            maybe_vecs = embed(model, [payload.query])
            vecs = await maybe_vecs if hasattr(maybe_vecs, "__await__") else maybe_vecs
            if not vecs or not vecs[0]:
                raise RuntimeError("Empty embedding vector returned")
            query_vector = vecs[0]

            # vector search
            search_fn = getattr(search_service, "search", None)
            if not callable(search_fn):
                raise RuntimeError("search_service.search() missing")
            maybe_hits = search_fn(dataset_id=dataset_id, query_vector=query_vector, top_k=payload.top_k)
            raw_hits = await maybe_hits if hasattr(maybe_hits, "__await__") else maybe_hits

            items: List[Dict[str, Any]] = []
            for hit in (raw_hits or []):
                score = hit.get("similarity_score") or hit.get("score") or hit.get("distance")
                if (score is not None) and ("distance" in hit):
                    try:
                        score = max(0.0, 1.0 - float(score))  # convert distance-ish to similarity-ish
                    except Exception:
                        pass

                row = {**hit.get("doc", {})} if isinstance(hit.get("doc"), dict) else {
                    k: v for k, v in hit.items() if k != "doc"
                }
                try:
                    row["similarity_score"] = float(score)
                except Exception:
                    row["similarity_score"] = 0.0
                items.append(row)

            bucket.items = items

        except Exception as e:
            bucket.error = f"{type(e).__name__}: {e}"

        results[model] = bucket

    return MultiSearchResponse(
        query=payload.query,
        results=results,
        duration_ms=int((time.time() - start) * 1000),
    )



# ---------- Example usage payload helpers (for docs / tests) ----------

def create_anthropic_thinking_example():
    """Example of using Anthropic's extended thinking feature."""
    return {
        "model": "claude-sonnet-4-20250514",
        "messages": [
            {"role": "user", "content": "Solve this complex math problem step by step: What is the derivative of x^3 * sin(x)?"}
        ],
        "max_tokens": 4096,
        "anthropic_params": {
            "thinking_enabled": True,
            "thinking_budget_tokens": 2048,
            "top_k": 40,
            "service_tier": "auto"
        }
    }

def create_openai_tool_example():
    """Example of using OpenAI's tool calling feature."""
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        "max_tokens": 1024,
        "openai_params": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        }
    }

def create_multi_provider_example():
    """Example of comparing responses across providers."""
    return {
        "models": ["claude-sonnet-4-20250514", "gpt-4", "gemini-pro"],
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms"}
        ],
        "max_tokens": 2048,
        "anthropic_params": {
            "thinking_enabled": True,
            "service_tier": "priority"
        },
        "openai_params": {
            "temperature": 0.7,
            "frequency_penalty": 0.1
        },
        "gemini_params": {
            "candidate_count": 1,
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
    }
