"""Legacy endpoints for backward compatibility."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import json
import time

from core.dependencies import (
    get_chat_service, get_embedding_service, 
    get_dataset_service, get_search_service, get_model_registry
)
from models.requests import ChatRequest, EmbeddingRequest, SearchRequest, DatasetUploadRequest
from models.responses import ProvidersResponse, EmbeddingModelsResponse
from config.logging import log_event

router = APIRouter(tags=["legacy"])


@router.get("/providers")
async def providers(registry = Depends(get_model_registry)):
    """Legacy providers endpoint."""
    provider_list = []
    for p in registry.providers.values():
        provider_list.append({
            "name": p.name,
            "type": p.type,
            "base_url": p.base_url,
            "models": p.models,
            "embedding_models": p.embedding_models,
            "auth_required": bool(p.api_key_env),
        })
    
    return {"providers": provider_list}


@router.get("/embedding-models")
async def embedding_models(registry = Depends(get_model_registry)):
    """Legacy embedding models endpoint."""
    return {"embedding_models": list(registry.embedding_map.keys())}


@router.get("/datasets")
async def list_datasets(service = Depends(get_dataset_service)):
    """Legacy datasets list endpoint."""
    response = await service.list_datasets()
    return {"datasets": response.datasets}


@router.post("/ask")
async def ask(request: ChatRequest, chat_service = Depends(get_chat_service)):
    """Legacy ask endpoint."""
    response = await chat_service.chat_completion(request)
    
    # Convert to legacy format
    answers = {}
    for model, answer in response.answers.items():
        if answer.error:
            answers[model] = {"error": answer.error}
        else:
            answers[model] = {"answer": answer.answer}
    
    return {
        "prompt": response.prompt,
        "models": response.models,
        "answers": answers
    }


@router.post("/ask/ndjson")
async def ask_ndjson(request: ChatRequest, chat_service = Depends(get_chat_service)):
    """Legacy streaming ask endpoint."""
    
    async def generate_stream():
        chosen_models = request.models or list(chat_service.registry.model_map.keys())
        
        # Send initial metadata
        yield json.dumps({"type": "meta", "models": chosen_models}) + "\n"
        
        # Create queue for results
        out_q: asyncio.Queue[str] = asyncio.Queue()
        
        async def process_model(model_name: str):
            try:
                # Build messages
                messages = [
                    {"role": "system", "content": "Answer clearly and concisely."},
                    {"role": "user", "content": request.prompt},
                ]
                
                provider, model = chat_service.registry.model_map[model_name]
                
                # Get model-specific parameters
                model_params = request.model_params.get(model_name, {})
                temperature = model_params.get("temperature", request.temperature or 0.7)
                max_tokens = model_params.get("max_tokens", request.max_tokens or 8192)
                min_tokens = model_params.get("min_tokens", request.min_tokens)
                
                log_event(
                    "call.start",
                    route="/ask/ndjson",
                    provider=provider.name,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                )
                
                start_time = time.perf_counter()
                
                result = await chat_service.chat_adapter.chat_completion(
                    provider=provider,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens
                )
                
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                
                log_event(
                    "call.end",
                    route="/ask/ndjson",
                    provider=provider.name,
                    model=model_name,
                    ok=True,
                    duration_ms=duration_ms,
                    answer_chars=len(result or ""),
                )
                
                await out_q.put(json.dumps({
                    "type": "chunk",
                    "model": model_name,
                    "answer": result,
                    "latency_ms": duration_ms
                }) + "\n")
                
            except Exception as e:
                duration_ms = int((time.perf_counter() - start_time) * 1000) if 'start_time' in locals() else 0
                
                log_event(
                    "call.end",
                    route="/ask/ndjson",
                    provider=provider.name,
                    model=model_name,
                    ok=False,
                    duration_ms=duration_ms,
                    error=str(e),
                )
                
                await out_q.put(json.dumps({
                    "type": "chunk",
                    "model": model_name,
                    "error": str(e),
                    "latency_ms": duration_ms
                }) + "\n")
        
        # Start all model tasks
        tasks = [asyncio.create_task(process_model(m)) for m in chosen_models]
        pending = set(tasks)
        
        # Stream results as they complete
        while pending or not out_q.empty():
            try:
                item = await asyncio.wait_for(out_q.get(), timeout=0.1)
                yield item
            except asyncio.TimeoutError:
                pending = {t for t in tasks if not t.done()}
        
        # Send completion signal
        yield json.dumps({"type": "done"}) + "\n"
    
    return StreamingResponse(generate_stream(), media_type="application/x-ndjson")


@router.post("/embeddings")
async def generate_embeddings(request: EmbeddingRequest, service = Depends(get_embedding_service)):
    """Legacy embeddings endpoint."""
    return await service.generate_embeddings(request)


@router.post("/upload-dataset")
async def upload_dataset(request: DatasetUploadRequest, service = Depends(get_dataset_service)):
    """Legacy dataset upload endpoint."""
    return await service.upload_dataset(request)


@router.post("/search")
async def semantic_search(request: SearchRequest, service = Depends(get_search_service)):
    """Legacy search endpoint."""
    return await service.semantic_search(request)


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, service = Depends(get_dataset_service)):
    """Legacy dataset deletion endpoint."""
    return await service.delete_dataset(dataset_id)