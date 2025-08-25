import uvicorn
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import asyncio
import time
import json

from config.logging import setup_logging, log_event
from config.settings import get_settings
from core.exceptions import AskManyLLMsException, ProviderError, ModelNotFoundError
from models.requests import ChatRequest, ChatMessage as RequestChatMessage
from models.responses import ChatResponse, ModelAnswer
from providers.registry import ModelRegistry
from providers.adapters.enhanced_chat_adapter import EnhancedChatAdapter
from providers.adapters.embedding_adapter import EmbeddingAdapter
from services.enhanced_chat_service import EnhancedChatService
from services.embedding_service import EmbeddingService
from services.dataset_service import DatasetService
from services.search_services import SearchService
from storage.memory_store import MemoryStorageBackend

# NEW: import the enhanced routes setup (adds /v2, including /v2/search/multi)
from api.enhanced_routes import setup_enhanced_chat_routes

# Global services
services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global services
    
    # Startup
    setup_logging()
    log_event("app.startup", message="Application starting up")
    
    # Find config
    config_paths = [
        "config/models.yaml",
        "../config/models.yaml",
        "/app/config/models.yaml",
        "/config/models.yaml",  # Docker mount point
        os.getenv("MODELS_CONFIG_PATH", "config/models.yaml"),
        os.getenv("MODELS_CONFIG", "/config/models.yaml")
    ]
    
    config_path = None
    for path in config_paths:
        if os.path.exists(path):
            config_path = path
            log_event("app.config", message=f"Found config at: {path}")
            break
    
    if not config_path:
        log_event("app.error", message="No config found, using empty registry")
        registry = ModelRegistry({})
    else:
        try:
            registry = ModelRegistry.from_path(config_path)
            log_event("app.registry", 
                     models=len(registry.model_map), 
                     embeddings=len(registry.embedding_map))
        except Exception as e:
            log_event("app.error", message=f"Failed to load config: {e}")
            registry = ModelRegistry({})
    
    # Initialize services
    chat_adapter = EnhancedChatAdapter()
    embedding_adapter = EmbeddingAdapter()
    storage = MemoryStorageBackend()
    
    services['chat'] = EnhancedChatService(registry, chat_adapter)
    services['embedding'] = EmbeddingService(registry, embedding_adapter)
    services['dataset'] = DatasetService(registry, services['embedding'], storage)
    services['search'] = SearchService(registry, services['embedding'], storage)
    services['registry'] = registry
    
    # Store in app state
    app.state.services = services
    app.state.registry = registry
    app.state.search_service = services['search']
    app.state.embedding_service = services['embedding']  
    app.state.memory_store = storage                    
    
    # NEW: include the enhanced router (adds /v2 endpoints incl. /v2/search/multi)
    app.include_router(setup_enhanced_chat_routes(services['chat'], registry))
    
    yield
    
    # Shutdown
    log_event("app.shutdown", message="Application shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Multi-Model API Platform",
        description="Unified API for multiple LLM providers with enhanced parameters",
        version="3.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS - Allow all for development (restrict in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True
    )
    
    # Exception handlers
    @app.exception_handler(AskManyLLMsException)
    async def handle_domain_exception(request: Request, exc: AskManyLLMsException):
        status_map = {
            "MODEL_NOT_FOUND": 404,
            "DATASET_NOT_FOUND": 404,
            "VALIDATION_ERROR": 400,
            "PROVIDER_ERROR": 502,
        }
        return JSONResponse(
            status_code=status_map.get(exc.code, 500),
            content={"error": exc.message, "code": exc.code, "details": exc.details}
        )
    
    # ==================== HEALTH & INFO ROUTES ====================
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "Multi-Model API Platform", "version": "3.0.0"}
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        services_map = getattr(app.state, 'services', {})
        registry = getattr(app.state, 'registry', None)
        return {
            "status": "healthy",
            "services": list(services_map.keys()),
            "models": len(registry.model_map) if registry else 0,
            "embedding_models": len(registry.embedding_map) if registry else 0
        }
    
    @app.get("/providers")
    @app.get("/api/providers")
    async def list_providers():
        """List all configured providers."""
        registry = app.state.registry
        providers_info = []
        
        for name, provider in registry.providers.items():
            providers_info.append({
                "name": name,
                "type": provider.type,
                "base_url": provider.base_url,
                "models": provider.models,
                "embedding_models": provider.embedding_models,
                "auth_required": bool(provider.api_key_env)
            })
        
        return {"providers": providers_info}
    
    @app.get("/models")
    @app.get("/api/models")
    @app.get("/v1/models")
    async def list_models():
        """List all available models (OpenAI compatible)."""
        registry = app.state.registry
        models = []
        
        for model_name in registry.model_map.keys():
            provider, _ = registry.model_map[model_name]
            models.append({
                "id": model_name,
                "object": "model",
                "provider": provider.name,
                "provider_type": provider.type,
                "created": 1677610602,
                "owned_by": provider.name
            })
        
        return {"object": "list", "data": models}
    
    # ==================== ENHANCED CHAT ROUTES ====================
    
    @app.post("/v2/chat/completions/enhanced")
    async def enhanced_chat_completions(request: Request):
        """Enhanced chat endpoint with provider-specific parameters."""
        chat_service = app.state.services['chat']
        registry = app.state.registry
        
        try:
            body = await request.json()
            log_event("enhanced_chat.request", endpoint="/v2/chat/completions/enhanced", body_keys=list(body.keys()))
            
            # Convert messages to proper format
            messages = body.get("messages", [])
            if not messages:
                raise HTTPException(status_code=400, detail="No messages provided")
            
            # Convert to ChatMessage objects
            from models.enhanced_requests import EnhancedChatRequest, ChatMessage
            
            chat_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    chat_messages.append(ChatMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", "")
                    ))
            
            # Get models
            models = body.get("models", [])
            if not models:
                models = list(registry.model_map.keys())
            
            # Validate models exist
            unknown_models = [m for m in models if m not in registry.model_map]
            if unknown_models:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Unknown models: {unknown_models}"
                )
            
            # Create enhanced request
            enhanced_request = EnhancedChatRequest(
                messages=chat_messages,
                models=models,
                temperature=body.get("temperature"),
                max_tokens=body.get("max_tokens"),
                min_tokens=body.get("min_tokens"),
                system=body.get("system"),
                anthropic_params=body.get("anthropic_params"),
                openai_params=body.get("openai_params"),
                gemini_params=body.get("gemini_params"),
                ollama_params=body.get("ollama_params"),
                provider_params=body.get("provider_params"),
                model_params=body.get("model_params")
            )
            
            # Process request
            response = await chat_service.chat_completion(enhanced_request)
            
            # Return response
            return response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            
        except HTTPException:
            raise
        except Exception as e:
            log_event("enhanced_chat.error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/chat")
    @app.post("/api/chat")
    @app.post("/v1/chat/completions")
    @app.post("/v2/chat/completions")
    async def unified_chat(request: Request):
        """Unified chat endpoint supporting multiple formats."""
        chat_service = app.state.services['chat']
        registry = app.state.registry
        
        try:
            body = await request.json()
            
            # For v1/chat/completions, use OpenAI format
            if "/v1/chat/completions" in str(request.url):
                # OpenAI single-model format
                model = body.get("model")
                if not model:
                    raise HTTPException(status_code=400, detail="Model not specified")
                
                if model not in registry.model_map:
                    raise HTTPException(status_code=404, detail=f"Model {model} not found")
                
                messages = body.get("messages", [])
                
                # Convert to internal format
                chat_messages = []
                for msg in messages:
                    chat_messages.append(RequestChatMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", "")
                    ))
                
                # Create request
                chat_request = ChatRequest(
                    messages=chat_messages,
                    models=[model],
                    temperature=body.get("temperature"),
                    max_tokens=body.get("max_tokens"),
                )
                
                # Process
                response = await chat_service.chat_completion(chat_request)
                
                # Format as OpenAI response
                answer = response.answers.get(model, ModelAnswer(error="No response"))
                
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": answer.answer or ""
                        },
                        "finish_reason": "stop" if not answer.error else "error"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
            
            else:
                # Multi-model format
                if "prompt" in body:
                    # Legacy prompt format
                    prompt = body["prompt"]
                    models = body.get("models", list(registry.model_map.keys()))
                    
                    chat_request = ChatRequest(
                        prompt=prompt,
                        models=models,
                        temperature=body.get("temperature"),
                        max_tokens=body.get("max_tokens"),
                        model_params=body.get("model_params")
                    )
                else:
                    # Messages format
                    messages = body.get("messages", [])
                    models = body.get("models", [])
                    if not models and "model" in body:
                        models = [body["model"]]
                    if not models:
                        models = list(registry.model_map.keys())
                    
                    chat_messages = []
                    for msg in messages:
                        chat_messages.append(RequestChatMessage(
                            role=msg.get("role", "user"),
                            content=msg.get("content", "")
                        ))
                    
                    chat_request = ChatRequest(
                        messages=chat_messages,
                        models=models,
                        temperature=body.get("temperature"),
                        max_tokens=body.get("max_tokens"),
                        model_params=body.get("model_params")
                    )
                
                # Process request
                response = await chat_service.chat_completion(chat_request)
                return response.dict()
                
        except HTTPException:
            raise
        except Exception as e:
            log_event("chat.error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==================== EMBEDDING ROUTES ====================
    
    @app.post("/embeddings")
    @app.post("/api/embeddings")
    @app.post("/v1/embeddings")
    async def generate_embeddings(request: Request):
        """Generate embeddings endpoint."""
        embedding_service = app.state.services['embedding']
        
        try:
            body = await request.json()
            
            # Handle different input formats
            texts = body.get("texts") or body.get("input", [])
            if isinstance(texts, str):
                texts = [texts]
            
            model = body.get("model", body.get("embedding_model"))
            if not model:
                raise HTTPException(status_code=400, detail="Model not specified")
            
            # Create request object
            from models.requests import EmbeddingRequest
            embedding_request = EmbeddingRequest(texts=texts, model=model)
            
            # Generate embeddings
            response = await embedding_service.generate_embeddings(embedding_request)
            
            # Format response
            if "/v1/embeddings" in str(request.url):
                # OpenAI compatible format
                return {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": emb}
                        for i, emb in enumerate(response.embeddings)
                    ],
                    "model": model,
                    "usage": response.usage.dict() if response.usage else {}
                }
            else:
                return response.dict()
                
        except HTTPException:
            raise
        except Exception as e:
            log_event("embeddings.error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==================== DATASET ROUTES ====================
    
    @app.post("/upload-dataset")
    @app.post("/datasets")
    @app.post("/api/datasets")
    async def upload_dataset(request: Request):
        """Upload a dataset with embeddings."""
        dataset_service = app.state.services['dataset']
        
        try:
            body = await request.json()
            
            from models.requests import DatasetUploadRequest
            upload_request = DatasetUploadRequest(**body)
            
            response = await dataset_service.upload_dataset(upload_request)
            return response.dict()
            
        except HTTPException:
            raise
        except Exception as e:
            log_event("dataset.error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/datasets")
    @app.get("/api/datasets")
    async def list_datasets():
        """List all datasets."""
        dataset_service = app.state.services['dataset']
        
        try:
            response = await dataset_service.list_datasets()
            return response.dict()
        except Exception as e:
            log_event("dataset.error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/datasets/{dataset_id}")
    @app.delete("/api/datasets/{dataset_id}")
    async def delete_dataset(dataset_id: str):
        """Delete a dataset."""
        dataset_service = app.state.services['dataset']
        
        try:
            response = await dataset_service.delete_dataset(dataset_id)
            return response
        except Exception as e:
            log_event("dataset.error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==================== SEARCH ROUTES ====================
    
    @app.post("/search")
    @app.post("/api/search")
    async def semantic_search(request: Request):
        """Perform semantic search."""
        search_service = app.state.services['search']
        
        try:
            body = await request.json()
            
            from models.requests import SearchRequest
            search_request = SearchRequest(**body)
            
            response = await search_service.semantic_search(search_request)
            return response.dict()
            
        except HTTPException:
            raise
        except Exception as e:
            log_event("search.error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==================== MODEL INFO ROUTES ====================
    
    @app.get("/model/{model_name}/capabilities")
    @app.get("/api/model/{model_name}/capabilities")
    async def get_model_capabilities(model_name: str):
        """Get capabilities for a specific model."""
        chat_service = app.state.services['chat']
        
        try:
            capabilities = chat_service.get_model_capabilities(model_name)
            return capabilities
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/providers/features")
    @app.get("/api/providers/features")
    async def get_provider_features():
        """Get feature matrix for all providers."""
        registry = app.state.registry
        features = {}
        
        for name, provider in registry.providers.items():
            provider_features = {
                "name": name,
                "type": provider.type,
                "models": provider.models,
                "embedding_models": provider.embedding_models,
                "features": {}
            }
            
            # Define features based on provider type
            if provider.type == "anthropic":
                provider_features["features"] = {
                    "thinking": True,
                    "tools": True,
                    "streaming": True,
                    "system_messages": True,
                    "stop_sequences": True,
                    "service_tiers": True,
                    "large_context": True,
                    "max_tokens": 200000
                }
            elif provider.type == "openai":
                provider_features["features"] = {
                    "function_calling": True,
                    "tools": True,
                    "streaming": True,
                    "json_mode": True,
                    "seed": True,
                    "logit_bias": True,
                    "system_messages": True,
                    "max_tokens": 128000
                }
            elif provider.type == "deepseek":
                provider_features["features"] = {
                    "streaming": True,
                    "system_messages": True,
                    "temperature": True,
                    "top_p": True,
                    "max_tokens": 65536,
                    "function_calling": True
                }
            elif provider.type == "ollama":
                provider_features["features"] = {
                    "local_deployment": True,
                    "custom_models": True,
                    "mirostat": True,
                    "streaming": True,
                    "system_messages": True,
                    "json_mode": True,
                    "max_tokens": 32768
                }
            elif provider.type == "gemini":
                provider_features["features"] = {
                    "multimodal": True,
                    "safety_settings": True,
                    "tools": True,
                    "streaming": True,
                    "system_messages": True,
                    "large_context": True,
                    "max_tokens": 1000000
                }
            
            features[name] = provider_features
        
        return {"providers": features}
    
    return app


# Create the app instance
app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    # Use PORT from environment, fallback to 8080
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_config=None
    )
