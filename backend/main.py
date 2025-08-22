import uvicorn
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import asyncio

from config.logging import setup_logging, log_event
from config.settings import get_settings
from core.exceptions import AskManyLLMsException
from models.requests import ChatRequest, ChatMessage

# Import services and dependencies
from providers.registry import ModelRegistry
from providers.adapters.chat_adapter import ChatAdapter
from providers.adapters.embedding_adapter import EmbeddingAdapter
from services.chat_service import ChatService
from services.embedding_service import EmbeddingService
from services.dataset_service import DatasetService
from services.search_services import SearchService
from storage.memory_store import MemoryStorageBackend

# Global service instances
chat_service: ChatService = None
registry: ModelRegistry = None

# OpenAI-compatible models
class OpenAIMessage(BaseModel):
    role: str = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")

class OpenAIChatRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[OpenAIMessage] = Field(..., description="A list of messages comprising the conversation so far")
    temperature: Optional[float] = Field(1.0, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress")

class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global chat_service, registry
    
    # Startup
    setup_logging()
    log_event("app.startup", message="Application starting up")
    
    # Find the config file - try multiple possible locations
    possible_config_paths = [
        "config/models.yaml",           # Current directory
        "../config/models.yaml",        # Parent directory
        "/app/config/models.yaml",      # Absolute path in Docker
        "../../config/models.yaml",     # Two levels up
    ]
    
    config_path = None
    for path in possible_config_paths:
        if os.path.exists(path):
            config_path = path
            log_event("app.config", message=f"Found config at: {path}")
            break
    
    if not config_path:
        # Try using environment variable or default
        config_path = os.getenv("MODELS_CONFIG_PATH", "config/models.yaml")
        log_event("app.config", message=f"Using config path from env or default: {config_path}")
    
    # Initialize services
    try:
        registry = ModelRegistry.from_path(config_path)
        log_event("app.registry", message=f"Loaded {len(registry.model_map)} models, {len(registry.embedding_map)} embedding models")
    except FileNotFoundError:
        log_event("app.error", message=f"Could not find models config at any of: {possible_config_paths}")
        # Create a minimal registry for development
        registry = ModelRegistry({})
        log_event("app.warning", message="Using empty registry - no models configured")
    
    chat_adapter = ChatAdapter()
    embedding_adapter = EmbeddingAdapter()
    storage = MemoryStorageBackend()
    
    chat_service = ChatService(registry, chat_adapter)
    embedding_service = EmbeddingService(registry, embedding_adapter)
    dataset_service = DatasetService(registry, embedding_service, storage)
    search_service = SearchService(registry, embedding_service, storage)
    
    # Store services in app state as well
    app.state.chat_service = chat_service
    app.state.embedding_service = embedding_service
    app.state.dataset_service = dataset_service
    app.state.search_service = search_service
    app.state.registry = registry
    
    yield
    
    # Shutdown
    log_event("app.shutdown", message="Application shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Ask Many Models API",
        description="Multi-model chat and embedding API with semantic search and conversation history",
        version="2.0.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handlers
    @app.exception_handler(AskManyLLMsException)
    async def custom_exception_handler(request: Request, exc: AskManyLLMsException):
        log_event(
            "app.exception",
            path=str(request.url.path),
            method=request.method,
            error=exc.message,
            code=exc.code,
            details=exc.details
        )
        
        status_code = {
            "MODEL_NOT_FOUND": 404,
            "DATASET_NOT_FOUND": 404,
            "VALIDATION_ERROR": 400,
            "PROVIDER_ERROR": 502,
        }.get(exc.code, 500)
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": exc.message,
                "code": exc.code,
                "details": exc.details
            }
        )
    
    # OpenAI-compatible chat completions endpoint
    @app.post("/v1/chat/completions", response_model=OpenAIChatResponse, tags=["OpenAI Compatible"])
    async def openai_chat_completions(request: OpenAIChatRequest):
        """OpenAI-compatible chat completions endpoint."""
        global chat_service, registry
        
        if not chat_service or not registry:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        try:
            log_event("chat.request", model=request.model, message_count=len(request.messages))
            
            # Validate model exists
            if request.model not in registry.model_map:
                available_models = list(registry.model_map.keys())
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model {request.model} not found. Available models: {available_models}"
                )
            
            # Convert to our internal format
            chat_messages = [
                ChatMessage(role=msg.role, content=msg.content) 
                for msg in request.messages
            ]
            
            internal_request = ChatRequest(
                messages=chat_messages,
                models=[request.model],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Process the request
            response = await chat_service.chat_completion(internal_request)
            
            # Convert to OpenAI format
            model_answer = response.answers[request.model]
            
            if model_answer.error:
                log_event("chat.error", model=request.model, error=model_answer.error)
                raise HTTPException(status_code=500, detail=model_answer.error)
            
            choice = OpenAIChoice(
                index=0,
                message=OpenAIMessage(role="assistant", content=model_answer.answer or ""),
                finish_reason="stop"
            )
            
            # Estimate token usage (rough approximation)
            prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
            completion_tokens = len((model_answer.answer or "").split())
            
            response_obj = OpenAIChatResponse(
                id=f"chatcmpl-{int(asyncio.get_event_loop().time())}",
                created=int(asyncio.get_event_loop().time()),
                model=request.model,
                choices=[choice],
                usage=OpenAIUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
            
            log_event("chat.success", model=request.model, tokens=response_obj.usage.total_tokens)
            return response_obj
            
        except AskManyLLMsException as e:
            log_event("chat.error", model=request.model, error=e.message)
            raise HTTPException(status_code=400, detail=e.message)
        except Exception as e:
            log_event("chat.error", model=request.model, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    # Debug endpoint to check available models
    @app.get("/v1/models", tags=["OpenAI Compatible"])
    async def list_models():
        """List available models."""
        global registry
        if not registry:
            return {"models": []}
        
        models = []
        for model_name in registry.model_map.keys():
            models.append({
                "id": model_name,
                "object": "model",
                "created": 1677610602,
                "owned_by": "multi-llm-platform"
            })
        
        return {"object": "list", "data": models}
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        global chat_service, registry
        return {
            "status": "healthy",
            "services_initialized": chat_service is not None and registry is not None,
            "models_count": len(registry.model_map) if registry else 0,
            "embedding_models_count": len(registry.embedding_map) if registry else 0
        }
    
    # Include legacy endpoints for backward compatibility
    try:
        from api.legacy import router as legacy_router
        app.include_router(legacy_router)
        log_event("app.routes", message="Legacy routes included")
    except ImportError as e:
        log_event("app.warning", message=f"Could not import legacy routes: {e}")
    
    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_config=None,  # Use our custom logging
    )