# routers/langgraph.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from services.langgraph_service import LangGraphService
from pydantic import BaseModel, Field

router = APIRouter(prefix="/langgraph", tags=["langgraph"])

# Request/Response models
class SingleChatRequest(BaseModel):
    wire: str = Field(..., description="Model wire identifier (e.g., 'openai:gpt-4o-mini')")
    messages: List[Dict[str, str]] = Field(default_factory=list, description="Chat messages")
    model_params: Optional[Dict[str, Any]] = Field(default=None, description="Model parameters")
    thread_id: Optional[str] = Field(default="default", description="Thread ID for conversation memory")

class MultiChatRequest(BaseModel):
    targets: List[str] = Field(..., description="List of model wire identifiers")
    messages: List[Dict[str, str]] = Field(default_factory=list, description="Chat messages")
    per_model_params: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Per-model parameters")
    thread_id: Optional[str] = Field(default="compare", description="Thread ID for conversation memory")

# Dependency injection
def get_langgraph_service(request: Request) -> LangGraphService:
    """Get or create the LangGraph service instance."""
    if not hasattr(request.app.state, "_langgraph_service"):
        request.app.state._langgraph_service = LangGraphService()
    return request.app.state._langgraph_service

# ---------------------------
# Routes
# ---------------------------
@router.post("/chat/single/stream")
async def chat_single_stream(
    payload: SingleChatRequest,
    request: Request,
    service: LangGraphService = Depends(get_langgraph_service)
):
    """Stream chat responses from a single model."""
    # Get registry and memory backend from app state
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(500, "Model registry is not initialized")
    
    memory_backend = getattr(request.app.state, "graph_memory", None)
    
    # Create async generator for streaming
    async def generate():
        async for chunk in service.stream_single_chat(
            registry=registry,
            wire=payload.wire,
            messages=payload.messages,
            model_params=payload.model_params,
            thread_id=payload.thread_id,
            memory_backend=memory_backend,
        ):
            yield chunk
    
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream",
        headers=LangGraphService.STREAM_HEADERS
    )

@router.post("/chat/multi/stream")
async def chat_multi_stream(
    payload: MultiChatRequest,
    request: Request,
    service: LangGraphService = Depends(get_langgraph_service)
):
    """Stream chat responses from multiple models for comparison."""
    # Get registry and memory backend from app state
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(500, "Model registry is not initialized")
    
    memory_backend = getattr(request.app.state, "graph_memory", None)
    
    # Create async generator for streaming
    async def generate():
        async for chunk in service.stream_multi_chat(
            registry=registry,
            targets=payload.targets,
            messages=payload.messages,
            per_model_params=payload.per_model_params,
            thread_id=payload.thread_id,
            memory_backend=memory_backend,
        ):
            yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream", 
        headers=LangGraphService.STREAM_HEADERS
    )

