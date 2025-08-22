from fastapi import APIRouter, HTTPException
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import asyncio
import json

from models.requests import ChatRequest, ChatMessage
from services.chat_service import ChatService
from providers.registry import ModelRegistry
from providers.adapters.chat_adapter import ChatAdapter
from core.exceptions import AskManyLLMsException

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])

# OpenAI-compatible request/response models
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

# Multi-model request (extension to OpenAI format)
class MultiModelChatRequest(BaseModel):
    models: List[str] = Field(..., description="List of model IDs to use")
    messages: List[OpenAIMessage] = Field(..., description="A list of messages comprising the conversation so far")
    temperature: Optional[float] = Field(1.0, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    model_params: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict, description="Per-model parameter overrides")

class MultiModelChoice(BaseModel):
    model: str
    message: OpenAIMessage
    error: Optional[str] = None
    finish_reason: Optional[str] = None
    latency_ms: Optional[int] = None

class MultiModelChatResponse(BaseModel):
    id: str
    object: str = "multi_model.chat.completion"
    created: int
    models: List[str]
    choices: List[MultiModelChoice]


def setup_chat_routes(chat_service: ChatService, registry: ModelRegistry):
    """Setup chat completion routes with dependency injection."""
    
    @router.post("/chat/completions", response_model=OpenAIChatResponse)
    async def openai_chat_completions(request: OpenAIChatRequest):
        """OpenAI-compatible chat completions endpoint."""
        try:
            # Validate model exists
            if request.model not in registry.model_map:
                raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
            
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
                raise HTTPException(status_code=500, detail=model_answer.error)
            
            choice = OpenAIChoice(
                index=0,
                message=OpenAIMessage(role="assistant", content=model_answer.answer or ""),
                finish_reason="stop"
            )
            
            # Estimate token usage (rough approximation)
            prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
            completion_tokens = len((model_answer.answer or "").split())
            
            return OpenAIChatResponse(
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
            
        except AskManyLLMsException as e:
            raise HTTPException(status_code=400, detail=e.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/chat/completions/multi", response_model=MultiModelChatResponse)
    async def multi_model_chat_completions(request: MultiModelChatRequest):
        """Multi-model chat completions endpoint (extension to OpenAI format)."""
        try:
            # Convert to our internal format
            chat_messages = [
                ChatMessage(role=msg.role, content=msg.content) 
                for msg in request.messages
            ]
            
            internal_request = ChatRequest(
                messages=chat_messages,
                models=request.models,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                model_params=request.model_params or {}
            )
            
            # Process the request
            response = await chat_service.chat_completion(internal_request)
            
            # Convert to multi-model format
            choices = []
            for model in request.models:
                model_answer = response.answers[model]
                choice = MultiModelChoice(
                    model=model,
                    message=OpenAIMessage(
                        role="assistant", 
                        content=model_answer.answer or ""
                    ),
                    error=model_answer.error,
                    finish_reason="stop" if not model_answer.error else "error",
                    latency_ms=model_answer.latency_ms
                )
                choices.append(choice)
            
            return MultiModelChatResponse(
                id=f"multicmpl-{int(asyncio.get_event_loop().time())}",
                created=int(asyncio.get_event_loop().time()),
                models=request.models,
                choices=choices
            )
            
        except AskManyLLMsException as e:
            raise HTTPException(status_code=400, detail=e.message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router