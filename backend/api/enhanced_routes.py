from fastapi import APIRouter, HTTPException, Depends
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import asyncio
import json

from models.enhanced_requests import EnhancedChatRequest, EnhancedOpenAIChatRequest, AnthropicProviderParams
from models.responses import ChatResponse, ModelAnswer
from services.chat_service import EnhancedChatService
from providers.registry import ModelRegistry
from providers.adapters.chat_adapter import EnhancedChatAdapter
from core.exceptions import AskManyLLMsException

router = APIRouter(prefix="/v2", tags=["Enhanced API"])

# Response models
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


def setup_enhanced_chat_routes(chat_service: EnhancedChatService, registry: ModelRegistry):
    """Setup enhanced chat completion routes with dependency injection."""
    
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
            
            # Estimate token usage
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
            from models.requests import create_anthropic_request
            
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


# Example usage functions for documentation
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