import asyncio
import time
from typing import Any, Dict, List, Optional

from config.logging import log_event
from core.exceptions import ModelNotFoundError, ProviderError
from models.requests import ChatRequest
from models.responses import ChatResponse, ModelAnswer
from providers.registry import ModelRegistry
from providers.adapters.chat_adapter import ChatAdapter


class ChatService:
    """Service for handling chat completions across multiple models."""
    
    def __init__(self, registry: ModelRegistry, chat_adapter: ChatAdapter):
        self.registry = registry
        self.chat_adapter = chat_adapter
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Handle chat completion for multiple models."""
        # Convert request to standardized messages format
        messages = request.to_messages()
        
        chosen_models = request.models or list(self.registry.model_map.keys())
        self._validate_models(chosen_models)
        
        tasks = [self._process_single_model(model_name, messages, request) 
                for model_name in chosen_models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        answers = {}
        for model, result in zip(chosen_models, results):
            if isinstance(result, Exception):
                answers[model] = ModelAnswer(error=str(result))
            else:
                answers[model] = ModelAnswer(answer=result)
        
        # For response, use the last user message as the "prompt" for backwards compatibility
        display_prompt = self._extract_display_prompt(messages)
        
        return ChatResponse(
            prompt=display_prompt,
            models=chosen_models,
            answers=answers
        )
    
    def _extract_display_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Extract the last user message for display purposes."""
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        return user_messages[-1] if user_messages else "No user message found"
    
    async def _process_single_model(
        self, 
        model_name: str, 
        messages: List[Dict[str, str]], 
        request: ChatRequest
    ) -> str:
        """Process chat completion for a single model."""
        provider, model = self.registry.model_map[model_name]
        
        # Get model-specific parameters
        model_params = request.model_params.get(model_name, {})
        temperature = model_params.get("temperature", request.temperature or 0.7)
        max_tokens = model_params.get("max_tokens", request.max_tokens or 8192)
        min_tokens = model_params.get("min_tokens", request.min_tokens)
        
        # Log the conversation context (just first and last messages for brevity)
        log_context = {
            "message_count": len(messages),
            "first_message": messages[0]["content"][:50] + "..." if messages else None,
            "last_message": messages[-1]["content"][:50] + "..." if messages else None
        }
        
        log_event(
            "chat.start",
            provider=provider.name,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            context=log_context,
        )
        
        start_time = time.perf_counter()
        
        try:
            result = await self.chat_adapter.chat_completion(
                provider=provider,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                min_tokens=min_tokens
            )
            
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event(
                "chat.end",
                provider=provider.name,
                model=model_name,
                ok=True,
                duration_ms=duration_ms,
                answer_chars=len(result or ""),
                context=log_context,
            )
            
            return result
            
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event(
                "chat.end",
                provider=provider.name,
                model=model_name,
                ok=False,
                duration_ms=duration_ms,
                error=str(e),
                context=log_context,
            )
            raise ProviderError(provider.name, str(e))
    
    def _validate_models(self, models: List[str]) -> None:
        """Validate that all requested models are available."""
        unknown_models = [m for m in models if m not in self.registry.model_map]
        if unknown_models:
            raise ModelNotFoundError(", ".join(unknown_models))