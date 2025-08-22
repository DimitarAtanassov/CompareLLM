import asyncio
import time
from typing import Any, Dict, List, Optional

from config.logging import log_event
from core.exceptions import ModelNotFoundError, ProviderError
from models.enhanced_requests import EnhancedChatRequest
from models.responses import ChatResponse, ModelAnswer
from providers.registry import ModelRegistry
from providers.adapters.enhanced_chat_adapter import EnhancedChatAdapter


class EnhancedChatService:
    """Enhanced service for handling chat completions with provider-specific parameters."""
    
    def __init__(self, registry: ModelRegistry, chat_adapter: EnhancedChatAdapter):
        self.registry = registry
        self.chat_adapter = chat_adapter
    
    async def chat_completion(self, request: EnhancedChatRequest) -> ChatResponse:
        """Handle chat completion for multiple models with enhanced parameters."""
        # Convert request to standardized messages format
        messages = request.to_messages()
        
        chosen_models = request.models or list(self.registry.model_map.keys())
        self._validate_models(chosen_models)
        
        # Log the enhanced request details
        log_event(
            "enhanced_chat.start",
            models=chosen_models,
            message_count=len(messages),
            has_anthropic_params=request.anthropic_params is not None,
            has_openai_params=request.openai_params is not None,
            has_gemini_params=request.gemini_params is not None,
            has_ollama_params=request.ollama_params is not None,
            has_provider_params=request.provider_params is not None
        )
        
        tasks = [self._process_single_model(model_name, messages, request) 
                for model_name in chosen_models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        answers = {}
        for model, result in zip(chosen_models, results):
            if isinstance(result, Exception):
                log_event("model.error", model=model, error=str(result))
                answers[model] = ModelAnswer(error=str(result))
            else:
                answers[model] = ModelAnswer(answer=result)
        
        # For response, use the last user message as the "prompt" for backwards compatibility
        display_prompt = self._extract_display_prompt(messages)
        
        log_event(
            "enhanced_chat.complete",
            models=chosen_models,
            success_count=sum(1 for ans in answers.values() if not ans.error),
            error_count=sum(1 for ans in answers.values() if ans.error)
        )
        
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
        request: EnhancedChatRequest
    ) -> str:
        """Process chat completion for a single model with enhanced parameters."""
        provider, model = self.registry.model_map[model_name]
        
        # Get base parameters
        temperature = request.temperature or 0.7
        max_tokens = request.max_tokens or 8192
        min_tokens = request.min_tokens
        
        # Get provider-specific parameters
        provider_params = request.get_provider_params(provider.type, model_name)
        
        # Override base parameters with provider-specific ones if present
        if "temperature" in provider_params:
            temperature = provider_params["temperature"]
        if "max_tokens" in provider_params:
            max_tokens = provider_params["max_tokens"]
        if "min_tokens" in provider_params:
            min_tokens = provider_params["min_tokens"]
        
        # Log the conversation context and parameters
        log_context = {
            "message_count": len(messages),
            "first_message": messages[0]["content"][:50] + "..." if messages else None,
            "last_message": messages[-1]["content"][:50] + "..." if messages else None,
            "provider_type": provider.type,
            "provider_params_count": len(provider_params),
            "provider_params_keys": list(provider_params.keys()) if provider_params else []
        }
        
        log_event(
            "enhanced_chat.model_start",
            provider=provider.name,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            context=log_context,
        )
        
        start_time = time.perf_counter()
        
        try:
            # Use enhanced chat adapter with provider parameters
            result = await self.chat_adapter.chat_completion(
                provider=provider,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                provider_params=provider_params
            )
            
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event(
                "enhanced_chat.model_end",
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
                "enhanced_chat.model_end",
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
    
    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """
        Get capabilities and configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model capabilities and configuration
        """
        if model_name not in self.registry.model_map:
            raise ModelNotFoundError(model_name)
        
        provider, model = self.registry.model_map[model_name]
        
        capabilities = {
            "model_name": model_name,
            "provider_name": provider.name,
            "provider_type": provider.type,
            "base_url": provider.base_url,
            "supports_streaming": True,  # Most models support streaming
            "supports_system_messages": True,
        }
        
        # Add provider-specific capabilities
        if provider.type == "anthropic":
            from providers.anthropic_params import get_model_config
            model_config = get_model_config(model_name)
            if model_config:
                capabilities.update({
                    "supports_thinking": model_config.supports_thinking,
                    "supports_tools": model_config.supports_tools,
                    "max_context_tokens": model_config.max_context_tokens,
                    "default_rpm": model_config.default_rpm,
                    "default_tpm": model_config.default_tpm,
                })
            else:
                capabilities.update({
                    "supports_thinking": True,  # Assume newer models support thinking
                    "supports_tools": True,
                    "max_context_tokens": 200000,  # Default for Claude models
                })
        elif provider.type == "openai":
            capabilities.update({
                "supports_tools": True,
                "supports_function_calling": True,
                "supports_json_mode": True,
                "max_context_tokens": 128000,  # Default for GPT-4
            })
        elif provider.type == "gemini":
            capabilities.update({
                "supports_tools": True,
                "supports_safety_settings": True,
                "supports_multimodal": True,
                "max_context_tokens": 1000000,  # Gemini has large context
            })
        elif provider.type == "ollama":
            capabilities.update({
                "supports_local_deployment": True,
                "supports_custom_models": True,
                "supports_mirostat": True,
            })
        
        return capabilities
    
    async def validate_request_for_model(
        self, 
        model_name: str, 
        request: EnhancedChatRequest
    ) -> Dict[str, Any]:
        """
        Validate that a request is compatible with a specific model.
        
        Args:
            model_name: Name of the model
            request: Enhanced chat request
            
        Returns:
            Dictionary with validation results and any warnings
        """
        capabilities = self.get_model_capabilities(model_name)
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        provider_type = capabilities["provider_type"]
        
        # Check Anthropic-specific validations
        if provider_type == "anthropic" and request.anthropic_params:
            if (request.anthropic_params.thinking_enabled and 
                not capabilities.get("supports_thinking", False)):
                validation_result["warnings"].append(
                    f"Model {model_name} may not support extended thinking"
                )
            
            if (request.anthropic_params.thinking_budget_tokens and 
                request.anthropic_params.thinking_budget_tokens > request.max_tokens):
                validation_result["errors"].append(
                    "Thinking budget tokens cannot exceed max_tokens"
                )
                validation_result["valid"] = False
        
        # Check token limits
        max_context = capabilities.get("max_context_tokens")
        if max_context and request.max_tokens > max_context:
            validation_result["warnings"].append(
                f"Requested max_tokens ({request.max_tokens}) exceeds model context limit ({max_context})"
            )
        
        # Check provider parameter compatibility
        provider_params = request.get_provider_params(provider_type, model_name)
        if provider_params:
            # Validate provider-specific parameter compatibility
            if provider_type == "anthropic":
                self._validate_anthropic_params(provider_params, validation_result)
            elif provider_type == "openai":
                self._validate_openai_params(provider_params, validation_result)
        
        return validation_result
    
    def _validate_anthropic_params(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate Anthropic-specific parameters."""
        if "thinking" in params:
            thinking = params["thinking"]
            if isinstance(thinking, dict) and thinking.get("type") == "enabled":
                budget = thinking.get("budget_tokens", 0)
                if budget < 1024:
                    result["errors"].append("Thinking budget_tokens must be at least 1024")
                    result["valid"] = False
        
        if "top_k" in params and params["top_k"] <= 0:
            result["errors"].append("top_k must be positive")
            result["valid"] = False
        
        if "top_p" in params and not (0 < params["top_p"] <= 1):
            result["errors"].append("top_p must be between 0 and 1")
            result["valid"] = False
    
    def _validate_openai_params(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate OpenAI-specific parameters."""
        if "frequency_penalty" in params:
            penalty = params["frequency_penalty"]
            if not (-2 <= penalty <= 2):
                result["errors"].append("frequency_penalty must be between -2 and 2")
                result["valid"] = False
        
        if "presence_penalty" in params:
            penalty = params["presence_penalty"]
            if not (-2 <= penalty <= 2):
                result["errors"].append("presence_penalty must be between -2 and 2")
                result["valid"] = False