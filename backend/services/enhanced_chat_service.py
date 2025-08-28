# app/backend/services/enhaced_chat_service.py
import asyncio
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Request

from config.logging import log_event
from core.exceptions import ModelNotFoundError, ProviderError
from models.enhanced_requests import EnhancedChatRequest
from models.responses import ChatResponse, ModelAnswer
from providers.registry import ModelRegistry, get_provider_for_model
from providers.adapters.enhanced_chat_adapter import EnhancedChatAdapter

def _get_wire(provider) -> str:
    wire = getattr(provider, "wire", None)
    if wire:
        return str(wire).lower()
    ptype = getattr(provider, "type", None)
    return str(ptype or "").lower()

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
        
        answers: Dict[str, ModelAnswer] = {}
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
    

        # ---------- Decide streaming vs non-stream ----------
    def is_openai_stream_model(m: str) -> bool:
        """
        Return True if model is an OpenAI GPT-family or O-preview variant
        that should always use streaming.
        Matches: gpt-3.5, gpt-4, gpt-4o, gpt-5, gpt-5o, o1-preview, o1-mini, o2-*
        """
        ml = m.lower()
        return bool(re.match(r"^(gpt-|o\d+)", ml))


    async def _process_single_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        request: EnhancedChatRequest
    ) -> str:
        provider, model = self.registry.model_map[model_name]

        mp = (request.model_params or {}).get(model_name, {}) or {}
        temperature = mp.get("temperature", request.temperature)
        max_tokens = mp.get("max_tokens", request.max_tokens)
        min_tokens = mp.get("min_tokens", request.min_tokens)

        wire = _get_wire(provider)

        # Build provider_params exactly once
        provider_params: Dict[str, Any] = {}
        if wire == "openai" and getattr(request, "openai_params", None):
            provider_params["openai_params"] = request.openai_params
        elif wire == "anthropic" and getattr(request, "anthropic_params", None):
            provider_params.update(request.anthropic_params)
        elif wire == "gemini" and getattr(request, "gemini_params", None):
            provider_params.update(request.gemini_params)
        elif wire == "ollama" and getattr(request, "ollama_params", None):
            provider_params.update(request.ollama_params)

        start_time = time.perf_counter()
        try:
            # Decide: stream via adapter.stream_chat, else adapter.chat_completion
            if wire == "openai" and self.is_openai_stream_model(model_name):
                chunks: List[str] = []
                async for delta in self.chat_adapter.stream_chat(
                    provider=provider,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    provider_params=provider_params,
                    timeout_s=180,
                ):
                    chunks.append(delta)
                result = "".join(chunks)
            else:
                result = await self.chat_adapter.chat_completion(
                    provider=provider,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    provider_params=provider_params,
                    timeout_s=180,
                )

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event("enhanced_chat.model_end",
                    provider=provider.name, model=model_name,
                    ok=True, duration_ms=duration_ms,
                    answer_chars=len(result or ""))
            return result

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event("enhanced_chat.model_end",
                    provider=provider.name, model=model_name,
                    ok=False, duration_ms=duration_ms, error=str(e))
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
        provider_wire = _get_wire(self.registry.model_map[model_name][0])

        # Effective max_tokens (global → per-model)
        mp: Dict[str, Any] = (request.model_params or {}).get(model_name, {}) or {}
        effective_max_tokens: Optional[int] = mp.get("max_tokens", request.max_tokens)
        
        # Check Anthropic-specific validations (by wire)
        if provider_wire == "anthropic" and request.anthropic_params:
            thinking_enabled = getattr(request.anthropic_params, "thinking_enabled", None)
            thinking_budget  = getattr(request.anthropic_params, "thinking_budget_tokens", None)

            if thinking_enabled and not capabilities.get("supports_thinking", False):
                validation_result["warnings"].append(
                    f"Model {model_name} may not support extended thinking"
                )

            if thinking_budget is not None and effective_max_tokens is not None:
                if thinking_budget > effective_max_tokens:
                    validation_result["errors"].append(
                        "Thinking budget tokens cannot exceed max_tokens"
                    )
                    validation_result["valid"] = False

        
        # Check token limits (guard None)
        max_context = capabilities.get("max_context_tokens")
        if max_context and (effective_max_tokens is not None) and (effective_max_tokens > max_context):
            validation_result["warnings"].append(
                f"Requested max_tokens ({effective_max_tokens}) exceeds model context limit ({max_context})"
            )
        
        # Check provider parameter compatibility
        if provider_wire == "openai" and getattr(request, "openai_params", None):
            openai_params_dict = (
                vars(request.openai_params)
                if hasattr(request.openai_params, "__dict__")
                else request.openai_params
            )
            self._validate_openai_params(openai_params_dict or {}, validation_result)

        elif provider_wire == "anthropic" and getattr(request, "anthropic_params", None):
            # Convert to dict if it's a pydantic model
            anth_params = (
                request.anthropic_params.model_dump(exclude_none=True)
                if hasattr(request.anthropic_params, "model_dump")
                else (vars(request.anthropic_params)
                    if hasattr(request.anthropic_params, "__dict__")
                    else request.anthropic_params)
            )
            self._validate_anthropic_params(anth_params or {}, validation_result)
        
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
    
    async def stream_answers(self, req: EnhancedChatRequest) -> AsyncIterator[Dict[str, Any]]:
        """
        Yields {"model": str, "delta": str, "latency_ms": int}
        """
        # Normalize request to messages/models just like non-streaming path
        messages = req.to_messages() if hasattr(req, "to_messages") else (req.messages or [])
        models = req.models or list(self.registry.model_map.keys())

        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        done = asyncio.Event()

        async def run_one(model_name: str):
            # Resolve provider/model from our registry (keeps consistent with non-streaming)
            try:
                provider, actual_model = self.registry.model_map[model_name]
            except KeyError as e:
                await queue.put({"model": model_name, "delta": f"[error] Unknown model {model_name}", "latency_ms": 0})
                return

            # Effective temps/tokens (global → per-model override), same logic as chat_completion
            mp: Dict[str, Any] = (getattr(req, "model_params", {}) or {}).get(model_name, {}) or {}
            temperature: Optional[float] = mp.get("temperature", getattr(req, "temperature", None))
            max_tokens: Optional[int]   = mp.get("max_tokens",  getattr(req, "max_tokens", None))
            min_tokens: Optional[int]   = mp.get("min_tokens",  getattr(req, "min_tokens", None))

            # Provider-specific bundle (anthropic/openai/gemini/ollama)
            params = self._build_provider_params(model_name, req)

            t0 = time.perf_counter()
            try:
                # NOTE: fixed attribute name: self.chat_adapter (not self._adapter)
                async for delta in self.chat_adapter.stream_chat(
                    provider=provider,
                    model=actual_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    provider_params=params,
                ):
                    await queue.put(
                        {
                            "model": model_name,
                            "delta": delta,
                            "latency_ms": int((time.perf_counter() - t0) * 1000),
                        }
                    )
            except Exception as e:
                await queue.put({"model": model_name, "delta": f"[error] {e}", "latency_ms": 0})

        async def fan_in(model_list: List[str]):
            tasks = [asyncio.create_task(run_one(m)) for m in model_list]
            try:
                await asyncio.gather(*tasks)
            finally:
                done.set()

        asyncio.create_task(fan_in(models))

        while True:
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=0.05)
                yield evt
            except asyncio.TimeoutError:
                if done.is_set() and queue.empty():
                    break
                continue


    # mirror your non-streaming merge logic here
    def _build_provider_params(self, model: str, req: EnhancedChatRequest) -> Dict[str, Any]:
        """
        Merge per-model overrides with group params (anthropic/openai/gemini/ollama)
        exactly like your existing non-streaming path.
        """
        merged: Dict[str, Any] = {}

        # Per-model overrides (temperature/max/min are passed separately, so ignore them here)
        if getattr(req, "model_params", None) and model in req.model_params:
            for k, v in (req.model_params[model] or {}).items():
                if v is not None and k not in ("temperature", "max_tokens", "min_tokens"):
                    merged[k] = v

        # Group params: copy only if present
        if req.anthropic_params:
            merged.update({k: v for k, v in req.anthropic_params.items() if v is not None})
        if req.openai_params:
            merged.update({"openai_params": {k: v for k, v in req.openai_params.items() if v is not None}})
        if req.gemini_params:
            merged.update({k: v for k, v in req.gemini_params.items() if v is not None})
        if req.ollama_params:
            merged.update({k: v for k, v in req.ollama_params.items() if v is not None})

        return merged


# Wire-up function used by FastAPI Depends(...)
def get_enhanced_chat_service(request: Request) -> EnhancedChatService:
    """
    Return the singleton EnhancedChatService that was created at app startup:
      app.state.registry        = ModelRegistry(...)
      app.state.chat_adapter    = EnhancedChatAdapter()
      app.state.services["chat"]= EnhancedChatService(registry, chat_adapter)
    """
    try:
        return request.app.state.services["chat"]
    except Exception as e:
        # Helpful error if startup didn't set it up
        raise RuntimeError(
            "EnhancedChatService singleton not initialized. "
            "Ensure main.py sets app.state.registry, app.state.chat_adapter, "
            "and app.state.services['chat'] in an @app.on_event('startup') hook."
        ) from e


# --- Backward compatibility alias (safe removal after full migration) ---
class ChatService(EnhancedChatService):
    pass