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
from providers.registry import ModelRegistry
from providers.adapters.enhanced_chat_adapter import EnhancedChatAdapter

def _dump_params(obj) -> Dict[str, Any]:
    if obj is None:
        return {}
    md = getattr(obj, "model_dump", None)
    if callable(md):
        return md(exclude_none=True)
    d = getattr(obj, "dict", None)
    if callable(d):
        return d(exclude_none=True)
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if v is not None}
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if v is not None}
    return {}

def _get_wire(provider) -> str:
    wire = getattr(provider, "wire", None)
    if wire:
        return str(wire).lower()
    ptype = getattr(provider, "type", None)
    return str(ptype or "").lower()

def _normalize_wire(wire: str) -> str:
    """Map OpenAI-compat providers onto the OpenAI wire for LangChain."""
    w = (wire or "").lower()
    if w in {"cerebras", "groq", "together"}:
        return "openai"
    return w

class EnhancedChatService:
    """Enhanced service for handling chat completions with provider-specific parameters."""

    def __init__(self, registry: ModelRegistry, chat_adapter: EnhancedChatAdapter):
        self.registry = registry
        self.chat_adapter = chat_adapter

    async def chat_completion(self, request: EnhancedChatRequest) -> ChatResponse:
        """Handle chat completion for multiple models with enhanced parameters."""
        messages = request.to_messages()

        chosen_models = request.models or list(self.registry.model_map.keys())
        self._validate_models(chosen_models)

        log_event(
            "enhanced_chat.start",
            models=chosen_models,
            message_count=len(messages),
            has_anthropic_params=request.anthropic_params is not None,
            has_openai_params=request.openai_params is not None,
            has_gemini_params=request.gemini_params is not None,
            has_ollama_params=request.ollama_params is not None,
            has_provider_params=request.provider_params is not None,
        )

        tasks = [self._process_single_model(model_name, messages, request) for model_name in chosen_models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        answers: Dict[str, ModelAnswer] = {}
        for model, result in zip(chosen_models, results):
            if isinstance(result, Exception):
                log_event("model.error", model=model, error=str(result))
                answers[model] = ModelAnswer(error=str(result))
            else:
                answers[model] = ModelAnswer(answer=result)

        display_prompt = self._extract_display_prompt(messages)

        log_event(
            "enhanced_chat.complete",
            models=chosen_models,
            success_count=sum(1 for ans in answers.values() if not ans.error),
            error_count=sum(1 for ans in answers.values() if ans.error),
        )

        return ChatResponse(prompt=display_prompt, models=chosen_models, answers=answers)

    def _extract_display_prompt(self, messages: List[Dict[str, str]]) -> str:
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        return user_messages[-1] if user_messages else "No user message found"

    # (kept for compatibility if you use it elsewhere — not used for branching anymore)
    @staticmethod
    def is_openai_stream_model(m: str) -> bool:
        ml = m.lower()
        return bool(re.match(r"^(gpt-|o\d+)", ml))

    async def _process_single_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        request: EnhancedChatRequest,
    ) -> str:
        provider, model = self.registry.model_map[model_name]

        # Effective gen params (global → per-model)
        mp = (request.model_params or {}).get(model_name, {}) or {}
        temperature: Optional[float] = mp.get("temperature", request.temperature)
        max_tokens: Optional[int] = mp.get("max_tokens", request.max_tokens)
        min_tokens: Optional[int] = mp.get("min_tokens", request.min_tokens)

        # Provider wire + normalization for OpenAI-compat
        wire = _normalize_wire(_get_wire(provider))

        # Provider-specific params bundle and fold gen knobs into it
        provider_params: Dict[str, Any] = self._build_provider_params(model_name, request)
        if temperature is not None:
            provider_params["temperature"] = float(temperature)
        if max_tokens is not None:
            provider_params["max_tokens"] = int(max_tokens)
        if min_tokens is not None:
            # Anthropic uses min_output_tokens; harmless passthrough for others
            provider_params["min_output_tokens"] = int(min_tokens)

        start_time = time.perf_counter()
        try:
            # Non-streaming path always uses LangChain ainvoke via adapter
            result = await self.chat_adapter.chat_completion(
                provider_wire=wire,
                model=model,
                messages=messages,
                provider_params=provider_params,
                base_url=getattr(provider, "base_url", None),
                api_key=getattr(provider, "api_key", None),
            )

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event(
                "enhanced_chat.model_end",
                provider=provider.name,
                model=model_name,
                ok=True,
                duration_ms=duration_ms,
                answer_chars=len(result or ""),
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
            )
            raise ProviderError(provider.name, str(e))

    def _validate_models(self, models: List[str]) -> None:
        unknown_models = [m for m in models if m not in self.registry.model_map]
        if unknown_models:
            raise ModelNotFoundError(", ".join(unknown_models))

    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.registry.model_map:
            raise ModelNotFoundError(model_name)

        provider, model = self.registry.model_map[model_name]

        capabilities = {
            "model_name": model_name,
            "provider_name": provider.name,
            "provider_type": provider.type,
            "base_url": provider.base_url,
            "supports_streaming": True,
            "supports_system_messages": True,
        }

        if provider.type == "anthropic":
            from providers.anthropic_params import get_model_config
            model_config = get_model_config(model_name)
            if model_config:
                capabilities.update(
                    {
                        "supports_thinking": model_config.supports_thinking,
                        "supports_tools": model_config.supports_tools,
                        "max_context_tokens": model_config.max_context_tokens,
                        "default_rpm": model_config.default_rpm,
                        "default_tpm": model_config.default_tpm,
                    }
                )
            else:
                capabilities.update(
                    {
                        "supports_thinking": True,
                        "supports_tools": True,
                        "max_context_tokens": 200000,
                    }
                )
        elif provider.type == "openai":
            capabilities.update(
                {
                    "supports_tools": True,
                    "supports_function_calling": True,
                    "supports_json_mode": True,
                    "max_context_tokens": 128000,
                }
            )
        elif provider.type == "gemini":
            capabilities.update(
                {
                    "supports_tools": True,
                    "supports_safety_settings": True,
                    "supports_multimodal": True,
                    "max_context_tokens": 1000000,
                }
            )
        elif provider.type == "ollama":
            capabilities.update(
                {
                    "supports_local_deployment": True,
                    "supports_custom_models": True,
                    "supports_mirostat": True,
                }
            )

        return capabilities

    async def validate_request_for_model(self, model_name: str, request: EnhancedChatRequest) -> Dict[str, Any]:
        capabilities = self.get_model_capabilities(model_name)
        validation_result = {"valid": True, "warnings": [], "errors": []}

        provider_wire = _normalize_wire(_get_wire(self.registry.model_map[model_name][0]))

        mp: Dict[str, Any] = (request.model_params or {}).get(model_name, {}) or {}
        effective_max_tokens: Optional[int] = mp.get("max_tokens", request.max_tokens)

        if provider_wire == "anthropic" and request.anthropic_params:
            thinking_enabled = getattr(request.anthropic_params, "thinking_enabled", None)
            thinking_budget = getattr(request.anthropic_params, "thinking_budget_tokens", None)

            if thinking_enabled and not capabilities.get("supports_thinking", False):
                validation_result["warnings"].append(f"Model {model_name} may not support extended thinking")

            if thinking_budget is not None and effective_max_tokens is not None:
                if thinking_budget > effective_max_tokens:
                    validation_result["errors"].append("Thinking budget tokens cannot exceed max_tokens")
                    validation_result["valid"] = False

        max_context = capabilities.get("max_context_tokens")
        if max_context and (effective_max_tokens is not None) and (effective_max_tokens > max_context):
            validation_result["warnings"].append(
                f"Requested max_tokens ({effective_max_tokens}) exceeds model context limit ({max_context})"
            )

        if provider_wire == "openai" and getattr(request, "openai_params", None):
            openai_params_dict = vars(request.openai_params) if hasattr(request.openai_params, "__dict__") else request.openai_params
            self._validate_openai_params(openai_params_dict or {}, validation_result)

        elif provider_wire == "anthropic" and getattr(request, "anthropic_params", None):
            anth_params = (
                request.anthropic_params.model_dump(exclude_none=True)
                if hasattr(request.anthropic_params, "model_dump")
                else (vars(request.anthropic_params) if hasattr(request.anthropic_params, "__dict__") else request.anthropic_params)
            )
            self._validate_anthropic_params(anth_params or {}, validation_result)

        return validation_result

    def _validate_anthropic_params(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
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
        messages = req.to_messages() if hasattr(req, "to_messages") else (req.messages or [])
        models = req.models or list(self.registry.model_map.keys())

        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        done = asyncio.Event()

        async def run_one(model_name: str):
            try:
                provider, actual_model = self.registry.model_map[model_name]
            except KeyError:
                await queue.put({"model": model_name, "delta": f"[error] Unknown model {model_name}", "latency_ms": 0})
                return

            # Effective gen params (global → per-model)
            mp: Dict[str, Any] = (getattr(req, "model_params", {}) or {}).get(model_name, {}) or {}
            temperature: Optional[float] = mp.get("temperature", getattr(req, "temperature", None))
            max_tokens: Optional[int] = mp.get("max_tokens", getattr(req, "max_tokens", None))
            min_tokens: Optional[int] = mp.get("min_tokens", getattr(req, "min_tokens", None))

            # Bundle provider params + fold gen knobs in
            params = self._build_provider_params(model_name, req)
            if temperature is not None:
                params["temperature"] = float(temperature)
            if max_tokens is not None:
                params["max_tokens"] = int(max_tokens)
            if min_tokens is not None:
                params["min_output_tokens"] = int(min_tokens)

            wire = _normalize_wire(_get_wire(provider))
            t0 = time.perf_counter()
            try:
                async for delta in self.chat_adapter.stream_chat(
                    provider_wire=wire,
                    model=actual_model,
                    messages=messages,
                    provider_params=params,
                    base_url=getattr(provider, "base_url", None),
                    api_key=getattr(provider, "api_key", None),
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

    def _build_provider_params(self, model: str, req: EnhancedChatRequest) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}

        if getattr(req, "model_params", None) and model in req.model_params:
            for k, v in (req.model_params[model] or {}).items():
                if v is not None and k not in ("temperature", "max_tokens", "min_tokens"):
                    merged[k] = v

        anth = _dump_params(getattr(req, "anthropic_params", None))
        oai = _dump_params(getattr(req, "openai_params", None))
        gem = _dump_params(getattr(req, "gemini_params", None))
        oll = _dump_params(getattr(req, "ollama_params", None))

        if anth:
            merged.update(anth)
        if gem:
            merged.update(gem)
        if oll:
            merged.update(oll)
        if oai:
            merged["openai_params"] = oai

        return merged

# Wire-up for FastAPI Depends(...)
def get_enhanced_chat_service(request: Request) -> EnhancedChatService:
    try:
        return request.app.state.services["chat"]
    except Exception as e:
        raise RuntimeError(
            "EnhancedChatService singleton not initialized. "
            "Ensure main.py sets app.state.registry, app.state.chat_adapter, "
            "and app.state.services['chat'] in the FastAPI lifespan."
        ) from e

class ChatService(EnhancedChatService):
    pass
