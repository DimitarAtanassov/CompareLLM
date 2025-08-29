# adapters/lc_chat_adapter.py
from __future__ import annotations
import os
from functools import lru_cache
from typing import AsyncIterator, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_cohere import ChatCohere  # NEW

_ANTHROPIC_THINKING_MODELS = {
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
}
def _supports_anthropic_thinking(model: str) -> bool:
    return model in _ANTHROPIC_THINKING_MODELS

def _to_lc_messages(messages: List[Dict[str, str]]):
    out = []
    for m in messages:
        role, content = m.get("role"), m.get("content", "")
        if role == "system":   out.append(SystemMessage(content=content))
        elif role == "assistant": out.append(AIMessage(content=content))
        else:                  out.append(HumanMessage(content=content))
    return out

def _split_params(params: Dict) -> Tuple[Dict, Dict]:
    init_keys = {"temperature", "max_tokens"}  # constructor-level knobs
    init_kwargs = {k: v for k, v in params.items() if k in init_keys and v is not None}
    invoke_kwargs = {k: v for k, v in params.items() if k not in init_keys and v is not None}
    return init_kwargs, invoke_kwargs

def _normalize_wire(w: str) -> str:
    w = (w or "").lower()
    return {"cerebras":"openai", "groq":"openai", "together":"openai"}.get(w, w)

class LCChatAdapter:
    """LangChain-backed adapter with lightweight model caching."""

    # ---- public API (used by your EnhancedChatAdapter) ----
    async def stream_chat(
        self,
        *,
        provider_wire: str,
        model: str,
        messages: List[Dict[str, str]],
        provider_params: Optional[Dict] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AsyncIterator[str]:
        wire = _normalize_wire(provider_wire)
        provider_params = provider_params or {}
        lc = self._build_lc_model(wire, model, provider_params, base_url, api_key)
        lc_messages = _to_lc_messages(messages)
        _, invoke_kwargs = _split_params(provider_params)
        if "stop_sequences" in invoke_kwargs and "stop" not in invoke_kwargs:
            invoke_kwargs["stop"] = invoke_kwargs.pop("stop_sequences")

        async for chunk in lc.astream(lc_messages, **invoke_kwargs):
            text = getattr(chunk, "content", None)
            if text:
                yield text

    async def chat_completion(
        self,
        *,
        provider_wire: str,
        model: str,
        messages: List[Dict[str, str]],
        provider_params: Optional[Dict] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        wire = _normalize_wire(provider_wire)
        provider_params = provider_params or {}
        lc = self._build_lc_model(wire, model, provider_params, base_url, api_key)
        lc_messages = _to_lc_messages(messages)
        _, invoke_kwargs = _split_params(provider_params)
        if "stop_sequences" in invoke_kwargs and "stop" not in invoke_kwargs:
            invoke_kwargs["stop"] = invoke_kwargs.pop("stop_sequences")
        result = await lc.ainvoke(lc_messages, **invoke_kwargs)
        return getattr(result, "content", "") or ""

    # ---- model factory + cache (cache only the constructor-level args) ----
    @lru_cache(maxsize=128)
    def _cached_model(
        self, wire: str, model: str, base_url: str | None, api_key: str | None, frozen_init: Tuple[Tuple[str, str], ...]
    ) -> Runnable:
        init_kwargs = {k: (float(v) if k=="temperature" else int(v)) for k, v in frozen_init}
        if wire == "anthropic":
            # thinking handled per-call via extra_body; no need here
            return ChatAnthropic(model=model, anthropic_api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), **init_kwargs)
        if wire == "openai":
            return ChatOpenAI(model=model, api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url or os.getenv("OPENAI_BASE_URL"), **init_kwargs)
        if wire == "gemini":
            return ChatGoogleGenerativeAI(model=model, google_api_key=api_key or os.getenv("GOOGLE_API_KEY"), **init_kwargs)
        if wire == "ollama":
            return ChatOllama(model=model, base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), **init_kwargs)
        if wire == "cohere":
            return ChatCohere(model=model, cohere_api_key=api_key or os.getenv("COHERE_API_KEY"), **init_kwargs)
        # default to OpenAI-compat
        return ChatOpenAI(model=model, api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url or os.getenv("OPENAI_BASE_URL"), **init_kwargs)

    def _build_lc_model(
        self,
        wire: str,
        model: str,
        provider_params: Dict,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> Runnable:
        init_kwargs, _ = _split_params(provider_params)

        # Anthropic "thinking" goes via per-call extra_body (invoke side), but
        # LangChain’s ChatAnthropic lets us set it on the client too. When supported:
        if wire == "anthropic" and provider_params.get("thinking_enabled") and _supports_anthropic_thinking(model):
            budget = int(provider_params.get("thinking_budget_tokens") or 2048)
            # NOTE: many LangChain versions accept extra_body at init; if yours doesn’t,
            # pass it at call time (invoke_kwargs) instead and keep the cached model “clean”.
            init_kwargs = dict(init_kwargs)  # shallow copy
            init_kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": budget}}

        frozen_init = tuple(sorted((k, str(v)) for k, v in init_kwargs.items()))
        return self._cached_model(wire, model, base_url, api_key, frozen_init)
