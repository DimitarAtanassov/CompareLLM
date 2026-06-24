"""Single source of truth for translating unified params to provider SDK kwargs.

The previous codebase normalized parameters in two different places. Here every
mapping lives in one module, one function per provider transport.
"""

from __future__ import annotations

from typing import Any

from app.domain.models import GenerationParams


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    """Assign ``value`` to ``target[key]`` only when it is not ``None``."""
    if value is not None:
        target[key] = value


def openai_kwargs(params: GenerationParams) -> dict[str, Any]:
    """Kwargs for ``openai`` chat.completions (OpenAI / DeepSeek / Cerebras / Ollama)."""
    out: dict[str, Any] = {}
    _put(out, "temperature", params.temperature)
    _put(out, "max_tokens", params.max_tokens)
    _put(out, "top_p", params.top_p)
    _put(out, "frequency_penalty", params.frequency_penalty)
    _put(out, "presence_penalty", params.presence_penalty)
    _put(out, "seed", params.seed)
    _put(out, "stop", params.stop)
    out.update(params.extra)
    return out


def anthropic_kwargs(params: GenerationParams, default_max_tokens: int) -> dict[str, Any]:
    """Kwargs for ``anthropic`` messages streaming."""
    out: dict[str, Any] = {"max_tokens": params.max_tokens or default_max_tokens}
    _put(out, "top_p", params.top_p)
    _put(out, "top_k", params.top_k)
    _put(out, "stop_sequences", params.stop)

    if params.thinking_budget is not None and params.thinking_budget > 0:
        # Extended thinking requires temperature to be unset (defaults to 1).
        out["thinking"] = {"type": "enabled", "budget_tokens": params.thinking_budget}
    else:
        _put(out, "temperature", params.temperature)

    out.update(params.extra)
    return out


def gemini_config_kwargs(params: GenerationParams) -> dict[str, Any]:
    """Kwargs for ``google.genai`` ``GenerateContentConfig`` (excluding thinking).

    The thinking budget is returned separately because it is a nested config
    object constructed by the adapter.
    """
    out: dict[str, Any] = {}
    _put(out, "temperature", params.temperature)
    _put(out, "top_p", params.top_p)
    _put(out, "top_k", params.top_k)
    _put(out, "max_output_tokens", params.max_tokens)
    _put(out, "stop_sequences", params.stop)
    out.update(params.extra)
    return out


def cohere_kwargs(params: GenerationParams) -> dict[str, Any]:
    """Kwargs for ``cohere`` v2 chat streaming."""
    out: dict[str, Any] = {}
    _put(out, "temperature", params.temperature)
    _put(out, "max_tokens", params.max_tokens)
    _put(out, "p", params.top_p)
    _put(out, "k", params.top_k)
    _put(out, "frequency_penalty", params.frequency_penalty)
    _put(out, "presence_penalty", params.presence_penalty)
    _put(out, "seed", params.seed)
    _put(out, "stop_sequences", params.stop)
    out.update(params.extra)
    return out
