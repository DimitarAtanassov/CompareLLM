# app/core/model_factory.py
from __future__ import annotations
import os
from typing import Any, Dict, Tuple

# LangChain chat integrations
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_cohere import ChatCohere
from langchain_cerebras import ChatCerebras
# Optional native DeepSeek; if you prefer OpenAI-wire, omit this
try:
    from langchain_deepseek import ChatDeepSeek  # type: ignore
except Exception:
    ChatDeepSeek = None  # pragma: no cover

def _env_val(name: str | None) -> str | None:
    return os.getenv(name) if name else None

def build_chat_model(provider_key: str, provider_cfg: Dict[str, Any], model_name: str) -> Any:
    """
    Given a provider config & model name, return an initialized LangChain ChatModel.
    """
    ptype = provider_cfg.get("type")          # e.g. openai, anthropic, gemini, ollama, cohere, deepseek, cerebras
    wire  = provider_cfg.get("wire")          # e.g. "openai" for OpenAI-compatible providers
    base_url = provider_cfg.get("base_url")
    headers  = provider_cfg.get("headers") or {}
    api_key  = _env_val(provider_cfg.get("api_key_env"))

    if ptype == "openai":
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, default_headers=headers)

    elif ptype == "anthropic":
        return ChatAnthropic(model=model_name, api_key=api_key, default_headers=headers)

    elif ptype == "gemini":
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key)

    elif ptype == "ollama":
        return ChatOllama(model=model_name, base_url=base_url)

    elif ptype == "cohere":
        return ChatCohere(model=model_name, api_key=api_key)

    elif ptype == "deepseek" and ChatDeepSeek is not None:
        return ChatDeepSeek(model=model_name, api_key=api_key)

    elif ptype == "cerebras":
        # Cerebras LC wrapper supports OpenAI-ish params
        return ChatCerebras(model=model_name, api_key=api_key, base_url=base_url)

    # OpenAI-wire compatible fallback (e.g., deepseek/cerebras exposed via openai wire)
    elif wire == "openai":
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, default_headers=headers)

    raise ValueError(f"Unsupported provider or wire for '{provider_key}' (type={ptype}, wire={wire})")

# ---------- parameter normalization ----------
def normalize_chat_params(provider_type: str | None, params: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Map your unified UI params -> provider-specific kwargs for LangChain chat models.
    Supported unified keys: temperature, max_tokens, top_p, top_k, frequency_penalty, presence_penalty, stop
    """
    if not params:
        return {}
    pt = (provider_type or "").lower()
    out: Dict[str, Any] = {}

    # temperature (common)
    if "temperature" in params and params["temperature"] is not None:
        out["temperature"] = params["temperature"]

    # max tokens
    mt = params.get("max_tokens")
    if mt is not None:
        if pt == "gemini":
            out["max_output_tokens"] = mt
        elif pt == "ollama":
            out["num_predict"] = mt
        else:
            out["max_tokens"] = mt  # openai/anthropic/cohere/deepseek/cerebras

    # top_p
    if "top_p" in params and params["top_p"] is not None:
        if pt == "cohere":
            out["p"] = params["top_p"]
        else:
            out["top_p"] = params["top_p"]

    # top_k (gemini/cohere/ollama)
    if "top_k" in params and params["top_k"] is not None:
        if pt in ("gemini", "cohere", "ollama"):
            out["top_k"] = params["top_k"]

    # penalties (OpenAI-ish stacks)
    for k in ("frequency_penalty", "presence_penalty"):
        if k in params and params[k] is not None and pt in ("openai", "deepseek", "cerebras"):
            out[k] = params[k]

    # stop sequences (most wrappers accept)
    if "stop" in params and params["stop"]:
        out["stop"] = params["stop"]

    # timeouts/retries (LC common kwargs)
    for k in ("timeout", "max_retries"):
        if k in params and params[k] is not None:
            out[k] = params[k]

    return out

def parse_wire(wire: str) -> Tuple[str, str]:
    """
    Expect 'provider_key:model_name'. If missing, raise for clarity.
    """
    if ":" not in wire:
        raise ValueError(f"Wire must be 'provider:model', got '{wire}'")
    pkey, model = wire.split(":", 1)
    return pkey, model

def resolve_and_bind_from_registry(registry, wire: str, params: Dict[str, Any] | None):
    """
    Fetch a prebuilt model from the registry and bind normalized generation kwargs.
    """
    pkey, model_name = parse_wire(wire)
    base = registry.get(pkey, model_name)
    ptype = registry.provider_type(pkey)
    bound_kwargs = normalize_chat_params(ptype, params or {})
    return base.bind(**bound_kwargs)
