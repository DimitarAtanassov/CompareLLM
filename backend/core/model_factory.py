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

try:
    from langchain_deepseek import ChatDeepSeek  # type: ignore
except Exception:
    ChatDeepSeek = None  # pragma: no cover


def _env_val(name: str | None) -> str | None:
    return os.getenv(name) if name else None


def build_chat_model(provider_key: str, provider_cfg: Dict[str, Any], model_name: str) -> Any:
    """
    Base model for inventory / pooling (no per-request params).
    """
    ptype = provider_cfg.get("type")
    wire  = provider_cfg.get("wire")
    base_url = provider_cfg.get("base_url")
    headers  = provider_cfg.get("headers") or {}
    api_key  = _env_val(provider_cfg.get("api_key_env"))

    if ptype == "openai":
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, default_headers=headers)
    elif ptype == "anthropic":
        return ChatAnthropic(model=model_name, api_key=api_key, default_headers=headers)
    elif ptype == "google":
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
    elif ptype == "ollama":
        return ChatOllama(model=model_name, base_url=base_url)
    elif ptype == "cohere":
        return ChatCohere(model=model_name, api_key=api_key)
    elif ptype == "deepseek" and ChatDeepSeek is not None:
        return ChatDeepSeek(model=model_name, api_key=api_key)
    elif ptype == "cerebras":
        return ChatCerebras(model=model_name, api_key=api_key, base_url=base_url)
    elif wire == "openai":
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, default_headers=headers)

    raise ValueError(f"Unsupported provider or wire for '{provider_key}' (type={ptype}, wire={wire})")

# ---------- parameter normalization ----------
def normalize_chat_params(provider_type: str | None, params: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Map unified UI params -> provider-specific kwargs for LangChain chat models.

    Supported unified keys: temperature, max_tokens, top_p, top_k,
    frequency_penalty, presence_penalty, stop, timeout, max_retries,
    thinking_budget (Gemini, flat), thinking_budget_tokens (legacy UI)
    """
    if not params:
        return {}
    pt = (provider_type or "").lower()
    out: Dict[str, Any] = {}

    # temperature
    if params.get("temperature") is not None:
        out["temperature"] = params["temperature"]

    # max tokens
    mt = params.get("max_tokens")
    if mt is not None:
        if pt == "google":
            out["max_output_tokens"] = mt
        elif pt == "ollama":
            out["num_predict"] = mt
        else:
            out["max_tokens"] = mt  # openai / anthropic / cohere / deepseek / cerebras

    # top_p
    if params.get("top_p") is not None:
        if pt == "cohere":
            out["p"] = params["top_p"]    # Cohere uses 'p'
        else:
            out["top_p"] = params["top_p"]

    # top_k
    if params.get("top_k") is not None:
        if pt == "cohere":
            out["k"] = params["top_k"]    # Cohere uses 'k'
        elif pt in ("google", "ollama"):
            out["top_k"] = params["top_k"]

    # ✅ GEMINI: flat reasoning budget
    if pt == "google":
        # prefer flat `thinking_budget`, fall back to `thinking_budget_tokens`
        tb = params.get("thinking_budget")
        if tb is None:
            tb = params.get("thinking_budget_tokens")

        if tb is not None:
            try:
                out["thinking_budget"] = int(tb)  # -1 (dynamic), 0 (disable), or positive budget
            except Exception:
                pass

        # optional passthroughs if your UI ever sends these
        if params.get("include_thoughts") is not None:
            out["include_thoughts"] = params["include_thoughts"]
        if params.get("response_modalities") is not None:
            out["response_modalities"] = params["response_modalities"]
        if params.get("safety_settings") is not None:
            out["safety_settings"] = params["safety_settings"]

    # penalties (OpenAI-ish stacks)
    for k in ("frequency_penalty", "presence_penalty"):
        if params.get(k) is not None and pt in ("openai", "deepseek", "cerebras"):
            out[k] = params[k]

    # stop sequences
    if params.get("stop"):
        out["stop"] = params["stop"]

    # timeouts / retries (LC common kwargs)
    for k in ("timeout", "max_retries"):
        if params.get(k) is not None:
            out[k] = params[k]

    return out

def parse_wire(wire: str) -> Tuple[str, str]:
    if ":" not in wire:
        raise ValueError(f"Wire must be 'provider:model', got '{wire}'")
    pkey, model = wire.split(":", 1)
    return pkey, model


# ---------- build with params at initialization ----------
def build_chat_model_with_params(
    provider_key: str,
    provider_cfg: Dict[str, Any],
    model_name: str,
    gen_params: Dict[str, Any] | None = None,
) -> Any:
    """
    Construct a *new* LC chat model instance with generation params applied
    at initialization (preferred over .bind()).
    """
    ptype = provider_cfg.get("type")
    wire  = provider_cfg.get("wire")
    base_url = provider_cfg.get("base_url")
    headers  = provider_cfg.get("headers") or {}
    api_key  = _env_val(provider_cfg.get("api_key_env"))

    norm = normalize_chat_params(ptype, gen_params or {})
    ctor_kwargs: Dict[str, Any] = dict(model=model_name)

    if ptype in ("openai", "cerebras") or (wire == "openai"):
        if base_url:
            ctor_kwargs["base_url"] = base_url
        if headers:
            ctor_kwargs["default_headers"] = headers
        if ptype in ("openai",) or wire == "openai":
            ctor_kwargs["api_key"] = api_key
            cls = ChatOpenAI
        elif ptype == "cerebras":
            ctor_kwargs["api_key"] = api_key
            cls = ChatCerebras
        ctor_kwargs.update(norm)
        model_obj = cls(**ctor_kwargs)

    elif ptype == "anthropic":
        model_obj = ChatAnthropic(model=model_name, api_key=api_key, default_headers=headers, **norm)

    elif ptype == "google":
        model_obj = ChatGoogleGenerativeAI(model=model_name, api_key=api_key, **norm)

    elif ptype == "cohere":
        model_obj = ChatCohere(model=model_name, api_key=api_key, **norm)

    elif ptype == "ollama":
        model_obj = ChatOllama(model=model_name, base_url=base_url, **norm)

    elif ptype == "deepseek" and ChatDeepSeek is not None:
        model_obj = ChatDeepSeek(model=model_name, api_key=api_key, **norm)

    else:
        raise ValueError(f"Unsupported provider or wire for '{provider_key}' (type={ptype}, wire={wire})")

    # Safety: bind any leftover params if ctor didn't consume them (older LC versions)
    leftover = {}
    try:
        defaults = getattr(model_obj, "_default_params", None) or {}
        for k, v in (norm or {}).items():
            if k not in defaults:
                leftover[k] = v
    except Exception:
        for k, v in (norm or {}).items():
            leftover[k] = v

    if leftover:
        model_obj = model_obj.bind(**leftover)

    return model_obj


def resolve_and_bind_from_registry(registry, wire: str, params: Dict[str, Any] | None):
    """
    (Legacy) Fetch pooled base model & bind params. Kept for backwards compat.
    """
    pkey, model_name = parse_wire(wire)
    base = registry.get(pkey, model_name)
    ptype = None
    try:
        ptype = registry.provider_type(pkey)
    except Exception:
        pass
    bound_kwargs = normalize_chat_params(ptype, params or {})
    return base.bind(**bound_kwargs)


# ---------- NEW: resolve + *init* (preferred path) ----------
def resolve_and_init_from_registry(registry, wire: str, params: Dict[str, Any] | None):
    """
    Build a brand-new model instance using the provider config in the registry
    and applying normalized generation params at constructor time.

    Compatible with:
      - core.model_registry.ModelRegistry (provider_cfg / get_providers_cfg)
      - providers.registry.ModelRegistry (providers_config()).
    """
    pkey, model_name = parse_wire(wire)

    pcfg: Dict[str, Any] | None = None

    # Preferred: direct single-provider accessor
    if hasattr(registry, "provider_cfg"):
        try:
            pcfg = registry.provider_cfg(pkey)
        except Exception:
            pcfg = None

    # Fallbacks: whole providers dict accessors
    if not pcfg:
        if hasattr(registry, "get_providers_cfg"):
            try:
                all_cfg = registry.get_providers_cfg()
                pcfg = (all_cfg or {}).get(pkey)
            except Exception:
                pcfg = None
        elif hasattr(registry, "providers_config"):
            try:
                all_cfg = registry.providers_config()
                pcfg = (all_cfg or {}).get(pkey)
            except Exception:
                pcfg = None

    if not pcfg:
        raise ValueError(f"No provider config found for '{pkey}'")

    return build_chat_model_with_params(pkey, pcfg, model_name, params or {})
