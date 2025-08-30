# core/model_factory.py
from __future__ import annotations
import os
from typing import Any, Dict

# LangChain chat integrations
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_cohere import ChatCohere
from langchain_deepseek import ChatDeepSeek  # optional, if you want native DeepSeek

# Notes on providers:
# - OpenAI-compatible (wire="openai") providers (e.g., deepseek, cerebras) can use ChatOpenAI with base_url.
# - Gemini (Google) typically ignores base_url and uses api_key.
# - Ollama requires base_url=http://host:11434 and a local model.
# - Cohere uses api_key + model.
# - Anthropic uses api_key + model.

def _env_val(name: str | None) -> str | None:
    return os.getenv(name) if name else None

def build_chat_model(provider_key: str, provider_cfg: Dict[str, Any], model_name: str) -> Any:
    """
    Given a single provider config & a model name, return an initialized LangChain ChatModel.
    """
    ptype = provider_cfg.get("type")          # e.g. "openai", "anthropic", "gemini", "ollama", "cohere", "deepseek", "cerebras"
    wire = provider_cfg.get("wire")          # e.g. "openai" if OpenAI-compatible API
    base_url = provider_cfg.get("base_url")
    headers = provider_cfg.get("headers") or {}
    api_key = _env_val(provider_cfg.get("api_key_env"))

    # ---- OpenAI-native ----
    if ptype == "openai":
        # ChatOpenAI accepts base_url (new LC versions) and api_key directly.
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, default_headers=headers)

    # ---- Anthropic ----
    if ptype == "anthropic":
        return ChatAnthropic(model=model_name, api_key=api_key, default_headers=headers)

    # ---- Gemini ----
    if ptype == "gemini":
        # ChatGoogleGenerativeAI(api_key=...) is expected; base_url is not used.
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key)

    # ---- Ollama (local) ----
    if ptype == "ollama":
        # base_url is needed (e.g., http://localhost:11434); no api_key required
        return ChatOllama(model=model_name, base_url=base_url)

    # ---- Cohere ----
    if ptype == "cohere":
        return ChatCohere(model=model_name, api_key=api_key)

    # ---- DeepSeek ----
    # Option A: DeepSeek native LC integration (langchain-deepseek)
    if ptype == "deepseek" and wire is None:
        return ChatDeepSeek(model=model_name, api_key=api_key)

    # Option B: Providers that are OpenAI-compatible ("wire": "openai"), e.g. deepseek, cerebras, others
    if wire == "openai":
        # Use ChatOpenAI with a custom base_url & key
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, default_headers=headers)

    # If we reach here, we don't know how to initialize this provider.
    raise ValueError(f"Unsupported provider or wire for '{provider_key}' (type={ptype}, wire={wire})")
