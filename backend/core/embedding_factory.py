# core/embedding_factory.py
from __future__ import annotations
import os
from typing import Any, Dict, Optional

# ---- Embedding integrations ----
# OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

def _env_val(name: Optional[str]) -> Optional[str]:
    return os.getenv(name) if name else None


def _log(msg: str) -> None:
    print(f"[EmbeddingFactory] {msg}")

def build_embedding_model(
    provider_key: str,
    provider_cfg: Dict[str, Any],
    model_name: str,
) -> Any:
    """
    Initialize a LangChain Embeddings instance for the given provider+model.

    Supported ptypes (from models.yaml):
      - openai
      - gemini
      - cohere
      - voyage
      - ollama
    Also: providers that declare wire: "openai" for OpenAI-compatible embeddings.

    Raises ValueError for unsupported combos so the caller can log/skip.
    """
    ptype: str = provider_cfg.get("type") or ""
    wire: Optional[str] = provider_cfg.get("wire")
    base_url: Optional[str] = provider_cfg.get("base_url")
    api_key_env: Optional[str] = provider_cfg.get("api_key_env")
    api_key: Optional[str] = _env_val(api_key_env)

    _log(f"Build embedding -> provider='{provider_key}', type='{ptype}', model='{model_name}'")

    # ---- OpenAI ----
    if ptype == "openai" or wire == "openai":
        # OpenAIEmbeddings supports base_url for compatible endpoints (incl. Azure, Cerebras-as-OpenAI, etc.)
        # If your deployment needs extra headers/params, pass via env or add here.
        return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url)

    # ---- Google Gemini ----
    if ptype == "gemini":
        if GoogleGenerativeAIEmbeddings is None:
            raise RuntimeError("langchain-google-genai not installed")
        # Typical models: 'text-embedding-004', 'embedding-001'
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    # ---- Cohere ----
    if ptype == "cohere":
        if CohereEmbeddings is None:
            raise RuntimeError("langchain-cohere not installed")
        # Models: embed-english-v3.0, embed-multilingual-v3.0, *-light-v3.0
        return CohereEmbeddings(model=model_name, cohere_api_key=api_key)

    # ---- Voyage AI ----
    if ptype == "voyage":
        if VoyageAIEmbeddings is None:
            raise RuntimeError("langchain-voyageai not installed")
        # Models: voyage-large-2, voyage-code-2, voyage-2, voyage-lite-02-instruct
        return VoyageAIEmbeddings(model=model_name, voyage_api_key=api_key)

    # ---- Ollama (local) ----
    if ptype == "ollama":
        if OllamaEmbeddings is None:
            raise RuntimeError("OllamaEmbeddings not available (install langchain-ollama or langchain-community)")
        # NOTE: ensure the embedding model is present in Ollama (e.g., 'nomic-embed-text', 'all-minilm')
        # base_url like http://localhost:11434 from your YAML.
        return OllamaEmbeddings(model=model_name, base_url=base_url)

    # ---- Providers without embeddings in your YAML ----
    # DeepSeek & Cerebras currently have embedding_models: [] in models.yaml.
    # If you add them later via a compatible wire (e.g., openai), they’ll be covered above.
    raise ValueError(f"Unsupported embedding provider '{provider_key}' (type={ptype}, wire={wire})")
