from __future__ import annotations
import json
import os, asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import httpx
import yaml
import numpy as np

@dataclass
class Provider:
    name: str
    type: str                 
    base_url: str
    api_key_env: Optional[str]
    headers: Dict[str, str]
    models: List[str]
    embedding_models: List[str] = None  # New field for embedding models

    def __post_init__(self):
        if self.embedding_models is None:
            self.embedding_models = []

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env) if self.api_key_env else None

class ModelRegistry:
    def __init__(self, cfg: Dict[str, Any]):
        self.providers: Dict[str, Provider] = {}
        for pname, p in cfg.get("providers", {}).items():
            self.providers[pname] = Provider(
                name=pname,
                type=p["type"],
                base_url=p["base_url"].rstrip("/"),
                api_key_env=(p.get("api_key_env") or None),
                headers=(p.get("headers") or {}),
                models=(p.get("models") or []),
                embedding_models=(p.get("embedding_models") or []),
            )
        # map model -> (provider, model_name)
        self.model_map: Dict[str, Tuple[Provider, str]] = {}
        for prov in self.providers.values():
            for m in prov.models:
                self.model_map[m] = (prov, m)
        
        # map embedding_model -> (provider, model_name)
        self.embedding_map: Dict[str, Tuple[Provider, str]] = {}
        for prov in self.providers.values():
            for m in prov.embedding_models:
                self.embedding_map[m] = (prov, m)

    @classmethod
    def from_path(cls, path: str) -> "ModelRegistry":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(cfg)

# ------------ per-host serialization (helps on Mac CPU) ------------
_HOST_LOCKS: Dict[str, asyncio.Semaphore] = {}
def host_lock_key(provider: Provider) -> str:
    return provider.base_url
def host_lock(provider: Provider) -> asyncio.Semaphore:
    k = host_lock_key(provider)
    if k not in _HOST_LOCKS:
        _HOST_LOCKS[k] = asyncio.Semaphore(1)
    return _HOST_LOCKS[k]

# -------------------------- Embedding adapters --------------------------

async def embedding_call(
    provider: Provider,
    model: str,
    texts: List[str],
    timeout_s: int = 180,
) -> List[List[float]]:
    """
    Unified embedding call for: OpenAI-compatible providers, Gemini, Cohere, and Voyage.
    """
    headers = dict(provider.headers or {})

    if provider.type == "openai":
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        url = f"{provider.base_url}/embeddings"
        
        payload = {
            "model": model,
            "input": texts,
        }
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            _raise_nice(r)
            data = r.json()
            return [item["embedding"] for item in data["data"]]

    elif provider.type == "gemini":
        # Gemini embedding API: models/*:embedContent
        if provider.api_key:
            params = {"key": provider.api_key}
        else:
            params = {}
        
        url = f"{provider.base_url}/v1beta/models/{model}:embedContent"
        
        # Gemini expects single text input, so we'll batch them
        all_embeddings = []
        for text in texts:
            payload = {
                "content": {
                    "parts": [{"text": text}]
                }
            }
            
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                r = await client.post(url, params=params, headers=headers, json=payload)
                _raise_nice(r)
                data = r.json()
                all_embeddings.append(data["embedding"]["values"])
        
        return all_embeddings

    elif provider.type == "cohere":
        # Cohere embeddings API
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        
        url = f"{provider.base_url}/v1/embed"
        
        payload = {
            "model": model,
            "texts": texts,
            "input_type": "search_document"  # Can be adjusted based on use case
        }
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            _raise_nice(r)
            data = r.json()
            return data["embeddings"]

    elif provider.type == "voyage":
        # Voyage AI embeddings API
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        
        url = f"{provider.base_url}/v1/embeddings"
        
        payload = {
            "model": model,
            "input": texts,
            "input_type": "document"  # Can be adjusted based on use case
        }
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            _raise_nice(r)
            data = r.json()
            return [item["embedding"] for item in data["data"]]

    else:
        raise ValueError(f"Unsupported provider type for embeddings: {provider.type}")

# -------------------------- Chat adapters --------------------------

async def chat_call(
    provider: Provider,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    min_tokens: Optional[int] = None,    # <— added
    timeout_s: int = 180,
) -> str:
    """
    Unified chat call for: OpenAI-compatible providers + Gemini.
    """
    headers = dict(provider.headers or {})

    if provider.type == "openai":
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        url = f"{provider.base_url}/chat/completions"

        def build_payload(use_completion_tokens: bool, include_temperature: bool):
            body = {
                "model": model,
                "messages": messages,
            }
            if include_temperature:
                body["temperature"] = temperature
            if use_completion_tokens:
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens
            # no min_tokens equivalent here
            return body
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            use_completion = False           # start with max_tokens
            include_temp   = True            # start including temperature

            # attempt 1
            r = await client.post(
                url, headers=headers, json=build_payload(use_completion, include_temp)
            )

            # If bad request, inspect and adapt up to two times
            for _ in range(2):
                if r.status_code != 400:
                    break
                try:
                    err = r.json()
                    # OpenAI errors may be {"error": {...}} OR flat dict
                    meta = err.get("error", err)
                    msg = str(meta.get("message", meta))
                except Exception:
                    msg = ""

                need_completion = ("max_tokens" in msg and "max_completion_tokens" in msg)
                temp_unsupported = ("temperature" in msg and "supported" in msg)

                if not (need_completion or temp_unsupported):
                    break  # unknown 400—let _raise_nice handle it

                # adjust knobs based on the message
                if need_completion:
                    use_completion = True
                if temp_unsupported:
                    include_temp = False

                # retry with adjusted payload
                r = await client.post(
                    url, headers=headers, json=build_payload(use_completion, include_temp)
                )

            _raise_nice(r)
            data = r.json()
            return data["choices"][0]["message"]["content"]

    elif provider.type == "gemini":
        # Gemini: models/*:generateContent
        # Convert OpenAI-style messages -> Gemini contents.
        # Prefer using systemInstruction if present.
        sys_texts = [m["content"] for m in messages if m["role"] == "system"]
        user_assistant_msgs = [m for m in messages if m["role"] in ("user", "assistant")]

        contents = []
        for m in user_assistant_msgs:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        params = {"key": provider.api_key} if provider.api_key else {}
        url = f"{provider.base_url}/v1beta/models/{model}:generateContent"
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if sys_texts:
            payload["systemInstruction"] = {"role": "user", "parts": [{"text": "\n".join(sys_texts)}]}

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, params=params, headers=headers, json=payload)
            _raise_nice(r)
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    elif provider.type == "anthropic":
        # Anthropic Messages API: POST /v1/messages
        # Required headers:
        #   - x-api-key: <ANTHROPIC_API_KEY>
        #   - anthropic-version: e.g. "2023-06-01"
        if provider.api_key:
            headers["x-api-key"] = provider.api_key
        # Allow overriding via config headers; otherwise set a sane default.
        headers.setdefault("anthropic-version", "2023-06-01")

        url = f"{provider.base_url.rstrip('/')}/v1/messages"

        # Extract system message (concat if multiple; order preserved)
        system_texts = [m["content"] for m in messages if m.get("role") == "system"]
        system_str = "\n".join(system_texts) if system_texts else None

        # Translate the remaining turns
        turns = []
        for m in messages:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            # Anthropic expects content as array of parts; simplest is one text part
            turns.append({
                "role": "user" if role == "user" else "assistant",
                "content": [{"type": "text", "text": m.get("content", "")}],
            })

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": turns,
        }
        if system_str:
            payload["system"] = system_str
        if min_tokens is not None:
            payload["min_output_tokens"] = int(min_tokens)

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            _raise_nice(r)
            data = r.json()

            # data.content is a list of blocks; collect any text blocks
            parts = data.get("content", []) or []
            out: List[str] = []
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "text" and "text" in p:
                    out.append(p["text"])
            return "".join(out) if out else json.dumps(data)

    else:
        raise ValueError(f"Unsupported provider type: {provider.type}")

# -------------------------- Similarity functions --------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    
    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))

def find_similar_documents(
    query_embedding: List[float],
    document_embeddings: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Find the most similar documents to a query embedding.
    
    Args:
        query_embedding: The embedding vector of the query
        document_embeddings: List of dicts with 'embedding' and other metadata
        top_k: Number of top results to return
    
    Returns:
        List of documents sorted by similarity score (highest first)
    """
    similarities = []
    
    for doc in document_embeddings:
        if 'embedding' not in doc:
            continue
            
        similarity = cosine_similarity(query_embedding, doc['embedding'])
        doc_copy = doc.copy()
        doc_copy['similarity_score'] = similarity
        similarities.append(doc_copy)
    
    # Sort by similarity score (descending) and return top_k
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similarities[:top_k]


def _raise_nice(r: httpx.Response) -> None:
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            j = r.json()
            detail = j.get("error", j.get("message", "")) or str(j)
        except Exception:
            pass
        raise RuntimeError(f"{r.status_code} {r.reason_phrase}: {detail}") from e