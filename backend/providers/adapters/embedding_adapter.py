import asyncio
from typing import List
import httpx

from core.exceptions import ProviderError
from providers.base import Provider
from providers.adapters.chat_adapter import get_host_lock


class EmbeddingAdapter:
    """Adapter for embedding generation across different provider types."""
    
    async def generate_embeddings(
        self,
        provider: Provider,
        model: str,
        texts: List[str],
        timeout_s: int = 180,
    ) -> List[List[float]]:
        """Generate embeddings using the appropriate provider."""
        try:
            if provider.type == "openai":
                return await self._openai_embeddings(provider, model, texts, timeout_s)
            elif provider.type == "gemini":
                return await self._gemini_embeddings(provider, model, texts, timeout_s)
            elif provider.type == "cohere":
                return await self._cohere_embeddings(provider, model, texts, timeout_s)
            elif provider.type == "voyage":
                return await self._voyage_embeddings(provider, model, texts, timeout_s)
            elif provider.type == "ollama":   # 🔥 NEW
                return await self._ollama_embeddings(provider, model, texts, timeout_s)
            else:
                raise ProviderError(provider.name, f"Unsupported provider type: {provider.type}")
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(provider.name, str(e))
    
    async def _openai_embeddings(
        self, provider: Provider, model: str, texts: List[str], timeout_s: int
    ) -> List[List[float]]:
        """Handle OpenAI-compatible embeddings."""
        headers = dict(provider.headers or {})
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        
        url = f"{provider.base_url}/embeddings"
        payload = {"model": model, "input": texts}
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            self._raise_for_status(r)
            data = r.json()
            return [item["embedding"] for item in data["data"]]
    
    async def _gemini_embeddings(
        self, provider: Provider, model: str, texts: List[str], timeout_s: int
    ) -> List[List[float]]:
        """Handle Gemini embeddings."""
        params = {"key": provider.api_key} if provider.api_key else {}
        headers = dict(provider.headers or {})
        
        url = f"{provider.base_url}/v1beta/models/{model}:embedContent"
        
        all_embeddings = []
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            for text in texts:
                payload = {"content": {"parts": [{"text": text}]}}
                r = await client.post(url, params=params, headers=headers, json=payload)
                self._raise_for_status(r)
                data = r.json()
                all_embeddings.append(data["embedding"]["values"])
        
        return all_embeddings
    
    async def _cohere_embeddings(
        self, provider: Provider, model: str, texts: List[str], timeout_s: int
    ) -> List[List[float]]:
        """Handle Cohere embeddings."""
        headers = dict(provider.headers or {})
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        
        url = f"{provider.base_url}/v1/embed"
        payload = {
            "model": model,
            "texts": texts,
            "input_type": "search_document"
        }
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            self._raise_for_status(r)
            data = r.json()
            return data["embeddings"]
    
    async def _voyage_embeddings(
        self, provider: Provider, model: str, texts: List[str], timeout_s: int
    ) -> List[List[float]]:
        """Handle Voyage AI embeddings."""
        headers = dict(provider.headers or {})
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        
        url = f"{provider.base_url}/v1/embeddings"
        payload = {
            "model": model,
            "input": texts,
            "input_type": "document"
        }
        
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            self._raise_for_status(r)
            data = r.json()
            return [item["embedding"] for item in data["data"]]

    async def _ollama_embeddings(
        self, provider: Provider, model: str, texts: List[str], timeout_s: int
    ) -> List[List[float]]:
        """
        Supports both Ollama endpoints:
          - OpenAI compatible:  POST {base}/v1/embeddings  -> {"object":"list","data":[{"embedding":[...]}]}
          - Native Ollama:      POST {base}/api/embeddings -> {"embedding":[...]} (one text per call)
        We auto-select the URL based on base_url; if it contains /v1 we use the OAI-compatible batch call.
        """
        headers = dict(provider.headers or {})
        base = provider.base_url.rstrip("/")

        use_oai = base.endswith("/v1") or "/v1" in base  # e.g., http://ollama:11434/v1
        url = f"{base}/embeddings" if use_oai else f"{base}/api/embeddings"

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            if use_oai:
                # Batch once
                payload = {"model": model, "input": texts}
                r = await client.post(url, headers=headers, json=payload)
                self._raise_for_status(r)
                data = r.json()

                # Accept both OpenAI-compatible ("data") and native just in case
                if isinstance(data, dict) and "data" in data:
                    return [item["embedding"] for item in data["data"]]
                if isinstance(data, dict) and "embedding" in data:
                    # Some proxies might still return single-object
                    return [data["embedding"]]

                raise RuntimeError(f"Ollama returned unexpected response: {data}")

            else:
                # Native API: one request per text
                out: List[List[float]] = []
                for t in texts:
                    payload = {"model": model, "input": t}
                    r = await client.post(url, headers=headers, json=payload)
                    self._raise_for_status(r)
                    data = r.json()
                    if isinstance(data, dict) and "embedding" in data:
                        out.append(data["embedding"])
                    elif isinstance(data, dict) and "data" in data:
                        # Defensive: some builds expose OAI format even on /api
                        out.append(data["data"][0]["embedding"])
                    else:
                        raise RuntimeError(f"Ollama returned unexpected response: {data}")
                return out


    
    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise appropriate exception for HTTP errors."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                j = response.json()
                detail = j.get("error", j.get("message", "")) or str(j)
            except Exception:
                pass
            raise RuntimeError(f"{response.status_code} {response.reason_phrase}: {detail}") from e