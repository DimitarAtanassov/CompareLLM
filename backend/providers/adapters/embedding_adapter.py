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