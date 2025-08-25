from typing import List
import httpx

from core.exceptions import ProviderError
from providers.base import Provider


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
        Robust Ollama embeddings:
        1) Try OpenAI-compatible /v1/embeddings (batch).
        2) Fallback to native /api/embeddings (one text per call).
            - Try {"input": "..."} first, then retry with {"prompt": "..."} if needed.
        """
        import os, json, httpx

        headers = dict(provider.headers or {})
        base = (provider.base_url or "http://ollama:11434").rstrip("/")

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            # ---------- Try OpenAI-compatible batch once ----------
            try:
                oai_url = f"{base}/v1/embeddings"
                oai_payload = {"model": model, "input": texts}
                r = await client.post(oai_url, headers=headers, json=oai_payload)
                if r.status_code < 400:
                    data = r.json()
                    if isinstance(data, dict) and "data" in data:
                        vecs = [row.get("embedding") for row in data["data"]]
                        if all(isinstance(v, list) and len(v) > 0 for v in vecs):
                            return vecs
                        # If shapes look wrong, fall through to native
            except Exception:
                # Ignore and fall through to native
                pass

            # ---------- Native per-text with input→prompt fallback ----------
            native_url = f"{base}/api/embeddings"
            out: List[List[float]] = []
            for t in texts:
                # 1) try input
                payload = {"model": model, "input": t}
                r = await client.post(native_url, headers=headers, json=payload)
                ok_vec = None
                if r.status_code < 400:
                    j = r.json() or {}
                    v = j.get("embedding") or (j.get("data", [{}])[0].get("embedding") if isinstance(j.get("data"), list) and j["data"] else None)
                    if isinstance(v, list) and len(v) > 0:
                        ok_vec = v

                # 2) if empty/missing, retry with prompt
                if ok_vec is None:
                    payload2 = {"model": model, "prompt": t}
                    r2 = await client.post(native_url, headers=headers, json=payload2)
                    j2 = r2.json() if r2.status_code < 400 else {}
                    v2 = j2.get("embedding") or (j2.get("data", [{}])[0].get("embedding") if isinstance(j2.get("data"), list) and j2["data"] else None)
                    if not (isinstance(v2, list) and len(v2) > 0):
                        # Helpful error
                        sample = json.dumps((j2 or j), separators=(",", ":"), ensure_ascii=False)
                        if len(sample) > 300:
                            sample = sample[:300] + "…"
                        raise RuntimeError(
                            f"Ollama returned no embedding for model={model}. "
                            f"Tried input and prompt. Last payload preview: {sample}"
                        )
                    ok_vec = v2

                out.append([float(x) for x in ok_vec])

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