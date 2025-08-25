import time
from typing import List

from config.logging import log_event
from core.exceptions import ModelNotFoundError, ProviderError
from models.requests import EmbeddingRequest
from models.responses import EmbeddingResponse, EmbeddingUsage  # ADD THIS IMPORT
from providers.registry import ModelRegistry
from providers.adapters.embedding_adapter import EmbeddingAdapter


class EmbeddingService:
    """Service for handling embedding generation."""
    
    def __init__(self, registry: ModelRegistry, embedding_adapter: EmbeddingAdapter):
        self.registry = registry
        self.embedding_adapter = embedding_adapter
    
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for the given texts."""
        if request.model not in self.registry.embedding_map:
            raise ModelNotFoundError(request.model)
        
        provider, model = self.registry.embedding_map[request.model]
        
        log_event(
            "embedding.start",
            provider=provider.name,
            model=request.model,
            text_count=len(request.texts)
        )
        
        start_time = time.perf_counter()
        
        try:
            embeddings = await self.embedding_adapter.generate_embeddings(
                provider=provider,
                model=model,
                texts=request.texts
            )
            
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event(
                "embedding.end",
                provider=provider.name,
                model=request.model,
                ok=True,
                duration_ms=duration_ms,
                text_count=len(request.texts)
            )
            
            return EmbeddingResponse(
                model=request.model,
                embeddings=embeddings,
                usage=EmbeddingUsage(  # USE THE IMPORTED CLASS
                    prompt_tokens=sum(len(text.split()) for text in request.texts),
                    total_tokens=sum(len(text.split()) for text in request.texts)
                )
            )
            
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event(
                "embedding.end",
                provider=provider.name,
                model=request.model,
                ok=False,
                duration_ms=duration_ms,
                error=str(e)
            )
            raise ProviderError(provider.name, str(e))
        
    async def embed_texts(self, model: str, texts: List[str]) -> List[List[float]]:
        """Shim used by /v2/search/self-dataset-compare."""
        req = EmbeddingRequest(texts=texts, model=model)
        resp = await self.generate_embeddings(req)  # your existing method
        # Defensive checks
        if not resp or not getattr(resp, "embeddings", None):
            raise RuntimeError(f"Provider returned no embeddings for model={model}")
        if not resp.embeddings[0]:
            raise RuntimeError(f"Empty embedding vector from model={model}")
        return resp.embeddings