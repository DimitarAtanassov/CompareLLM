import time
from typing import List

from config.logging import log_event
from core.exceptions import DatasetNotFoundError, ModelNotFoundError
from models.requests import SearchRequest
from models.responses import SearchResponse
from providers.registry import ModelRegistry
from services.embedding_service import EmbeddingService
from storage.base import StorageBackend
from utils.similarity import find_similar_documents


class SearchService:
    """Service for semantic search functionality."""
    
    def __init__(
        self, 
        registry: ModelRegistry, 
        embedding_service: EmbeddingService,
        storage: StorageBackend
    ):
        self.registry = registry
        self.embedding_service = embedding_service
        self.storage = storage
    
    async def semantic_search(self, request: SearchRequest) -> SearchResponse:
        """Perform semantic search against a dataset."""
        if request.embedding_model not in self.registry.embedding_map:
            raise ModelNotFoundError(request.embedding_model)
        
        if not await self.storage.dataset_exists(request.dataset_id):
            raise DatasetNotFoundError(request.dataset_id)
        
        provider, model = self.registry.embedding_map[request.embedding_model]
        
        log_event(
            "search.start",
            dataset_id=request.dataset_id,
            provider=provider.name,
            model=request.embedding_model,
            query=request.query[:100]
        )
        
        start_time = time.perf_counter()
        
        try:
            # Generate query embedding
            embedding_request = type('EmbeddingRequest', (), {
                'texts': [request.query],
                'model': request.embedding_model
            })()
            
            embedding_response = await self.embedding_service.generate_embeddings(
                embedding_request
            )
            query_embedding = embedding_response.embeddings[0]
            
            # Get dataset documents
            documents = await self.storage.get_dataset(request.dataset_id)
            
            # Find similar documents
            similar_docs = find_similar_documents(
                query_embedding, 
                documents, 
                request.top_k or 5
            )
            
            # Remove embeddings from response to reduce payload size
            for doc in similar_docs:
                doc.pop('embedding', None)
            
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event(
                "search.end",
                dataset_id=request.dataset_id,
                provider=provider.name,
                model=request.embedding_model,
                ok=True,
                duration_ms=duration_ms,
                results_count=len(similar_docs)
            )
            
            return SearchResponse(
                query=request.query,
                dataset_id=request.dataset_id,
                embedding_model=request.embedding_model,
                results=similar_docs,
                total_documents=len(documents)
            )
            
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event(
                "search.end",
                dataset_id=request.dataset_id,
                provider=provider.name,
                model=request.embedding_model,
                ok=False,
                duration_ms=duration_ms,
                error=str(e)
            )
            raise