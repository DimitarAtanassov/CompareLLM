import time
from typing import List, Dict, Any

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

    async def search(
        self,
        dataset_id: str,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Vector search against a stored dataset. Returns a list of hit dicts.
        Shape is compatible with the self-dataset-compare route, which
        accepts either:
          - {'similarity_score': float, **doc_fields}
          - or {'score': float, 'doc': {...}}
        We’ll return the first form.
        """
        # Ensure dataset exists
        if not await self.storage.dataset_exists(dataset_id):
            raise DatasetNotFoundError(dataset_id)

        # Load documents (each should contain an 'embedding' field)
        documents = await self.storage.get_dataset(dataset_id)

        # Rank by cosine similarity (uses your existing utility)
        ranked = find_similar_documents(query_vector, documents, top_k or 5)
        # Remove stored embeddings to keep payload small
        for item in ranked:
            item.pop("embedding", None)

        # If your find_similar_documents already returns objects that include
        # 'similarity_score' and the other doc fields, we’re done. If instead it
        # returns {'similarity_score': s, 'doc': {...}}, flatten here:
        flattened: List[Dict[str, Any]] = []
        for hit in ranked:
            if "doc" in hit and isinstance(hit["doc"], dict):
                row = {**hit["doc"]}
                row["similarity_score"] = float(hit.get("similarity_score", hit.get("score", 0.0)))
                flattened.append(row)
            else:
                # already flattened
                flattened.append(hit)

        return flattened
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

    # ---------------- NEW: multi-provider side-by-side search ---------------- #

    async def semantic_search_multi(
        self,
        base_dataset_id: str,
        embedding_models: List[str],
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform semantic search across multiple embedding providers/models using the
        SAME query, and return results bucketed per model.

        Returns:
          {
            "query": "...",
            "results": {
              "<modelA>": { "dataset_id": "...", "items": [...], "total_documents": N } | { "error": "...", "items": [] },
              "<modelB>": { ... },
              ...
            }
          }
        """
        start_all = time.perf_counter()
        out: Dict[str, Any] = {"query": query, "results": {}}

        # Validate models exist
        for m in embedding_models:
            if m not in self.registry.embedding_map:
                raise ModelNotFoundError(m)

        # Process each model independently
        for m in embedding_models:
            dataset_id = f"{base_dataset_id}_{m}"
            provider, _ = self.registry.embedding_map[m]

            # Per-model logging start
            log_event(
                "searchmulti.start",
                base_dataset_id=base_dataset_id,
                dataset_id=dataset_id,
                provider=provider.name,
                model=m,
                query=query[:100],
                top_k=top_k,
            )

            t0 = time.perf_counter()
            try:
                if not await self.storage.dataset_exists(dataset_id):
                    raise DatasetNotFoundError(dataset_id)

                # Embed the query with this specific model
                embedding_request = type("EmbeddingRequest", (), {
                    "texts": [query],
                    "model": m,
                })()

                embedding_response = await self.embedding_service.generate_embeddings(
                    embedding_request
                )
                qvec = embedding_response.embeddings[0]

                # Fetch docs for this dataset
                docs = await self.storage.get_dataset(dataset_id)

                # Rank with the same utility used by single-model search
                ranked = find_similar_documents(qvec, docs, top_k or 5)

                # Strip stored vectors from response
                for d in ranked:
                    d.pop("embedding", None)

                out["results"][m] = {
                    "dataset_id": dataset_id,
                    "items": ranked,
                    "total_documents": len(docs),
                }

                # Per-model logging end (success)
                log_event(
                    "searchmulti.end",
                    base_dataset_id=base_dataset_id,
                    dataset_id=dataset_id,
                    provider=provider.name,
                    model=m,
                    ok=True,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    results_count=len(ranked),
                )

            except Exception as e:
                # Per-model logging end (failure)
                log_event(
                    "searchmulti.end",
                    base_dataset_id=base_dataset_id,
                    dataset_id=dataset_id,
                    provider=provider.name,
                    model=m,
                    ok=False,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    error=str(e),
                )
                # Gracefully record the error for this model and continue
                out["results"][m] = {
                    "error": str(e),
                    "items": [],
                }

        # Optional: overall aggregation timing
        out["duration_ms"] = int((time.perf_counter() - start_all) * 1000)
        return out
