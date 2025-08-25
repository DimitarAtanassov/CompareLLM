import time
from typing import Any, Dict, List

from config.logging import log_event
from core.exceptions import DatasetNotFoundError, ModelNotFoundError, ValidationError
from models.enhanced_requests import DatasetUploadRequest
from models.responses import DatasetUploadResponse, DatasetListResponse
from providers.registry import ModelRegistry
from services.embedding_service import EmbeddingService
from storage.base import StorageBackend


class DatasetService:
    """Service for managing datasets with embeddings."""
    
    def __init__(
        self, 
        registry: ModelRegistry, 
        embedding_service: EmbeddingService,
        storage: StorageBackend
    ):
        self.registry = registry
        self.embedding_service = embedding_service
        self.storage = storage
    
    async def upload_dataset(self, request: DatasetUploadRequest) -> DatasetUploadResponse:
        """Upload and process a dataset with embeddings."""
        if request.embedding_model not in self.registry.embedding_map:
            raise ModelNotFoundError(request.embedding_model)
        
        if not request.documents:
            raise ValidationError("documents", "No documents provided")
        
        # Validate all documents have the required text field
        texts = []
        for i, doc in enumerate(request.documents):
            if request.text_field not in doc:
                raise ValidationError(
                    f"documents[{i}]", 
                    f"Missing required field: {request.text_field}"
                )
            texts.append(str(doc[request.text_field]))
        
        provider, model = self.registry.embedding_map[request.embedding_model]
        
        log_event(
            "dataset.upload.start",
            dataset_id=request.dataset_id,
            provider=provider.name,
            model=request.embedding_model,
            document_count=len(request.documents)
        )
        
        start_time = time.perf_counter()
        
        try:
            # Generate embeddings
            embedding_request = type('EmbeddingRequest', (), {
                'texts': texts,
                'model': request.embedding_model
            })()
            
            embedding_response = await self.embedding_service.generate_embeddings(
                embedding_request
            )
            
            # Create dataset key
            dataset_key = f"{request.dataset_id}_{request.embedding_model}"
            
            # Enhance documents with embeddings
            enhanced_docs = []
            for i, doc in enumerate(request.documents):
                enhanced_doc = doc.copy()
                enhanced_doc['embedding'] = embedding_response.embeddings[i]
                enhanced_doc['_text_field'] = request.text_field
                enhanced_doc['_embedding_model'] = request.embedding_model
                enhanced_docs.append(enhanced_doc)
            
            # Store the dataset
            await self.storage.store_dataset(dataset_key, enhanced_docs)
            
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event(
                "dataset.upload.end",
                dataset_id=request.dataset_id,
                provider=provider.name,
                model=request.embedding_model,
                ok=True,
                duration_ms=duration_ms,
                document_count=len(request.documents)
            )
            
            return DatasetUploadResponse(
                dataset_id=dataset_key,
                document_count=len(enhanced_docs),
                embedding_model=request.embedding_model,
                message="Dataset uploaded and embeddings generated successfully"
            )
            
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_event(
                "dataset.upload.end",
                dataset_id=request.dataset_id,
                provider=provider.name,
                model=request.embedding_model,
                ok=False,
                duration_ms=duration_ms,
                error=str(e)
            )
            raise
    
    async def list_datasets(self) -> DatasetListResponse:
        """List all available datasets."""
        datasets = await self.storage.list_datasets()
        
        dataset_info = []
        for dataset_id in datasets:
            docs = await self.storage.get_dataset(dataset_id)
            if docs:
                dataset_info.append({
                    "dataset_id": dataset_id,
                    "document_count": len(docs),
                    "sample_fields": list(docs[0].keys()) if docs else []
                })
        
        return DatasetListResponse(datasets=dataset_info)
    
    async def delete_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Delete a dataset."""
        if not await self.storage.dataset_exists(dataset_id):
            raise DatasetNotFoundError(dataset_id)
        
        doc_count = len(await self.storage.get_dataset(dataset_id))
        await self.storage.delete_dataset(dataset_id)
        
        log_event("dataset.deleted", dataset_id=dataset_id, document_count=doc_count)
        
        return {
            "dataset_id": dataset_id,
            "message": f"Dataset deleted successfully ({doc_count} documents removed)"
        }