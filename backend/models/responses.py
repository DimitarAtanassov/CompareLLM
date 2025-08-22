from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelAnswer(BaseModel):
    answer: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[int] = None


class ChatResponse(BaseModel):
    prompt: str
    models: List[str]
    answers: Dict[str, ModelAnswer]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    model: str
    embeddings: List[List[float]]
    usage: EmbeddingUsage


class SearchResponse(BaseModel):
    query: str
    dataset_id: str
    embedding_model: str
    results: List[Dict[str, Any]]
    total_documents: int


class DatasetInfo(BaseModel):
    dataset_id: str
    document_count: int
    sample_fields: List[str]


class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    document_count: int
    embedding_model: str
    message: str


class ProviderInfo(BaseModel):
    name: str
    type: str
    base_url: str
    models: List[str]
    embedding_models: List[str]
    auth_required: bool


class ProvidersResponse(BaseModel):
    providers: List[ProviderInfo]


class EmbeddingModelsResponse(BaseModel):
    embedding_models: List[str]