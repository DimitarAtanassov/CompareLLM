from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, validator


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The chat prompt")
    models: Optional[List[str]] = Field(None, description="List of models to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(8192, gt=0, description="Maximum tokens to generate")
    min_tokens: Optional[int] = Field(None, gt=0, description="Minimum tokens to generate")
    model_params: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Per-model parameter overrides"
    )
    
    model_config = ConfigDict(extra="ignore")


class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of texts to embed")
    model: str = Field(..., description="Embedding model to use")
    
    model_config = ConfigDict(extra="ignore")
    
    @validator("texts")
    def validate_texts(cls, v):
        if not v:
            raise ValueError("At least one text must be provided")
        for text in v:
            if not text.strip():
                raise ValueError("Empty texts are not allowed")
        return v


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    embedding_model: str = Field(..., description="Embedding model used for the dataset")
    dataset_id: str = Field(..., description="Dataset identifier")
    top_k: Optional[int] = Field(5, gt=0, le=100, description="Number of results to return")
    
    model_config = ConfigDict(extra="ignore")


class DatasetUploadRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1, description="Unique dataset identifier")
    documents: List[Dict[str, Any]] = Field(..., min_items=1, description="List of documents")
    embedding_model: str = Field(..., description="Model to use for generating embeddings")
    text_field: Optional[str] = Field("text", description="Field containing text to embed")
    
    model_config = ConfigDict(extra="ignore")
    
    @validator("dataset_id")
    def validate_dataset_id(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Dataset ID must contain only alphanumeric characters, hyphens, and underscores")
        return v
