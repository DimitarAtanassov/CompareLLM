from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., min_length=1, description="Message content")
    
    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ["system", "user", "assistant"]:
            raise ValueError("Role must be one of: system, user, assistant")
        return v


class ChatRequest(BaseModel):
    # Support both legacy prompt-based and new messages-based formats
    prompt: Optional[str] = Field(None, description="Legacy single prompt (deprecated)")
    messages: Optional[List[ChatMessage]] = Field(None, description="Conversation messages")
    
    models: Optional[List[str]] = Field(None, description="List of models to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(8192, gt=0, description="Maximum tokens to generate")
    min_tokens: Optional[int] = Field(None, gt=0, description="Minimum tokens to generate")
    model_params: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Per-model parameter overrides"
    )
    
    # Add system message support
    system: Optional[str] = Field(None, description="System message for the conversation")
    
    model_config = ConfigDict(extra="ignore")
    
    @model_validator(mode='after')
    def validate_prompt_or_messages(self):
        prompt = self.prompt
        messages = self.messages
        
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        
        if prompt and messages:
            raise ValueError("Cannot provide both 'prompt' and 'messages'")
        
        # If using legacy prompt, validate it has content
        if prompt is not None and not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        # If using messages, validate we have at least one message
        if messages is not None and len(messages) == 0:
            raise ValueError("Messages array cannot be empty")
            
        return self
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert the request to a standardized messages format."""
        if self.messages:
            # Convert ChatMessage objects to dict format
            message_dicts = []
            
            # Add system message if provided
            if self.system:
                message_dicts.append({"role": "system", "content": self.system})
            
            # Add conversation messages
            for msg in self.messages:
                message_dicts.append({"role": msg.role, "content": msg.content})
                
            return message_dicts
        
        elif self.prompt:
            # Legacy format - convert to messages
            message_dicts = []
            
            # Add default system message if none provided
            if self.system:
                message_dicts.append({"role": "system", "content": self.system})
            else:
                message_dicts.append({"role": "system", "content": "Answer clearly and concisely."})
            
            # Add the prompt as a user message
            message_dicts.append({"role": "user", "content": self.prompt})
            
            return message_dicts
        
        else:
            raise ValueError("No prompt or messages provided")


class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts to embed")
    model: str = Field(..., description="Embedding model to use")
    
    model_config = ConfigDict(extra="ignore")
    
    @field_validator("texts")
    @classmethod
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
    documents: List[Dict[str, Any]] = Field(..., min_length=1, description="List of documents")
    embedding_model: str = Field(..., description="Model to use for generating embeddings")
    text_field: Optional[str] = Field("text", description="Field containing text to embed")
    
    model_config = ConfigDict(extra="ignore")
    
    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Dataset ID must contain only alphanumeric characters, hyphens, and underscores")
        return v