# app/backend/models/enhanced_requests.py
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


class ProviderParameters(BaseModel):
    """Base class for provider-specific parameters."""
    pass


class AnthropicProviderParams(BaseModel):
    """Anthropic-specific parameters that clients are allowed to set."""
    model_config = ConfigDict(extra="ignore")  # silently drop unknown keys

    # Extended thinking
    thinking_enabled: Optional[bool] = Field(None, description="Enable extended thinking")
    thinking_budget_tokens: Optional[int] = Field(None, ge=1024, description="Thinking budget tokens")

    # Sampling
    top_k: Optional[int] = Field(None, gt=0, description="Top-K sampling")
    top_p: Optional[float] = Field(None, gt=0, le=1, description="Top-P sampling")

    # Control
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(None, description="Stream response")

    # (Optional) infra hint you may want to keep
    container: Optional[str] = Field(None, description="Container identifier")

    def to_anthropic_dict(self) -> Dict[str, Any]:
        """Convert to Anthropic payload fields (service tier & tools removed)."""
        result: Dict[str, Any] = {}

        if self.thinking_enabled is not None:
            result["thinking"] = (
                {"type": "disabled"}
                if not self.thinking_enabled
                else {"type": "enabled", "budget_tokens": self.thinking_budget_tokens or 2048}
            )

        simple_fields = {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": self.stop_sequences,
            "stream": self.stream,
            "container": self.container,
        }
        for k, v in simple_fields.items():
            if v is not None:
                result[k] = v
                
        return result

class OpenAIProviderParams(ProviderParameters):
    """OpenAI-specific parameters."""
    top_p: Optional[float] = Field(None, gt=0, le=1, description="Top-P sampling")
    frequency_penalty: Optional[float] = Field(None, ge=-2, le=2, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, ge=-2, le=2, description="Presence penalty")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(None, description="Stream response")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")
    seed: Optional[int] = Field(None, description="Random seed")
    response_format: Optional[Dict[str, Any]] = Field(None, description="Response format")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tool definitions")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice")


class GeminiProviderParams(ProviderParameters):
    """Google Gemini-specific parameters."""
    top_k: Optional[int] = Field(None, gt=0, description="Top-K sampling")
    top_p: Optional[float] = Field(None, gt=0, le=1, description="Top-P sampling")
    candidate_count: Optional[int] = Field(None, gt=0, le=4, description="Number of candidates")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    safety_settings: Optional[List[Dict[str, Any]]] = Field(None, description="Safety settings")


class OllamaProviderParams(ProviderParameters):
    """Ollama-specific parameters."""
    mirostat: Optional[int] = Field(None, ge=0, le=2, description="Mirostat algorithm")
    mirostat_eta: Optional[float] = Field(None, gt=0, description="Mirostat learning rate")
    mirostat_tau: Optional[float] = Field(None, gt=0, description="Mirostat target entropy")
    num_ctx: Optional[int] = Field(None, gt=0, description="Context window size")
    repeat_last_n: Optional[int] = Field(None, ge=0, description="Repeat penalty lookback")
    repeat_penalty: Optional[float] = Field(None, gt=0, description="Repeat penalty")
    tfs_z: Optional[float] = Field(None, gt=0, description="Tail free sampling")
    seed: Optional[int] = Field(None, description="Random seed")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    top_k: Optional[int] = Field(None, gt=0, description="Top-K sampling")
    top_p: Optional[float] = Field(None, gt=0, le=1, description="Top-P sampling")
    format: Optional[str] = Field(None, description="Response format (json)")


class EnhancedChatRequest(BaseModel):
    """Enhanced chat request with provider-specific parameter support."""
    
    # Core request fields (same as before)
    prompt: Optional[str] = Field(None, description="Legacy single prompt (deprecated)")
    messages: Optional[List[ChatMessage]] = Field(None, description="Conversation messages")
    
    models: Optional[List[str]] = Field(None, description="List of models to use")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")
    min_tokens: Optional[int] = Field(None, gt=0, description="Minimum tokens to generate")
    
    # System message support
    system: Optional[str] = Field(None, description="System message for the conversation")
    
    # Enhanced parameter support
    model_params: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Per-model parameter overrides (optional)"
    )
    
    # Provider-specific parameters
    anthropic_params: Optional[AnthropicProviderParams] = Field(
        None, description="Anthropic-specific parameters"
    )
    openai_params: Optional[OpenAIProviderParams] = Field(
        None, description="OpenAI-specific parameters"  
    )
    gemini_params: Optional[GeminiProviderParams] = Field(
        None, description="Gemini-specific parameters"
    )
    ollama_params: Optional[OllamaProviderParams] = Field(
        None, description="Ollama-specific parameters"
    )
    
    # Global provider parameters (alternative to provider-specific)
    provider_params: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, description="Provider-specific parameters by provider type"
    )
    
    model_config = ConfigDict(extra="ignore")
    
    @model_validator(mode='after')
    def validate_prompt_or_messages(self):
        prompt = self.prompt
        messages = self.messages
        
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        
        if prompt and messages:
            raise ValueError("Cannot provide both 'prompt' and 'messages'")
        
        if prompt is not None and not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        if messages is not None and len(messages) == 0:
            raise ValueError("Messages array cannot be empty")
            
        return self
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert the request to a standardized messages format."""
        if self.messages:
            message_dicts = []
            
            if self.system:
                message_dicts.append({"role": "system", "content": self.system})
            
            for msg in self.messages:
                message_dicts.append({"role": msg.role, "content": msg.content})
                
            return message_dicts
        
        elif self.prompt:
            message_dicts = []
            
            if self.system:
                message_dicts.append({"role": "system", "content": self.system})
            else:
                message_dicts.append({"role": "system", "content": "Answer clearly and concisely."})
            
            message_dicts.append({"role": "user", "content": self.prompt})
            
            return message_dicts
        
        else:
            raise ValueError("No prompt or messages provided")
    
    def get_provider_params(self, provider_type: str, model_name: str) -> Dict[str, Any]:
        """
        Get provider-specific parameters for a given provider type and model.
        """
        params: Dict[str, Any] = {}

        if self.model_params and model_name in self.model_params:
            params.update(self.model_params[model_name])

        if provider_type == "anthropic" and self.anthropic_params:
            params.update(self.anthropic_params.to_anthropic_dict())
        elif provider_type == "openai" and self.openai_params:
            params.update(self.openai_params.model_dump(exclude_unset=True))
        elif provider_type == "gemini" and self.gemini_params:
            params.update(self.gemini_params.model_dump(exclude_unset=True))
        elif provider_type == "ollama" and self.ollama_params:
            params.update(self.ollama_params.model_dump(exclude_unset=True))

        if getattr(self, "provider_params", None) and provider_type in self.provider_params:
            params.update(self.provider_params[provider_type])

        return params



# OpenAI-compatible request models (enhanced)
class EnhancedOpenAIChatRequest(BaseModel):
    """Enhanced OpenAI-compatible chat request with provider parameter support."""
    model: str = Field(..., description="ID of the model to use")
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages")
    temperature: Optional[float] = Field(1.0, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress")
    
    # Provider-specific extensions
    anthropic_params: Optional[AnthropicProviderParams] = Field(
        None, description="Anthropic-specific parameters when using Claude models"
    )
    openai_params: Optional[OpenAIProviderParams] = Field(
        None, description="OpenAI-specific parameters"
    )
    gemini_params: Optional[GeminiProviderParams] = Field(
        None, description="Gemini-specific parameters"
    )
    
    def to_enhanced_request(self) -> EnhancedChatRequest:
        """Convert to internal enhanced request format."""
        # Convert messages format
        chat_messages = []
        for msg in self.messages:
            chat_messages.append(ChatMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", "")
            ))
        
        return EnhancedChatRequest(
            messages=chat_messages,
            models=[self.model],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            anthropic_params=self.anthropic_params,
            openai_params=self.openai_params,
            gemini_params=self.gemini_params
        )


# Example usage classes
class AnthropicChatExample(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    messages: List[Dict[str, str]]
    max_tokens: int = 8192

    enable_thinking: bool = False
    thinking_budget: Optional[int] = None
    stop_sequences: Optional[List[str]] = None

    def to_request(self) -> EnhancedChatRequest:
        chat_messages = [ChatMessage(**msg) for msg in self.messages]
        anthropic_params = AnthropicProviderParams(
            thinking_enabled=self.enable_thinking,
            thinking_budget_tokens=self.thinking_budget,
            stop_sequences=self.stop_sequences,
        )
        return EnhancedChatRequest(
            messages=chat_messages,
            models=[self.model],
            max_tokens=self.max_tokens,
            anthropic_params=anthropic_params,
        )



# Convenience functions
def create_anthropic_request(
    messages: List[Dict[str, str]],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    thinking_enabled: bool = False,
    thinking_budget: Optional[int] = None,
    **kwargs
) -> EnhancedChatRequest:
    """Create an enhanced request optimized for Anthropic models."""
    chat_messages = [ChatMessage(**msg) for msg in messages]
    
    anthropic_params = AnthropicProviderParams(
        thinking_enabled=thinking_enabled,
        thinking_budget_tokens=thinking_budget,
        **kwargs
    )
    
    return EnhancedChatRequest(
        messages=chat_messages,
        models=[model],
        max_tokens=max_tokens,
        temperature=temperature,
        anthropic_params=anthropic_params
    )


def create_openai_request(
    messages: List[Dict[str, str]],
    model: str = "gpt-4",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> EnhancedChatRequest:
    """Create an enhanced request optimized for OpenAI models."""
    chat_messages = [ChatMessage(**msg) for msg in messages]
    
    openai_params = OpenAIProviderParams(**kwargs)
    
    return EnhancedChatRequest(
        messages=chat_messages,
        models=[model],
        max_tokens=max_tokens,
        temperature=temperature,
        openai_params=openai_params
    )

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


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    embedding_model: str = Field(..., description="Embedding model used for the dataset")
    dataset_id: str = Field(..., description="Dataset identifier")
    top_k: Optional[int] = Field(5, gt=0, le=100, description="Number of results to return")
    
    model_config = ConfigDict(extra="ignore")
