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


class AnthropicProviderParams(ProviderParameters):
    """Anthropic-specific parameters."""
    # Extended thinking
    thinking_enabled: Optional[bool] = Field(None, description="Enable extended thinking")
    thinking_budget_tokens: Optional[int] = Field(None, ge=1024, description="Thinking budget tokens")
    
    # Sampling parameters
    top_k: Optional[int] = Field(None, gt=0, description="Top-K sampling")
    top_p: Optional[float] = Field(None, gt=0, le=1, description="Top-P sampling")
    
    # Control parameters
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(None, description="Stream response")
    
    # Tool configuration
    tool_choice_type: Optional[str] = Field(None, description="Tool choice type (auto, any, tool, none)")
    tool_choice_name: Optional[str] = Field(None, description="Specific tool name")
    disable_parallel_tool_use: Optional[bool] = Field(None, description="Disable parallel tool use")
    
    # Service configuration
    service_tier: Optional[str] = Field(None, description="Service tier (auto, standard_only)")
    container: Optional[str] = Field(None, description="Container identifier")
    
    # Metadata
    user_id: Optional[str] = Field(None, description="User identifier for abuse detection")
    
    def to_anthropic_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by AnthropicParameters."""
        result = {}
        
        # Handle thinking configuration
        if self.thinking_enabled is not None:
            if self.thinking_enabled:
                result["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget_tokens or 2048
                }
            else:
                result["thinking"] = {"type": "disabled"}
        
        # Handle tool choice
        if self.tool_choice_type:
            tool_choice = {"type": self.tool_choice_type}
            if self.tool_choice_name:
                tool_choice["name"] = self.tool_choice_name
            if self.disable_parallel_tool_use is not None:
                tool_choice["disable_parallel_tool_use"] = self.disable_parallel_tool_use
            result["tool_choice"] = tool_choice
        
        # Simple field mappings
        simple_fields = {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": self.stop_sequences,
            "stream": self.stream,
            "service_tier": self.service_tier,
            "container": self.container
        }
        
        for field, value in simple_fields.items():
            if value is not None:
                result[field] = value
        
        # Handle metadata
        if self.user_id:
            result["metadata"] = {"user_id": self.user_id}
        
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
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(8192, gt=0, description="Maximum tokens to generate")
    min_tokens: Optional[int] = Field(None, gt=0, description="Minimum tokens to generate")
    
    # System message support
    system: Optional[str] = Field(None, description="System message for the conversation")
    
    # Enhanced parameter support
    model_params: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Per-model parameter overrides (legacy format)"
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
        
        Args:
            provider_type: Provider type (anthropic, openai, gemini, ollama)
            model_name: Model name
            
        Returns:
            Dictionary of provider-specific parameters
        """
        params = {}
        
        # Start with legacy model_params for backward compatibility
        if model_name in self.model_params:
            params.update(self.model_params[model_name])
        
        # Add provider-specific parameters
        if provider_type == "anthropic" and self.anthropic_params:
            params.update(self.anthropic_params.to_anthropic_dict())
        elif provider_type == "openai" and self.openai_params:
            params.update(self.openai_params.model_dump(exclude_unset=True))
        elif provider_type == "gemini" and self.gemini_params:
            params.update(self.gemini_params.model_dump(exclude_unset=True))
        elif provider_type == "ollama" and self.ollama_params:
            params.update(self.ollama_params.model_dump(exclude_unset=True))
        
        # Add global provider parameters
        if self.provider_params and provider_type in self.provider_params:
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
    """Example of how to structure an Anthropic-specific request."""
    model: str = "claude-sonnet-4-20250514"
    messages: List[Dict[str, str]]
    max_tokens: int = 8192
    
    # Anthropic-specific features
    enable_thinking: bool = False
    thinking_budget: Optional[int] = None
    service_tier: str = "auto"
    stop_sequences: Optional[List[str]] = None
    
    def to_request(self) -> EnhancedChatRequest:
        """Convert to enhanced request."""
        chat_messages = [ChatMessage(**msg) for msg in self.messages]
        
        anthropic_params = AnthropicProviderParams(
            thinking_enabled=self.enable_thinking,
            thinking_budget_tokens=self.thinking_budget,
            service_tier=self.service_tier,
            stop_sequences=self.stop_sequences
        )
        
        return EnhancedChatRequest(
            messages=chat_messages,
            models=[self.model],
            max_tokens=self.max_tokens,
            anthropic_params=anthropic_params
        )


# Convenience functions
def create_anthropic_request(
    messages: List[Dict[str, str]],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 8192,
    temperature: float = 1.0,
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
    max_tokens: int = 4096,
    temperature: float = 0.7,
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