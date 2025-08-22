from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class ServiceTier(str, Enum):
    """Service tier options for Anthropic API."""
    AUTO = "auto"
    STANDARD_ONLY = "standard_only"


class ToolChoiceType(str, Enum):
    """Tool choice type options."""
    AUTO = "auto"
    ANY = "any"
    TOOL = "tool"
    NONE = "none"


class ThinkingConfigType(str, Enum):
    """Thinking configuration type options."""
    ENABLED = "enabled"
    DISABLED = "disabled"


class ThinkingConfig(BaseModel):
    """Configuration for Claude's extended thinking."""
    type: ThinkingConfigType
    budget_tokens: Optional[int] = Field(
        None, 
        ge=1024, 
        description="Tokens Claude can use for internal reasoning. Must be ≥1024 and less than max_tokens"
    )
    
    @model_validator(mode='after')
    def validate_budget_tokens(self):
        if self.type == ThinkingConfigType.ENABLED and self.budget_tokens is None:
            raise ValueError("budget_tokens is required when thinking is enabled")
        if self.type == ThinkingConfigType.DISABLED and self.budget_tokens is not None:
            raise ValueError("budget_tokens should not be provided when thinking is disabled")
        return self


class ToolChoice(BaseModel):
    """Tool choice configuration."""
    type: ToolChoiceType
    name: Optional[str] = Field(None, description="Tool name when type is 'tool'")
    disable_parallel_tool_use: Optional[bool] = Field(
        False, 
        description="Whether to disable parallel tool use"
    )
    
    @model_validator(mode='after')
    def validate_tool_name(self):
        if self.type == ToolChoiceType.TOOL and not self.name:
            raise ValueError("name is required when tool choice type is 'tool'")
        if self.type != ToolChoiceType.TOOL and self.name:
            raise ValueError("name should only be provided when tool choice type is 'tool'")
        return self


class AnthropicParameters(BaseModel):
    """
    Comprehensive parameter configuration for Anthropic Claude models.
    Based on the official Anthropic Messages API specification.
    """
    
    # Core parameters
    model: str = Field(..., description="Claude model to use (e.g., claude-sonnet-4-20250514)")
    max_tokens: int = Field(..., gt=0, description="Maximum tokens to generate")
    
    # Optional generation parameters
    temperature: Optional[float] = Field(
        1.0, 
        ge=0.0, 
        le=1.0, 
        description="Sampling temperature (0.0-1.0)"
    )
    top_k: Optional[int] = Field(
        None, 
        gt=0, 
        description="Only sample from top K options for each token"
    )
    top_p: Optional[float] = Field(
        None, 
        gt=0.0, 
        le=1.0, 
        description="Nucleus sampling probability threshold"
    )
    
    # Control parameters
    stop_sequences: Optional[List[str]] = Field(
        None, 
        description="Custom sequences that will stop generation"
    )
    stream: Optional[bool] = Field(
        False, 
        description="Whether to stream the response"
    )
    
    # Advanced features
    thinking: Optional[ThinkingConfig] = Field(
        None, 
        description="Extended thinking configuration"
    )
    tool_choice: Optional[ToolChoice] = Field(
        None, 
        description="How the model should use provided tools"
    )
    
    # Service and infrastructure
    service_tier: Optional[ServiceTier] = Field(
        ServiceTier.AUTO, 
        description="Service tier for request priority"
    )
    container: Optional[str] = Field(
        None, 
        description="Container identifier for reuse across requests"
    )
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Request metadata (e.g., user_id for abuse detection)"
    )
    
    @field_validator("stop_sequences")
    @classmethod
    def validate_stop_sequences(cls, v):
        if v is not None and len(v) == 0:
            return None  # Convert empty list to None
        return v
    
    @model_validator(mode='after')
    def validate_sampling_params(self):
        """Validate that top_p and temperature aren't both specified at extreme values."""
        if (self.temperature is not None and self.top_p is not None and 
            self.temperature != 1.0 and self.top_p != 1.0):
            # This is just a warning case - both can be used but it's not typical
            pass
        return self
    
    def to_anthropic_payload(
        self, 
        messages: List[Dict[str, Any]], 
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Convert parameters to Anthropic API payload format.
        
        Args:
            messages: Conversation messages in Anthropic format
            system: Optional system prompt
            tools: Optional tool definitions
            
        Returns:
            Dictionary ready for Anthropic API request
        """
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages
        }
        
        # Add optional parameters only if they're set
        if self.temperature is not None:
            payload["temperature"] = self.temperature
            
        if self.top_k is not None:
            payload["top_k"] = self.top_k
            
        if self.top_p is not None:
            payload["top_p"] = self.top_p
            
        if self.stop_sequences:
            payload["stop_sequences"] = self.stop_sequences
            
        if self.stream is not None:
            payload["stream"] = self.stream
            
        if system:
            payload["system"] = system
            
        if tools:
            payload["tools"] = tools
            
        if self.thinking:
            if self.thinking.type == ThinkingConfigType.ENABLED:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking.budget_tokens
                }
            else:
                payload["thinking"] = {"type": "disabled"}
                
        if self.tool_choice:
            tool_choice_dict = {"type": self.tool_choice.type.value}
            if self.tool_choice.name:
                tool_choice_dict["name"] = self.tool_choice.name
            if self.tool_choice.disable_parallel_tool_use is not None:
                tool_choice_dict["disable_parallel_tool_use"] = self.tool_choice.disable_parallel_tool_use
            payload["tool_choice"] = tool_choice_dict
            
        if self.service_tier != ServiceTier.AUTO:
            payload["service_tier"] = self.service_tier.value
            
        if self.container:
            payload["container"] = self.container
            
        if self.metadata:
            payload["metadata"] = self.metadata
            
        return payload
    
    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "AnthropicParameters":
        """
        Create AnthropicParameters from a dictionary, handling nested objects.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            AnthropicParameters instance
        """
        # Handle thinking config
        if "thinking" in params and isinstance(params["thinking"], dict):
            thinking_dict = params["thinking"]
            params["thinking"] = ThinkingConfig(**thinking_dict)
            
        # Handle tool choice
        if "tool_choice" in params and isinstance(params["tool_choice"], dict):
            tool_choice_dict = params["tool_choice"]
            params["tool_choice"] = ToolChoice(**tool_choice_dict)
            
        return cls(**params)
    
    def merge_with_defaults(self, **overrides) -> "AnthropicParameters":
        """
        Create a new instance with specified overrides.
        
        Args:
            **overrides: Parameter overrides
            
        Returns:
            New AnthropicParameters instance with overrides applied
        """
        current_dict = self.model_dump(exclude_unset=True)
        current_dict.update(overrides)
        return self.from_dict(current_dict)


class AnthropicModelConfig(BaseModel):
    """
    Model-specific configuration for Anthropic models.
    This can be used to define default parameters per model.
    """
    model_name: str
    default_params: AnthropicParameters
    max_context_tokens: Optional[int] = Field(None, description="Maximum context window size")
    supports_thinking: bool = Field(True, description="Whether model supports extended thinking")
    supports_tools: bool = Field(True, description="Whether model supports tool use")
    
    # Rate limiting info (for client-side awareness)
    default_rpm: Optional[int] = Field(None, description="Default requests per minute limit")
    default_tpm: Optional[int] = Field(None, description="Default tokens per minute limit")
    
    def create_params(self, **overrides) -> AnthropicParameters:
        """Create parameters for this model with optional overrides."""
        return self.default_params.merge_with_defaults(**overrides)


# Predefined configurations for common Claude models
CLAUDE_MODEL_CONFIGS = {
    "claude-sonnet-4-20250514": AnthropicModelConfig(
        model_name="claude-sonnet-4-20250514",
        default_params=AnthropicParameters(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=1.0,
            service_tier=ServiceTier.AUTO
        ),
        max_context_tokens=200000,
        supports_thinking=True,
        supports_tools=True,
        default_rpm=50,
        default_tpm=40000
    ),
    "claude-opus-4-20250514": AnthropicModelConfig(
        model_name="claude-opus-4-20250514", 
        default_params=AnthropicParameters(
            model="claude-opus-4-20250514",
            max_tokens=8192,
            temperature=1.0,
            service_tier=ServiceTier.AUTO
        ),
        max_context_tokens=200000,
        supports_thinking=True,
        supports_tools=True,
        default_rpm=25,
        default_tpm=20000
    ),
    "claude-3-5-sonnet-20241022": AnthropicModelConfig(
        model_name="claude-3-5-sonnet-20241022",
        default_params=AnthropicParameters(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            temperature=1.0,
            service_tier=ServiceTier.AUTO
        ),
        max_context_tokens=200000,
        supports_thinking=False,  # Older model
        supports_tools=True,
        default_rpm=50,
        default_tpm=40000
    )
}


def get_model_config(model_name: str) -> Optional[AnthropicModelConfig]:
    """Get configuration for a specific Claude model."""
    return CLAUDE_MODEL_CONFIGS.get(model_name)


def create_anthropic_params(model: str, **kwargs) -> AnthropicParameters:
    """
    Convenience function to create Anthropic parameters.
    
    Args:
        model: Model name
        **kwargs: Parameter overrides
        
    Returns:
        AnthropicParameters instance
    """
    config = get_model_config(model)
    if config:
        return config.create_params(**kwargs)
    else:
        # Create basic params for unknown models
        base_params = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", 8192),
            "temperature": kwargs.pop("temperature", 1.0),
            **kwargs
        }
        return AnthropicParameters(**base_params)