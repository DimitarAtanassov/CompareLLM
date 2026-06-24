"""Validated schema for the provider catalogue (``models.yaml``).

Previously the YAML was consumed as a raw ``dict`` with defensive ``.get`` calls
scattered everywhere. Modeling it with Pydantic gives us fail-fast validation at
startup and a typed object the rest of the app can rely on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The provider "type" determines which adapter is used. "wire" optionally
# overrides the transport for OpenAI-compatible gateways.
ProviderType = Literal[
    "openai",
    "anthropic",
    "gemini",
    "ollama",
    "cohere",
    "deepseek",
    "cerebras",
    "voyage",
]
ProviderWire = Literal["openai", "anthropic", "gemini", "ollama", "cohere"]


class ProviderConfig(BaseModel):
    """Configuration for a single provider entry in ``models.yaml``."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., description="Provider key, e.g. 'openai' (injected from the mapping).")
    type: ProviderType
    wire: ProviderWire | None = None
    name: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    models: list[str] = Field(default_factory=list)
    embedding_models: list[str] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.name or self.key

    @property
    def requires_api_key(self) -> bool:
        return bool(self.api_key_env)


class ModelsConfig(BaseModel):
    """Root of the validated ``models.yaml`` document."""

    model_config = ConfigDict(extra="ignore")

    providers: dict[str, ProviderConfig]

    def chat_targets(self) -> list[str]:
        """Return every ``provider:model`` chat target across all providers."""
        return [
            f"{key}:{model}"
            for key, provider in self.providers.items()
            for model in provider.models
        ]

    def embedding_targets(self) -> list[str]:
        """Return every ``provider:model`` embedding key across all providers."""
        return [
            f"{key}:{model}"
            for key, provider in self.providers.items()
            for model in provider.embedding_models
        ]
