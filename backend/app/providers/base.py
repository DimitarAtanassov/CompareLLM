"""Provider protocols and shared value objects."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.domain.models import ChatMessage, GenerationParams
from app.errors import ValidationError


@dataclass(frozen=True)
class ProviderSpec:
    """Immutable connection details for a provider, derived from ``models.yaml``."""

    key: str
    type: str
    wire: str | None
    base_url: str | None
    api_key: str | None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def effective_wire(self) -> str:
        """The transport to use: explicit ``wire`` overrides ``type``."""
        return self.wire or self.type


@runtime_checkable
class ChatProvider(Protocol):
    """A model that streams a chat completion as plain text deltas."""

    model: str

    def stream(self, messages: list[ChatMessage], params: GenerationParams) -> AsyncIterator[str]:
        """Yield text deltas for the given conversation."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A model that produces embedding vectors for input texts."""

    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...


def parse_target(target: str) -> tuple[str, str]:
    """Split a ``provider:model`` target into its parts."""
    if ":" not in target:
        raise ValidationError(f"Target must be 'provider:model', got '{target}'")
    provider_key, model = target.split(":", 1)
    if not provider_key or not model:
        raise ValidationError(f"Target must be 'provider:model', got '{target}'")
    return provider_key, model


def split_system(
    messages: list[ChatMessage],
) -> tuple[str | None, list[ChatMessage]]:
    """Separate leading/scattered system messages from the conversation.

    Several providers (Anthropic, Gemini) take a dedicated ``system`` field rather
    than a system role in the message list. System messages are concatenated.
    """
    system_parts: list[str] = []
    rest: list[ChatMessage] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
        else:
            rest.append(message)
    system = "\n\n".join(system_parts) if system_parts else None
    return system, rest
