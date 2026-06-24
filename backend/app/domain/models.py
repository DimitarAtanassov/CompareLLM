"""Provider-agnostic domain models shared across services and the API layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """A single chat message in a conversation."""

    model_config = ConfigDict(extra="ignore")

    role: Role
    content: str


class GenerationParams(BaseModel):
    """Unified generation parameters.

    These are provider-agnostic; each adapter translates the relevant subset into
    its own SDK kwargs via :mod:`app.providers.params`. Unknown/extra knobs can be
    passed through ``extra`` for advanced use.
    """

    model_config = ConfigDict(extra="ignore")

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    # Reasoning budget (Gemini 2.5 / Anthropic extended thinking).
    thinking_budget: int | None = None
    extra: dict[str, object] = Field(default_factory=dict)


class SearchHit(BaseModel):
    """A single similarity-search result."""

    page_content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    score: float | None = None
