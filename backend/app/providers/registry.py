"""Provider registry: resolves ``provider:model`` targets to live adapters.

The registry holds immutable :class:`ProviderSpec` connection details derived from
the validated config and constructs adapters on demand. Adapter modules (and their
heavy SDK imports) are imported lazily so that unused providers never need their
SDK installed and so the registry itself stays cheap to import (e.g. in tests).
"""

from __future__ import annotations

import os

from app.config.schema import ModelsConfig, ProviderConfig
from app.errors import NotFoundError, ValidationError
from app.log import get_logger
from app.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    ProviderSpec,
    parse_target,
)
from app.settings import Settings

log = get_logger(__name__)

# Provider types that speak the OpenAI wire protocol.
_OPENAI_CHAT_TYPES = frozenset({"openai", "deepseek", "cerebras", "ollama"})
_OPENAI_EMBED_TYPES = frozenset({"openai", "deepseek", "cerebras", "ollama"})


class ProviderRegistry:
    """Builds and caches provider specs; mints adapters per request."""

    def __init__(self, config: ModelsConfig, settings: Settings) -> None:
        self._config = config
        self._settings = settings
        self._specs: dict[str, ProviderSpec] = {
            key: self._build_spec(provider) for key, provider in config.providers.items()
        }
        log.info("provider_registry_ready", providers=sorted(self._specs))

    # --- Spec construction ---
    def _build_spec(self, provider: ProviderConfig) -> ProviderSpec:
        api_key = os.getenv(provider.api_key_env) if provider.api_key_env else None
        if provider.requires_api_key and not api_key:
            log.warning("provider_api_key_missing", provider=provider.key, env=provider.api_key_env)
        base_url = provider.base_url
        # Ollama exposes an OpenAI-compatible API under /v1.
        if provider.type == "ollama" and base_url:
            base_url = base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
        return ProviderSpec(
            key=provider.key,
            type=provider.type,
            wire=provider.wire,
            base_url=base_url,
            api_key=api_key,
            headers=provider.headers,
        )

    def _spec(self, provider_key: str) -> ProviderSpec:
        spec = self._specs.get(provider_key)
        if spec is None:
            raise NotFoundError(f"Unknown provider: '{provider_key}'")
        return spec

    # --- Inventory ---
    def chat_models(self) -> list[str]:
        return sorted(self._config.chat_targets())

    def embedding_models(self) -> list[str]:
        return sorted(self._config.embedding_targets())

    def has_chat(self, target: str) -> bool:
        return target in set(self._config.chat_targets())

    def has_embedding(self, target: str) -> bool:
        return target in set(self._config.embedding_targets())

    # --- Adapter factories ---
    def get_chat(self, target: str) -> ChatProvider:
        provider_key, model = parse_target(target)
        spec = self._spec(provider_key)
        kind = "openai" if spec.type in _OPENAI_CHAT_TYPES or spec.wire == "openai" else spec.type
        timeout = self._settings.request_timeout_seconds

        if kind == "openai":
            from app.providers.openai_compat import OpenAICompatChat

            return OpenAICompatChat(spec, model, timeout)
        if kind == "anthropic":
            from app.providers.anthropic_provider import AnthropicChat

            return AnthropicChat(spec, model, timeout, self._settings.anthropic_default_max_tokens)
        if kind == "gemini":
            from app.providers.gemini_provider import GeminiChat

            return GeminiChat(spec, model, timeout)
        if kind == "cohere":
            from app.providers.cohere_provider import CohereChat

            return CohereChat(spec, model, timeout)
        raise ValidationError(f"No chat adapter for provider type '{spec.type}'")

    def get_embedder(self, target: str) -> EmbeddingProvider:
        provider_key, model = parse_target(target)
        spec = self._spec(provider_key)
        kind = "openai" if spec.type in _OPENAI_EMBED_TYPES or spec.wire == "openai" else spec.type
        timeout = self._settings.request_timeout_seconds

        if kind == "openai":
            from app.providers.openai_compat import OpenAICompatEmbeddings

            return OpenAICompatEmbeddings(spec, model, timeout)
        if kind == "gemini":
            from app.providers.gemini_provider import GeminiEmbeddings

            return GeminiEmbeddings(spec, model, timeout)
        if kind == "cohere":
            from app.providers.cohere_provider import CohereEmbeddings

            return CohereEmbeddings(spec, model, timeout)
        if kind == "voyage":
            from app.providers.voyage_provider import VoyageEmbeddings

            return VoyageEmbeddings(spec, model, timeout)
        raise ValidationError(f"No embedding adapter for provider type '{spec.type}'")

    def providers(self) -> dict[str, ProviderConfig]:
        return self._config.providers
