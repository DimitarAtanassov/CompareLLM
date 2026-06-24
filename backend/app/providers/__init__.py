"""Provider layer: native-SDK adapters behind small protocols.

No LangChain. Each adapter speaks directly to a provider SDK and yields plain
text deltas, so the rest of the system never sees provider-specific chunk shapes.
"""

from app.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    ProviderSpec,
    parse_target,
    split_system,
)
from app.providers.registry import ProviderRegistry

__all__ = [
    "ChatProvider",
    "EmbeddingProvider",
    "ProviderRegistry",
    "ProviderSpec",
    "parse_target",
    "split_system",
]
