from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Provider:
    name: str
    type: str
    base_url: str
    api_key_env: Optional[str]
    headers: Dict[str, str]
    models: List[str]
    embedding_models: List[str]

    @property
    def api_key(self) -> Optional[str]:
        import os
        return os.getenv(self.api_key_env) if self.api_key_env else None


class ChatProvider(ABC):
    """Abstract interface for chat providers."""
    
    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 8192,
        min_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """Generate chat completion."""
        pass


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""
    
    @abstractmethod
    async def generate_embeddings(
        self,
        model: str,
        texts: List[str],
        **kwargs: Any
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass