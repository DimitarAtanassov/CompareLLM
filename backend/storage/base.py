from abc import ABC, abstractmethod
from typing import Any, Dict, List


class StorageBackend(ABC):
    """Abstract interface for storage backends."""
    
    @abstractmethod
    async def store_dataset(self, dataset_id: str, documents: List[Dict[str, Any]]) -> None:
        """Store a dataset with its documents."""
        pass
    
    @abstractmethod
    async def get_dataset(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Retrieve a dataset by ID."""
        pass
    
    @abstractmethod
    async def dataset_exists(self, dataset_id: str) -> bool:
        """Check if a dataset exists."""
        pass
    
    @abstractmethod
    async def list_datasets(self) -> List[str]:
        """List all dataset IDs."""
        pass
    
    @abstractmethod
    async def delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset."""
        pass