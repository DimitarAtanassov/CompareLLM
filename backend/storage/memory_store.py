import asyncio
from typing import Any, Dict, List

from storage.base import StorageBackend
from core.exceptions import DatasetNotFoundError


class MemoryStorageBackend(StorageBackend):
    """In-memory storage backend for development and testing."""
    
    def __init__(self):
        self._datasets: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
    
    async def store_dataset(self, dataset_id: str, documents: List[Dict[str, Any]]) -> None:
        """Store a dataset with its documents."""
        async with self._lock:
            self._datasets[dataset_id] = documents.copy()
    
    async def get_dataset(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Retrieve a dataset by ID."""
        async with self._lock:
            if dataset_id not in self._datasets:
                raise DatasetNotFoundError(dataset_id)
            return self._datasets[dataset_id].copy()
    
    async def dataset_exists(self, dataset_id: str) -> bool:
        """Check if a dataset exists."""
        async with self._lock:
            return dataset_id in self._datasets
    
    async def list_datasets(self) -> List[str]:
        """List all dataset IDs."""
        async with self._lock:
            return list(self._datasets.keys())
    
    async def delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset."""
        async with self._lock:
            if dataset_id not in self._datasets:
                raise DatasetNotFoundError(dataset_id)
            del self._datasets[dataset_id]