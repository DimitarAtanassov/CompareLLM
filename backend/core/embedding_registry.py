from __future__ import annotations
from typing import Dict, Iterable, Any, Optional
from time import perf_counter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever

def _log(msg: str) -> None:
    print(f"[EmbeddingRegistry] {msg}")

class EmbeddingRegistry:
    """
    Keeps embedding models and vector stores.
    - Embedding models keyed by 'provider:model' (e.g., 'openai:text-embedding-3-large')
    - Vector stores keyed by 'store_id' (each bound to one embeddings instance)
    """
    def __init__(self) -> None:
        self._embeddings: Dict[str, Any] = {}
        self._stores: Dict[str, InMemoryVectorStore] = {}
        self._store_to_embedding_key: Dict[str, str] = {}
        _log("Initialized empty registry")

    @staticmethod
    def make_embedding_key(provider_key: str, model_name: str) -> str:
        return f"{provider_key}:{model_name}"

    def add_embedding(self, provider_key: str, model_name: str, emb: Any) -> None:
        key = self.make_embedding_key(provider_key, model_name)
        self._embeddings[key] = emb
        _log(f"Added embedding -> {key}")

    def get_embedding(self, provider_key: str, model_name: str) -> Any:
        key = self.make_embedding_key(provider_key, model_name)
        if key not in self._embeddings:
            _log(f"ERROR: Embedding not registered -> {key}")
            raise KeyError(f"Embedding not registered: {key}")
        _log(f"Retrieved embedding -> {key}")
        return self._embeddings[key]

    def embedding_keys(self) -> Iterable[str]:
        keys = list(self._embeddings.keys())
        _log(f"embedding_keys() -> {len(keys)} keys")
        return self._embeddings.keys()

    # ---- Vector store management ----
    def create_store(self, store_id: str, embedding_key: str) -> InMemoryVectorStore:
        if store_id in self._stores:
            _log(f"ERROR: Store already exists -> {store_id}")
            raise ValueError(f"Store '{store_id}' already exists")
        emb = self._embeddings.get(embedding_key)
        if emb is None:
            _log(f"ERROR: Embedding key not found -> {embedding_key}")
            raise KeyError(f"Embedding key not found: {embedding_key}")
        t0 = perf_counter()
        vs = InMemoryVectorStore(emb)
        self._stores[store_id] = vs
        self._store_to_embedding_key[store_id] = embedding_key
        _log(f"Created store '{store_id}' (embedding='{embedding_key}', took {(perf_counter()-t0)*1000:.1f} ms)")
        return vs

    def get_store(self, store_id: str) -> InMemoryVectorStore:
        if store_id not in self._stores:
            _log(f"ERROR: Vector store not found -> {store_id}")
            raise KeyError(f"Vector store '{store_id}' not found")
        _log(f"Retrieved store -> {store_id}")
        return self._stores[store_id]

    def get_retriever(
        self,
        store_id: str,
        *,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> VectorStoreRetriever:
        vs = self.get_store(store_id)
        retr = vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs or {})
        _log(f"Built retriever -> store='{store_id}', type='{search_type}', kwargs={search_kwargs or {}}")
        return retr

    def delete_store(self, store_id: str) -> None:
        existed = store_id in self._stores
        self._stores.pop(store_id, None)
        self._store_to_embedding_key.pop(store_id, None)
        _log(f"Deleted store '{store_id}' (existed={existed})")

    def list_stores(self) -> Dict[str, str]:
        """
        Returns mapping: store_id -> embedding_key
        """
        m = dict(self._store_to_embedding_key)
        _log(f"list_stores() -> {len(m)} store(s)")
        return m

    def store_embedding_key(self, store_id: str) -> Optional[str]:
        ek = self._store_to_embedding_key.get(store_id)
        _log(f"store_embedding_key('{store_id}') -> {ek}")
        return ek

    def approximate_len(self, store_id: str) -> int:
        vs = self.get_store(store_id)
        try:
            n = len(getattr(vs, "store", {}))
        except Exception:
            n = 0
        _log(f"approximate_len('{store_id}') -> {n}")
        return n
