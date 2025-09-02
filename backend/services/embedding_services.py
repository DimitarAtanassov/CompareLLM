from __future__ import annotations
from typing import Any, Dict, List, Optional
from time import perf_counter

from langchain_core.documents import Document
from core.embedding_registry import EmbeddingRegistry

def _log(msg: str) -> None:
    print(f"[EmbeddingService] {msg}")

class EmbeddingService:
    def __init__(self, registry: EmbeddingRegistry) -> None:
        self._reg = registry
        _log("Initialized")

    # ---------- Inventory ----------
    def list_embedding_models(self) -> List[str]:
        models = sorted(list(self._reg.embedding_keys()))
        _log(f"list_embedding_models -> {len(models)}")
        return models

    def list_stores(self) -> Dict[str, str]:
        m = self._reg.list_stores()
        _log(f"list_stores -> {len(m)}")
        return m

    # ---------- Store lifecycle ----------
    def create_store(self, store_id: str, embedding_key: str) -> None:
        _log(f"create_store(store_id='{store_id}', embedding_key='{embedding_key}')")
        self._reg.create_store(store_id, embedding_key)

    def delete_store(self, store_id: str) -> None:
        _log(f"delete_store(store_id='{store_id}')")
        self._reg.delete_store(store_id)

    # ---------- Indexing ----------
    async def aadd_texts(
        self,
        store_id: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        t0 = perf_counter()
        vs = self._reg.get_store(store_id)
        ids_out = await vs.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
        _log(f"aadd_texts(store='{store_id}', n={len(texts)}) -> ids={len(ids_out)} (took {(perf_counter()-t0)*1000:.1f} ms)")
        return ids_out

    async def aadd_documents(self, store_id: str, docs: List[Document]) -> List[str]:
        t0 = perf_counter()
        vs = self._reg.get_store(store_id)
        ids_out = await vs.aadd_documents(documents=docs)
        _log(f"aadd_documents(store='{store_id}', n={len(docs)}) -> ids={len(ids_out)} (took {(perf_counter()-t0)*1000:.1f} ms)")
        return ids_out

    # ---------- Search ----------
    async def asimilarity_search(
        self,
        store_id: str,
        query: str,
        *,
        k: int = 5,
        with_scores: bool = False,
    ):
        t0 = perf_counter()
        vs = self._reg.get_store(store_id)
        if with_scores:
            results = await vs.asimilarity_search_with_score(query=query, k=k)
        else:
            results = await vs.asimilarity_search(query=query, k=k)
        _log(
            f"asimilarity_search(store='{store_id}', k={k}, with_scores={with_scores}) "
            f"-> {len(results)} result(s) (took {(perf_counter()-t0)*1000:.1f} ms)"
        )
        return results
