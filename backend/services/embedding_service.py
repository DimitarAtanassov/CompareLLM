from __future__ import annotations
from typing import Any, Dict, List, Optional
from time import perf_counter
from langchain_core.documents import Document
from core.embedding_registry import EmbeddingRegistry
from fastapi import HTTPException

def _log(msg: str) -> None:
    print(f"[EmbeddingService] {msg}")

class EmbeddingService:
    """
    Service layer for embedding operations. Handles all business logic for vector store and embedding management.
    """
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
        try:
            self._reg.create_store(store_id, embedding_key)
        except (KeyError, ValueError) as e:
            _log(f"create_store ERROR: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    def delete_store(self, store_id: str) -> None:
        _log(f"delete_store(store_id='{store_id}')")
        self._reg.delete_store(store_id)

    # ---------- Indexing ----------
    async def index_texts(
        self,
        store_id: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        t0 = perf_counter()
        vs = self._reg.get_store(store_id)
        ids_out = await vs.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
        _log(f"index_texts(store='{store_id}', n={len(texts)}) -> ids={len(ids_out)} (took {(perf_counter()-t0)*1000:.1f} ms)")
        return ids_out

    async def index_docs(self, store_id: str, docs: List[Document]) -> List[str]:
        t0 = perf_counter()
        vs = self._reg.get_store(store_id)
        ids_out = await vs.aadd_documents(documents=docs)
        _log(f"index_docs(store='{store_id}', n={len(docs)}) -> ids={len(ids_out)} (took {(perf_counter()-t0)*1000:.1f} ms)")
        return ids_out

    # ---------- Search ----------
    async def similarity_search(
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
            f"similarity_search(store='{store_id}', k={k}, with_scores={with_scores}) "
            f"-> {len(results)} result(s) (took {(perf_counter()-t0)*1000:.1f} ms)"
        )
        return results

    def get_retriever(self, store_id: str, search_type: str = "similarity", search_kwargs: Optional[Dict[str, Any]] = None):
        return self._reg.get_retriever(store_id, search_type=search_type, search_kwargs=search_kwargs)

    async def compare_across_models(
        self,
        dataset_id: str,
        embedding_models: List[str],
        query: str,
        search_params: Dict[str, Any],
        memory_backend: Any,
    ) -> Dict[str, Any]:
        from graphs.factory import build_embedding_comparison_graph
        import uuid
        graph, _ = build_embedding_comparison_graph(
            registry=self._reg,
            embedding_keys=embedding_models,
            dataset_id=dataset_id,
            memory_backend=memory_backend,
        )
        thread_id = str(uuid.uuid4())
        final_state = await graph.ainvoke(
            {
                "query": query,
                "targets": embedding_models,
                "search_params": search_params,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        results = final_state.get("results", {})
        errors = final_state.get("errors", {})
        for emb_key, err_msg in errors.items():
            if emb_key not in results:
                results[emb_key] = {"items": [], "error": err_msg}
            else:
                results[emb_key]["error"] = err_msg
        return {
            "query": query,
            "dataset_id": dataset_id,
            "k": search_params.get("k"),
            "results": results,
        }
