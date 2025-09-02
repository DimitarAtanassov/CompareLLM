from __future__ import annotations
from typing import Dict, List, DefaultDict
from collections import defaultdict
from time import perf_counter

def _log(msg: str) -> None:
    print(f"[DatasetCatalog] {msg}")

class DatasetCatalog:
    """
    Tracks which (dataset_id, embedding_key) stores exist and rough doc counts per store.
    """
    def __init__(self) -> None:
        t0 = perf_counter()
        self._stores_by_dataset: DefaultDict[str, List[str]] = defaultdict(list)
        self._doc_counts: Dict[str, int] = {}  # keyed by store_id
        _log(f"Initialized (took {(perf_counter()-t0)*1000:.1f} ms)")

    def add_store(self, dataset_id: str, embedding_key: str, store_id: str):
        if store_id not in self._stores_by_dataset[dataset_id]:
            self._stores_by_dataset[dataset_id].append(store_id)
            _log(f"Added store -> dataset='{dataset_id}', embedding='{embedding_key}', store_id='{store_id}'")
        else:
            _log(f"Store already present -> dataset='{dataset_id}', store_id='{store_id}' (noop)")

    def add_docs(self, store_id: str, n: int):
        before = self._doc_counts.get(store_id, 0)
        self._doc_counts[store_id] = before + n
        _log(f"Incremented doc count -> store_id='{store_id}', +{n}, total={self._doc_counts[store_id]}")

    def remove_dataset(self, dataset_id: str):
        sids = list(self._stores_by_dataset.get(dataset_id, []))
        for sid in sids:
            self._doc_counts.pop(sid, None)
        self._stores_by_dataset.pop(dataset_id, None)
        _log(f"Removed dataset '{dataset_id}', deleted {len(sids)} stores from catalog")

    def datasets(self):
        out = []
        for ds, store_ids in self._stores_by_dataset.items():
            total_docs = sum(self._doc_counts.get(sid, 0) for sid in store_ids)
            out.append({"dataset_id": ds, "document_count": total_docs})
        out.sort(key=lambda x: x["dataset_id"])
        _log(f"Listing datasets -> {len(out)} found")
        return out

    def stores_for_model(self, embedding_key: str) -> List[str]:
        # All store_ids that belong to this embedding_key
        sids = []
        for ds, store_ids in self._stores_by_dataset.items():
            for sid in store_ids:
                if sid.endswith(f"::{embedding_key}") or sid.split("::", 1)[1] == embedding_key:
                    sids.append(sid)
        _log(f"stores_for_model('{embedding_key}') -> {len(sids)} store(s)")
        return sids

    def stores_for_dataset_and_model(self, dataset_id: str, embedding_key: str) -> List[str]:
        sids = [sid for sid in self._stores_by_dataset.get(dataset_id, []) if sid.endswith(f"::{embedding_key}")]
        _log(f"stores_for_dataset_and_model(dataset='{dataset_id}', emb='{embedding_key}') -> {len(sids)} store(s)")
        return sids
