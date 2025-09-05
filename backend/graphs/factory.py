from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from services.graph_service import GraphService

# Create a global instance of GraphService
_graph_service = GraphService()

# =========================
# MULTI-MODEL EMBEDDING SEARCH GRAPH (non-streaming)
# =========================
def build_embedding_comparison_graph(
    registry: Any,  # EmbeddingRegistry
    embedding_keys: List[str],
    dataset_id: str,
    memory_backend: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Delegate to GraphService for embedding comparison graph construction."""
    return _graph_service.build_embedding_comparison_graph(
        registry=registry,
        embedding_keys=embedding_keys,
        dataset_id=dataset_id,
        memory_backend=memory_backend
    )





# =========================
# SINGLE-MODEL GRAPH (streaming)
# =========================
def build_single_model_graph(
    registry: Any,
    wire: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    memory_backend: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Delegate to GraphService for single model graph construction."""
    return _graph_service.build_single_model_graph(
        registry=registry,
        wire=wire,
        model_kwargs=model_kwargs,
        memory_backend=memory_backend
    )


# =========================
# MULTI-MODEL COMPARE GRAPH (streaming deltas)
# =========================
def build_multi_model_graph(
    registry: Any,
    wires: List[str],
    per_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    memory_backend: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Delegate to GraphService for multi model graph construction."""
    return _graph_service.build_multi_model_graph(
        registry=registry,
        wires=wires,
        per_model_params=per_model_params,
        memory_backend=memory_backend
    )
