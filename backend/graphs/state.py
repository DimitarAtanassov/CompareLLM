from __future__ import annotations
from typing import Annotated, Dict, List, Literal, Optional, TypedDict, Any
from langgraph.graph.message import add_messages

def merge_str_dict(old: Dict[str, str] | None, new: Dict[str, str] | None) -> Dict[str, str]:
    if old is None:
        old = {}
    if new:
        # overwrite per key (we send full-so-far strings)
        old.update(new)
    return old

class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class SingleState(TypedDict):
    messages: Annotated[List[ChatMessage], add_messages]
    meta: Dict[str, str]

class MultiState(TypedDict):
    messages: Annotated[List[ChatMessage], add_messages]
    query: Optional[str] # for embedding graph
    targets: List[str]
    results: Annotated[Dict[str, str], merge_str_dict]  # <- reducer
    errors: Annotated[Dict[str, str], merge_str_dict]   # <- reducer

# ---- For embedding graphs ----
class EmbeddingSearchResultItem(TypedDict):
    page_content: str
    metadata: Dict[str, Any]
    score: Optional[float]

class EmbeddingSearchResult(TypedDict):
    items: List[EmbeddingSearchResultItem]
    error: Optional[str]

def merge_embedding_results(
    old: Dict[str, EmbeddingSearchResult] | None,
    new: Dict[str, EmbeddingSearchResult] | None
) -> Dict[str, EmbeddingSearchResult]:
    if old is None:
        old = {}
    if new:
        old.update(new)
    return old

class EmbeddingState(TypedDict):
    query: str
    targets: List[str] # embedding_keys
    search_params: Dict[str, Any]
    results: Annotated[Dict[str, EmbeddingSearchResult], merge_embedding_results]
    errors: Annotated[Dict[str, str], merge_str_dict]
