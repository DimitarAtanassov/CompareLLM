from __future__ import annotations
from typing import Annotated, Dict, List, Literal, TypedDict
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
    targets: List[str]
    results: Annotated[Dict[str, str], merge_str_dict]  # <- reducer
    errors: Annotated[Dict[str, str], merge_str_dict]   # <- reducer
    per_model_params: Dict[str, dict]
