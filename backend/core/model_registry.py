# core/model_registry.py
from __future__ import annotations
from typing import Dict, Iterable, Any

class ModelRegistry:
    """
    Holds initialized LangChain ChatModels keyed by their 'model key', e.g. "openai:gpt-4o".
    """
    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        print("[ModelRegistry] Initialized empty registry")

    @staticmethod
    def make_key(provider_key: str, model_name: str) -> str:
        key = f"{provider_key}:{model_name}"
        print(f"[ModelRegistry] make_key -> {key}")
        return key

    def add(self, provider_key: str, model_name: str, model_obj: Any) -> None:
        key = self.make_key(provider_key, model_name)
        self._models[key] = model_obj
        print(f"[ModelRegistry] Added model: {key} ({type(model_obj)})")

    def get(self, provider_key: str, model_name: str) -> Any:
        key = self.make_key(provider_key, model_name)
        if key not in self._models:
            print(f"[ModelRegistry] ERROR: Model not registered -> {key}")
            raise KeyError(f"Model not registered: {key}")
        print(f"[ModelRegistry] Retrieved model: {key}")
        return self._models[key]

    def by_provider(self, provider_key: str) -> Dict[str, Any]:
        prefix = f"{provider_key}:"
        result = {k.split(":", 1)[1]: v for k, v in self._models.items() if k.startswith(prefix)}
        print(f"[ModelRegistry] by_provider({provider_key}) -> {list(result.keys())}")
        return result

    def keys(self) -> Iterable[str]:
        print(f"[ModelRegistry] Current keys: {list(self._models.keys())}")
        return self._models.keys()

    def __len__(self) -> int:
        length = len(self._models)
        print(f"[ModelRegistry] Registry size: {length}")
        return length

    def __contains__(self, key: str) -> bool:
        contains = key in self._models
        print(f"[ModelRegistry] __contains__({key}) -> {contains}")
        return contains
