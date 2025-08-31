# main.py
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import embeddings
from core.dataset_catalog import DatasetCatalog
from core.embedding_factory import build_embedding_model
from core.embedding_registry import EmbeddingRegistry
from services.embedding_services import EmbeddingService
from core.config_loader import load_config
from core.model_registry import ModelRegistry
from core.model_factory import build_chat_model

# Routers
from routers import providers  # your /providers endpoints
from routers import chat


def _log(msg: str) -> None:
    print(f"[Main] {msg}")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ---- Startup ----
    cfg: Dict[str, Any] = load_config()  # reads config/models.yaml
    
    app.state.config = cfg
    print(f"[startup] Loaded config: {list(cfg.keys())}")  # debug

    reg = ModelRegistry()

    providers_cfg = (cfg.get("providers") or {})
    print(f"[startup] Found {len(providers_cfg)} providers in config")  # debug

    count = 0
    for pkey, pcfg in providers_cfg.items():
        print(f"[startup] Initializing provider: {pkey} | keys={list(pcfg.keys())}")  # debug
        for m in pcfg.get("models") or []:
            print(f"    -> Building model: {m}")  # debug
            try:
                model_obj = build_chat_model(pkey, pcfg, m)
                reg.add(pkey, m, model_obj)
                print(f"    ✅ Added model '{m}' for provider '{pkey}' to registry")  # debug
                count += 1
            except Exception as e:
                print(f"    ❌ Failed to build model '{m}' for provider '{pkey}': {e}")  # debug

    app.state.registry = reg
    print(f"[startup] Initialized {count} chat models across {len(providers_cfg)} providers")  # debug

    emb_reg = EmbeddingRegistry()
    providers_cfg = (cfg.get("providers") or {})
    emb_count = 0
    for pkey, pcfg in providers_cfg.items():
        for em in pcfg.get("embedding_models") or []:
            try:
                emb = build_embedding_model(pkey, pcfg, em)
                emb_reg.add_embedding(pkey, em, emb)
                emb_count += 1
            except Exception as e:
                _log(f"❌ Embedding build failed for {pkey}:{em} -> {e}")

    app.state.embedding_registry = emb_reg
    app.state.embedding_service = EmbeddingService(emb_reg)
    app.state.dataset_catalog = DatasetCatalog()
    _log(f"Initialized embeddings -> {emb_count} embedding model(s) available")

    try:
        yield
    finally:
        # ---- Shutdown ----
        print("[shutdown] Application shutting down, cleaning up models...")  # debug
        # If any models need cleanup, do it here.
        pass


app = FastAPI(title="CompareLLM", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(providers.router)
app.include_router(chat.router)
app.include_router(providers.router)
app.include_router(chat.router)
app.include_router(embeddings.router)  

# Optional: quick health check and inventory
@app.get("/health")
def health():
    print("[health] Health check called")  # debug
    return {"ok": True}

@app.get("/inventory")
def inventory():
    reg: ModelRegistry = app.state.registry
    models = sorted(list(reg.keys()))
    print(f"[inventory] Returning {len(models)} models: {models}")  # debug
    return {"models": models}


if __name__ == "__main__":
    import uvicorn
    print("[main] Starting uvicorn server on 0.0.0.0:8000")  # debug
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
