# main.py
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any, Dict

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from routers import vision
from routers import embeddings
from core.dataset_catalog import DatasetCatalog
from core.embedding_factory import build_embedding_model
from core.embedding_registry import EmbeddingRegistry
from services.embedding_service import EmbeddingService
from core.config_loader import load_config
from core.model_registry import ModelRegistry
from core.model_factory import build_chat_model

# Routers
from routers import providers  # /providers endpoints
from routers import chat
from routers import langgraph

# ✅ LangGraph memory (shared across all graphs/requests)
from langgraph.checkpoint.memory import InMemorySaver


def _log(msg: str) -> None:
    print(f"[Main] {msg}")


def _env_origins() -> list[str]:
    """
    Read allowed origins from env. If unset, default to local dev hosts.
    NOTE: With allow_credentials=True, you cannot use ["*"].
    """
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    # sensible dev defaults
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost",
        "http://127.0.0.1",
    ]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ---- Startup ----
    cfg: Dict[str, Any] = load_config()  # reads config/models.yaml

    app.state.config = cfg
    _log(f"[startup] Loaded config sections: {list(cfg.keys())}")

    # ✅ Create ONE shared LangGraph memory saver for the whole app
    app.state.graph_memory = InMemorySaver()
    _log("[startup] Initialized shared LangGraph memory saver (app.state.graph_memory)")

    reg = ModelRegistry()

    providers_cfg = (cfg.get("providers") or {})
    # ✅ Make providers config available to the registry for provider_type(), etc.
    reg.set_providers_cfg(providers_cfg)
    _log(f"[startup] Found {len(providers_cfg)} providers in config: {list(providers_cfg.keys())}")

    count = 0
    for pkey, pcfg in providers_cfg.items():
        _log(f"[startup] Initializing provider: {pkey} | keys={list(pcfg.keys())}")
        for m in pcfg.get("models") or []:
            _log(f"    -> Building model: {m}")
            try:
                model_obj = build_chat_model(pkey, pcfg, m)
                reg.add(pkey, m, model_obj)
                _log(f"    ✅ Added model '{m}' for provider '{pkey}'")
                count += 1
            except Exception as e:
                _log(f"    ❌ Failed to build model '{m}' for '{pkey}': {e}")

    app.state.registry = reg
    _log(f"[startup] Initialized {count} chat models across {len(providers_cfg)} providers")

    emb_reg = EmbeddingRegistry()
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
        _log("[shutdown] Application shutting down, cleaning up models...")
        # If any models need cleanup, do it here.


def create_app() -> FastAPI:
    app = FastAPI(title="CompareLLM", version="0.1.0", lifespan=lifespan)

    # --- CORS (must be added BEFORE routers) ---
    allow_origins = _env_origins()
    _log(f"[cors] allow_origins={allow_origins}")

    # If you need regex-based dev tunnels (e.g., ngrok), set CORS_ALLOW_ORIGIN_REGEX
    allow_origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX") or None
    if allow_origin_regex:
        _log(f"[cors] allow_origin_regex={allow_origin_regex}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_origin_regex=allow_origin_regex,
        allow_credentials=True,          # if you pass cookies/Authorization
        allow_methods=["GET", "POST", "OPTIONS"],  # keep tight but sufficient
        allow_headers=["*"],             # or enumerate: Authorization, Content-Type, etc.
        expose_headers=["X-Accel-Buffering", "X-Request-ID"],
        max_age=600,                     # cache preflight (seconds)
    )

    # Optional: handle any stray preflight requests explicitly (middleware should handle these already)
    @app.options("/{full_path:path}")
    def options_handler(full_path: str) -> Response:
        return Response(status_code=204)

    # Routers (mounted after CORS middleware)
    app.include_router(providers.router)
    app.include_router(chat.router)
    app.include_router(embeddings.router)
    app.include_router(langgraph.router)
    app.include_router(vision.router)

    # Health + inventory
    @app.get("/health")
    def health():
        _log("[health] Health check called")
        return {"ok": True}

    @app.get("/inventory")
    def inventory():
        reg: ModelRegistry = app.state.registry
        models = sorted(list(reg.keys()))
        _log(f"[inventory] Returning {len(models)} models")
        return {"models": models}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    _log("[main] Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
