
from fastapi import APIRouter

from .chat_routes import router as chat_router
from .embed_routes import router as embed_router
from .health_routes import router as health_router
from .utils_routes import router as utils_router

router = APIRouter()

# Health first (so /health is always available)
router.include_router(health_router, tags=["health"])

# Core feature routers
router.include_router(chat_router, prefix="/v2/chat", tags=["chat"])
router.include_router(embed_router, prefix="/v2", tags=["embeddings", "search", "datasets"])
router.include_router(utils_router, tags=["utils"])  # <-- add this
# Back-compat shims (if callers still hit legacy paths, they map to the same handlers)
# NOTE: Keep these until you confirm all clients migrated.
legacy = APIRouter()
legacy.include_router(chat_router, prefix="/chat")        # e.g., /chat/completions/enhanced
legacy.include_router(embed_router, prefix="")            # e.g., /embeddings
router.include_router(legacy, tags=["compat"])
