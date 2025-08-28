
import time
from fastapi import APIRouter

router = APIRouter()

_started_at = time.time()

@router.get("/health", summary="Liveness & quick diagnostics")
async def health():
    return {
        "status": "ok",
        "uptime_s": int(time.time() - _started_at),
    }

@router.get("/live", include_in_schema=False)
async def live():
    return {"status": "live"}

@router.get("/ready", include_in_schema=False)
async def ready():
    # If you want deeper checks (API keys, registry loaded), you can expand here.
    return {"status": "ready"}
