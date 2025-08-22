import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.logging import setup_logging, log_event
from config.settings import get_settings
from core.exceptions import AskManyLLMsException


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    setup_logging()
    log_event("app.startup", message="Application starting up")
    
    yield
    
    # Shutdown
    log_event("app.shutdown", message="Application shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Ask Many Models API",
        description="Multi-model chat and embedding API with semantic search",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handlers
    @app.exception_handler(AskManyLLMsException)
    async def custom_exception_handler(request: Request, exc: AskManyLLMsException):
        log_event(
            "app.exception",
            path=str(request.url.path),
            method=request.method,
            error=exc.message,
            code=exc.code,
            details=exc.details
        )
        
        status_code = {
            "MODEL_NOT_FOUND": 404,
            "DATASET_NOT_FOUND": 404,
            "VALIDATION_ERROR": 400,
            "PROVIDER_ERROR": 502,
        }.get(exc.code, 500)
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": exc.message,
                "code": exc.code,
                "details": exc.details
            }
        )
    
    # Include legacy endpoints for backward compatibility
    from api.legacy import router as legacy_router
    app.include_router(legacy_router)
    
    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",  # Changed from "src.main_new:app" to "main:app"
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_config=None,  # Use our custom logging
    )