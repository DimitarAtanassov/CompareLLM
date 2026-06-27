"""FastAPI application factory and entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from comparellm import __version__
from comparellm.api.container import AppContainer
from comparellm.api.middleware import RequestContextMiddleware
from comparellm.api.routers import chat, embeddings, health, prompts, providers
from comparellm.errors import register_exception_handlers
from comparellm.log import configure_logging, get_logger
from comparellm.settings import Settings, get_settings

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    # Honor a pre-seeded container (used by tests); otherwise build the real one.
    container = getattr(app.state, "container", None) or AppContainer(settings)
    app.state.container = container
    log.info("application_started", app=settings.app_name, version=__version__)
    try:
        yield
    finally:
        await container.aclose()
        log.info("application_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_origin_regex=settings.cors_allow_origin_regex,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(providers.router)
    app.include_router(chat.router)
    app.include_router(embeddings.router)
    app.include_router(prompts.router)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
    )


if __name__ == "__main__":
    main()
