"""Application error hierarchy and FastAPI exception handlers.

Errors raised by the domain/provider layers are plain Python exceptions with no
FastAPI coupling. The handlers registered here translate them into
``application/problem+json`` responses (RFC 9457).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from comparellm.log import get_logger

log = get_logger(__name__)


class AppError(Exception):
    """Base class for expected, client-relevant application errors."""

    status_code: int = 500
    title: str = "Internal Server Error"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class NotFoundError(AppError):
    status_code = 404
    title = "Not Found"


class ValidationError(AppError):
    status_code = 400
    title = "Bad Request"


class ConfigurationError(AppError):
    status_code = 503
    title = "Service Misconfigured"


class ProviderError(AppError):
    """Raised when an upstream provider call fails."""

    status_code = 502
    title = "Upstream Provider Error"


class UpstreamError(AppError):
    """Raised when a dependent service (e.g. Floating Prompts) fails."""

    status_code = 502
    title = "Upstream Service Error"


def _problem(status: int, title: str, detail: str, instance: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": "about:blank",
            "title": title,
            "status": status,
            "detail": detail,
            "instance": instance,
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that render exceptions as problem+json."""

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        log.warning(
            "app_error",
            status=exc.status_code,
            title=exc.title,
            detail=exc.detail,
            path=request.url.path,
        )
        return _problem(exc.status_code, exc.title, exc.detail, str(request.url.path))

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.error(
            "unhandled_exception",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path,
            exc_info=exc,
        )
        return _problem(
            500,
            "Internal Server Error",
            "An unexpected error occurred.",
            str(request.url.path),
        )
