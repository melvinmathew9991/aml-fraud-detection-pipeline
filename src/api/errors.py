"""
api/errors.py

RFC 7807 (problem+json) error responses (ARCHITECTURE.md §5): "Errors
return RFC 7807 problem+json, never a stack trace." Registered against
FastAPI's RequestValidationError (422s from Pydantic), HTTPException
(explicit 4xx/5xx raised in route handlers), and a catch-all for anything
unhandled, so a bug in a route never leaks a Python traceback to a caller.
"""

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("api.errors")

PROBLEM_JSON = "application/problem+json"


def _problem_response(status_code: int, title: str, detail: str, instance: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        media_type=PROBLEM_JSON,
        content={
            "type": "about:blank",
            "title": title,
            "status": status_code,
            "detail": detail,
            "instance": instance,
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return _problem_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            title="Request validation failed",
            detail="; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            ),
            instance=str(request.url.path),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return _problem_response(
            exc.status_code,
            title=exc.detail if isinstance(exc.detail, str) else "Request failed",
            detail=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            instance=str(request.url.path),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s", request.url.path)
        return _problem_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            title="Internal server error",
            detail="An unexpected error occurred.",
            instance=str(request.url.path),
        )
