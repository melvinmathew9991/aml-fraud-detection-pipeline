"""
api/limits.py

Rejects an oversized POST /score/batch body by Content-Length BEFORE
Starlette reads or Pydantic parses it (ARCHITECTURE.md §5: "Batch size
capped (default 10,000 rows / 10MB) so a single request cannot exhaust a
512MiB instance"). The row cap in api/schemas.py's BatchRequest already
bounds typical payloads well under 10MB for this fixed-field schema, but a
pathological payload (e.g. very long nameDest/nameOrig strings) could
inflate byte size without breaching the row count -- this catches that
case without paying the cost of reading the body first.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .errors import PROBLEM_JSON
from .schemas import BATCH_MAX_BYTES

LIMITED_PATHS = frozenset({"/score/batch"})


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int = BATCH_MAX_BYTES, limited_paths=LIMITED_PATHS):
        super().__init__(app)
        self.max_bytes = max_bytes
        self.limited_paths = limited_paths

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.limited_paths:
            content_length = request.headers.get("content-length")
            if content_length is not None and int(content_length) > self.max_bytes:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    media_type=PROBLEM_JSON,
                    content={
                        "type": "about:blank",
                        "title": "Payload too large",
                        "status": status.HTTP_413_CONTENT_TOO_LARGE,
                        "detail": f"Body of {content_length} bytes exceeds the "
                                  f"{self.max_bytes} byte limit for {request.url.path}.",
                        "instance": str(request.url.path),
                    },
                )
        return await call_next(request)
