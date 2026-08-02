"""
api/rate_limit.py

Per-IP rate-limit middleware (ARCHITECTURE.md §5, §9 cost controls): a
public scoring endpoint on a billing-enabled account is a financial risk,
and the cheapest defense is capping request rate before anything expensive
(model load, DuckDB, disk) runs.

In-memory sliding window, deliberately simple: a deque of request
timestamps per client IP, trimmed to the window on each check. This is
per-PROCESS state -- correct for the single Cloud Run instance
ARCHITECTURE.md's cost controls target (min-instances=0, low max-instances),
but it does NOT coordinate across multiple instances. That's an accepted
limitation of the same shape as the destination-state snapshot's "frozen at
one point in time" caveat: documented rather than silently assumed away.
"""

import time
from collections import defaultdict, deque

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .errors import PROBLEM_JSON

DEFAULT_MAX_REQUESTS = 60
DEFAULT_WINDOW_SECONDS = 60.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = DEFAULT_MAX_REQUESTS,
                 window_seconds: float = DEFAULT_WINDOW_SECONDS,
                 exempt_paths: frozenset[str] = frozenset({"/health"})):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.exempt_paths = exempt_paths
        self._hits: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        hits = self._hits[client_ip]

        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()

        if len(hits) >= self.max_requests:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type=PROBLEM_JSON,
                content={
                    "type": "about:blank",
                    "title": "Too many requests",
                    "status": status.HTTP_429_TOO_MANY_REQUESTS,
                    "detail": f"Rate limit of {self.max_requests} requests per "
                              f"{self.window_seconds:.0f}s exceeded.",
                    "instance": str(request.url.path),
                },
                headers={"Retry-After": str(int(self.window_seconds))},
            )

        hits.append(now)
        return await call_next(request)
