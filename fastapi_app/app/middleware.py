from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from .config import settings
from .problem import problem

try:  # optional OTEL
    from opentelemetry.trace import get_current_span
except Exception:  # pragma: no cover - OTEL optional
    get_current_span = None


class _InMemoryRateLimiter:
    def __init__(self, rate: float) -> None:
        self.rate = max(0.0, float(rate))
        self.capacity = max(1.0, self.rate if self.rate > 0 else 1.0)
        self.state: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, ts = self.state.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - ts) * self.rate)
        ok = tokens >= 1.0
        if ok:
            tokens -= 1.0
        self.state[key] = (tokens, now)
        return ok


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach request IDs, enforce rate limits, and emit JSON access logs."""

    def __init__(self, app: ASGIApp, header_name: str) -> None:
        super().__init__(app)
        self.header_name = header_name
        self._limiter = _InMemoryRateLimiter(settings.rate_limit_qps)
        self.logger = structlog.get_logger("access")

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get(self.header_name) or str(uuid.uuid4())
        request.state.request_id = rid
        start = time.perf_counter()
        client_ip = request.client.host if request.client else "-"
        path = request.url.path

        if settings.rate_limit_qps > 0:
            key = f"{client_ip}:{path}"
            if not self._limiter.allow(key):
                response = JSONResponse(
                    problem("Too Many Requests", 429, rid),
                    status_code=429,
                    media_type="application/problem+json",
                )
                response.headers[self.header_name] = rid
                return response

        response = await call_next(request)
        response.headers[self.header_name] = rid

        dur_ms = (time.perf_counter() - start) * 1000
        log_fields: dict[str, Any] = {
            "request_id": rid,
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "duration_ms": round(dur_ms, 2),
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "-"),
        }
        if get_current_span is not None:
            span = get_current_span()
            ctx = span.get_span_context()
            if ctx.trace_id and ctx.span_id:
                log_fields["trace_id"] = format(ctx.trace_id, "032x")
                log_fields["span_id"] = format(ctx.span_id, "016x")
        self.logger.info("request", **log_fields)
        return response
