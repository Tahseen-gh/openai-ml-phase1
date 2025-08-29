from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Awaitable, Callable

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.types import ASGIApp

from .config import settings
from .api.v1 import router as v1_router

# --- logging --------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")


def jsonlog(message: str, **fields: Any) -> None:
    logger.info(json.dumps({"message": message, **fields}))


def _problem(
    title: str, status: int, request_id: str, detail: str | None = None
) -> dict[str, Any]:
    return {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail or title,
        "request_id": request_id,
    }


# --- app ------------------------------------------------------------------
app = FastAPI(title=settings.app_name, debug=settings.debug)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Metrics
if settings.metrics_enabled:
    Instrumentator().instrument(app).expose(app)


# --- Body-size limit middleware ------------------------------------------
class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max(0, int(max_bytes))

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if self.max_bytes and request.method in {"POST", "PUT", "PATCH"}:
            cl = request.headers.get("content-length")
            try:
                if cl and int(cl) > self.max_bytes:
                    return PlainTextResponse(
                        "Request entity too large", status_code=413
                    )
            except Exception:
                pass
            body = await request.body()
            if len(body) > self.max_bytes:
                return PlainTextResponse("Request entity too large", status_code=413)
            # allow downstream to reuse without re-reading
            request._body = body
        return await call_next(request)


app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.request_body_max_bytes)


# --- Simple in-memory rate limiter ---------------------------------------
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


_limiter = _InMemoryRateLimiter(settings.rate_limit_qps)


# --- Request telemetry + rate limit --------------------------------------
@app.middleware("http")
async def add_request_id_and_timing(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    rid = str(uuid.uuid4())
    request.state.request_id = rid
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "-"
    path = request.url.path

    # rate limit (skip if disabled)
    if settings.rate_limit_qps > 0:
        key = f"{client_ip}:{path}"
        if not _limiter.allow(key):
            return JSONResponse(
                _problem("Too Many Requests", 429, rid), status_code=429
            )

    response: Response | None = None
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = (time.perf_counter() - start) * 1000
        jsonlog(
            "request",
            request_id=rid,
            method=request.method,
            path=path,
            status=(response.status_code if response else 500),
            duration_ms=round(dur_ms, 2),
            user_agent=request.headers.get("user-agent", "-"),
            ip=client_ip,
        )


# --- Error handlers (Problem Details) ------------------------------------
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException) -> Response:
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        _problem(exc.detail or "HTTP error", exc.status_code, rid),
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception) -> Response:
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.exception("unhandled", extra={"request_id": rid})
    return JSONResponse(_problem("Internal Server Error", 500, rid), status_code=500)


# --- Auth (API key OR JWT if configured) ---------------------------------
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
bearer = HTTPBearer(auto_error=False)


def require_auth(
    key: str | None = Depends(api_key_header),
    token: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> None:
    if not settings.api_key and not settings.jwt_secret:
        return  # no auth configured
    if settings.api_key and key == settings.api_key:
        return
    if settings.jwt_secret and token:
        try:
            jwt.decode(
                token.credentials,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm],
            )
            return
        except Exception:
            pass
    raise HTTPException(status_code=401, detail="Unauthorized")


APP_VERSION = "0.1.0"
GIT_SHA = os.getenv("GIT_SHA", "dev")


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": APP_VERSION, "git_sha": GIT_SHA}


# Small extra router: a protected ping + a POST sink for body-limit tests
from fastapi import APIRouter  # noqa: E402

_phase2 = APIRouter()


@_phase2.get("/secure/ping", dependencies=[Depends(require_auth)])
def secure_ping() -> dict[str, Any]:
    return {"ok": True, "secure": True}


@_phase2.post("/sink")
def sink_endpoint(_: Any = None) -> dict[str, Any]:
    return {"ok": True}


# Mount existing v1 routes + our small Phase 2 endpoints
app.include_router(v1_router, prefix="/api/v1")
app.include_router(_phase2, prefix="/api/v1")
