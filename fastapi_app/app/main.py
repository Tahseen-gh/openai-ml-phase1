from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import jwt
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from rag.retriever import BackendName
from rag.retriever import search as rag_search

from .api.v1 import router as v1_router
from .config import settings
from .logging import configure_logging
from .middleware import RequestIdMiddleware
from .problem import problem

if settings.fuzz_mode:
    settings.rate_limit_qps = float(os.getenv("RATE_LIMIT_QPS", settings.rate_limit_qps))
    settings.request_body_max_bytes = int(
        os.getenv("REQUEST_BODY_MAX_BYTES", settings.request_body_max_bytes)
    )
else:
    settings.rate_limit_qps = 5.0
    settings.request_body_max_bytes = 100_000

# --- logging --------------------------------------------------------------
logger = structlog.get_logger("app")


# --- app ------------------------------------------------------------------
app = FastAPI(title=settings.app_name, debug=settings.debug, openapi_url="/openapi.json")

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
                    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
                    request.state.request_id = rid
                    return JSONResponse(
                        problem("Request Entity Too Large", 413, rid),
                        status_code=413,
                        media_type="application/problem+json",
                    )
            except Exception:
                pass
            body = await request.body()
            if len(body) > self.max_bytes:
                rid = getattr(request.state, "request_id", str(uuid.uuid4()))
                request.state.request_id = rid
                return JSONResponse(
                    problem("Request Entity Too Large", 413, rid),
                    status_code=413,
                    media_type="application/problem+json",
                )
            # allow downstream to reuse without re-reading
            request._body = body
        return await call_next(request)


app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.request_body_max_bytes)
app.add_middleware(RequestIdMiddleware, header_name=settings.request_id_header)


# --- Error handlers (Problem Details) ------------------------------------
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException) -> Response:
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        problem(exc.detail or "HTTP error", exc.status_code, rid),
        status_code=exc.status_code,
        media_type="application/problem+json",
    )


@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception) -> Response:
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.exception("unhandled", request_id=rid)
    return JSONResponse(
        problem("Internal Server Error", 500, rid),
        status_code=500,
        media_type="application/problem+json",
    )


# --- Auth (API key OR JWT if configured) ---------------------------------
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
bearer = HTTPBearer(auto_error=False)
_api_key_dep = Depends(api_key_header)
_bearer_dep = Depends(bearer)


def require_auth(
    key: str | None = _api_key_dep,
    token: HTTPAuthorizationCredentials | None = _bearer_dep,
) -> None:
    if settings.fuzz_mode:
        return
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


def _ready_probe() -> bool:
    """Lightweight readiness check placeholder."""
    return True


class ReadyResponse(BaseModel):
    ready: bool
    version: str
    git_sha: str


# Small extra router: a protected ping + a POST sink for body-limit tests
from fastapi import APIRouter  # noqa: E402

_phase2 = APIRouter()


@_phase2.get(
    "/ready",
    response_model=ReadyResponse,
    responses={
        200: {
            "description": "Service readiness",
            "content": {
                "application/json": {
                    "example": {"ready": True, "version": APP_VERSION, "git_sha": GIT_SHA}
                }
            },
        }
    },
)
def ready() -> ReadyResponse:
    return ReadyResponse(ready=_ready_probe(), version=APP_VERSION, git_sha=GIT_SHA)


@_phase2.get(
    "/secure/ping",
    dependencies=[Depends(require_auth)],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        429: {"description": "Too Many Requests"},
    },
)
def secure_ping() -> dict[str, Any]:
    return {"ok": True, "secure": True}


@_phase2.post(
    "/sink",
    responses={
        413: {"description": "Request Entity Too Large"},
        429: {"description": "Too Many Requests"},
    },
)
def sink_endpoint(_: Any = None) -> dict[str, Any]:
    return {"ok": True}


@_phase2.get("/search")
def search_endpoint(q: str, backend: BackendName = "bm25", k: int = 5) -> dict[str, Any]:
    return rag_search(q, backend, k)


# Mount existing v1 routes + our small Phase 2 endpoints
app.include_router(v1_router, prefix="/api/v1")
app.include_router(_phase2, prefix="/api/v1")


# --- startup hook --------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    configure_logging(settings)
    try:
        from fastapi_app.app.telemetry import init_otel  # local import to avoid E402

        init_otel()
    except Exception:  # pragma: no cover - optional telemetry
        pass
