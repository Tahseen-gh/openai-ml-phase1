from __future__ import annotations

from typing import Any, Awaitable, Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response

import json
import logging
import os
import time
import uuid

from .config import settings
from .api.v1 import router as v1_router

# --- logging --------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")


def jsonlog(message: str, **fields: Any) -> None:
    logger.info(json.dumps({"message": message, **fields}))


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


# request timing + request-id
@app.middleware("http")
async def add_request_id_and_timing(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    rid = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    dur_ms = (time.perf_counter() - start) * 1000
    jsonlog(
        "request",
        request_id=rid,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(dur_ms, 2),
    )
    return response


# simple API key dependency; leave API_KEY blank in .env to disable
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


def require_api_key(key: str | None = Depends(api_key_header)) -> None:
    cfg = settings.api_key
    if cfg and key != cfg:
        raise HTTPException(status_code=401, detail="Unauthorized")


APP_VERSION = "0.1.0"
GIT_SHA = os.getenv("GIT_SHA", "dev")


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": APP_VERSION, "git_sha": GIT_SHA}


# mount v1 routes (protect sensitive ones later with: dependencies=[Depends(require_api_key)])
app.include_router(v1_router, prefix="/api/v1")
