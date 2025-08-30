from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastapi_app.app.api.v1 import router


def test_router_health_endpoint() -> None:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
