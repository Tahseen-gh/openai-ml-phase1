from fastapi.testclient import TestClient

from fastapi_app.app.config import settings
from fastapi_app.app.main import app

client = TestClient(app)


def test_metrics_exposed() -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    # Either default python metrics or instrumented HTTP metrics should appear
    assert "python_info" in r.text or "http_requests" in r.text


def test_body_size_limit_413() -> None:
    # push body beyond limit to trigger middleware
    big = "x" * (settings.request_body_max_bytes + 10)
    r = client.post("/api/v1/sink", json={"blob": big})
    assert r.status_code == 413


def test_secure_ping_without_auth_when_not_configured() -> None:
    # By default, no API key or JWT is configured -> route should be accessible
    r = client.get("/api/v1/secure/ping")
    assert r.status_code == 200
    assert r.json()["secure"] is True
