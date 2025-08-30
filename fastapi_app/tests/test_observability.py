import uuid

from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from fastapi_app.app.config import settings
from fastapi_app.app.main import app


def test_request_id_roundtrip() -> None:
    with TestClient(app) as client:
        rid = "abc"
        res = client.get("/api/v1/ready", headers={settings.request_id_header: rid})
        assert res.headers[settings.request_id_header] == rid
        res = client.get("/api/v1/ready")
        generated = res.headers[settings.request_id_header]
        uuid.UUID(generated)


def test_access_log_includes_request_id() -> None:
    with TestClient(app) as client, capture_logs() as logs:
        rid = "xyz"
        client.get("/api/v1/ready", headers={settings.request_id_header: rid})
        assert any(entry.get("request_id") == rid and entry.get("status") == 200 for entry in logs)


def test_problem_details_carries_request_id() -> None:
    with TestClient(app) as client:
        payload = "x" * (settings.request_body_max_bytes + 1)
        res = client.post("/api/v1/sink", data=payload)
        assert res.status_code == 413
        rid = res.headers[settings.request_id_header]
        body = res.json()
        assert body["request_id"] == rid
