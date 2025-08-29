from fastapi.testclient import TestClient

from fastapi_app.app.main import app

client = TestClient(app)


def test_health_has_fields() -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "version" in body and "git_sha" in body
