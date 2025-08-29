from fastapi.testclient import TestClient

from fastapi_app.app.main import app


def test_ready_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/v1/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert "version" in data and "git_sha" in data
