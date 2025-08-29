from fastapi.testclient import TestClient
from fastapi_app.app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
