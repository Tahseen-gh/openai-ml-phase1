from __future__ import annotations

from fastapi.testclient import TestClient

from fastapi_app.app import config
from fastapi_app.app.main import app


def test_api_search_backends() -> None:
    config.settings.use_dummy_embeddings = True
    client = TestClient(app)
    for name in ["bm25", "embed", "hybrid"]:
        r = client.get("/api/v1/search", params={"q": "fast", "backend": name, "k": 2})
        assert r.status_code == 200
        data = r.json()
        assert data["backend"] == name
        assert len(data["results"]) == 2
        assert {"doc_id", "score", "text"} <= data["results"][0].keys()
