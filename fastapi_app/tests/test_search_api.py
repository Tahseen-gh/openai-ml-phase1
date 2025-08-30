from __future__ import annotations

import os

os.environ["USE_DUMMY_EMBEDDINGS"] = "true"

from fastapi.testclient import TestClient

from fastapi_app.app.main import app


def test_api_search_backend_switch() -> None:
    client = TestClient(app)
    for backend in ["bm25", "embed", "hybrid"]:
        r = client.get("/api/v1/search", params={"q": "cat", "backend": backend, "k": 3})
        assert r.status_code == 200
        data = r.json()
        assert data["backend"] == backend
        assert len(data["results"]) == 3
        for item in data["results"]:
            assert "doc_id" in item and "score" in item
