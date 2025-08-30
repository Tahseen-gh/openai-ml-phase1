from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol

import numpy as np

from .base import RetrievalBackend

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Minimal embedding interface."""

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (n, d)."""


class DummyEmbeddingModel:
    """Deterministic hash-based embeddings used for tests."""

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        vecs: list[np.ndarray] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = np.frombuffer(h[:32], dtype=np.uint8).astype(np.float32)
            vecs.append(vec)
        arr = np.vstack(vecs)
        return _normalize(arr)


try:  # pragma: no cover - exercised in tests
    import faiss
except Exception:  # pragma: no cover - module may be absent
    faiss = None
    logger.warning("faiss not available; using numpy cosine search")


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class EmbeddingBackend(RetrievalBackend):
    def __init__(self, model: EmbeddingModel) -> None:
        self.model = model
        self._ids: list[str] = []
        self._vecs: np.ndarray | None = None
        self._index: Any | None = None

    def build(
        self, docs: list[str], ids: list[str] | None = None, *, seed: int | None = None
    ) -> None:
        self._ids = ids or [str(i) for i in range(len(docs))]
        vecs = self.model.encode_texts(docs)
        vecs = _normalize(vecs.astype(np.float32))
        if faiss is not None:
            self._index = faiss.IndexFlatIP(vecs.shape[1])
            self._index.add(vecs)
        else:
            self._vecs = vecs

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = self.model.encode_texts([query])[0]
        q = _normalize(q.reshape(1, -1)).astype(np.float32)
        if self._index is not None:
            scores, idxs = self._index.search(q, k)
            scores = scores[0]
            idxs = idxs[0]
        elif self._vecs is not None:
            scores = (self._vecs @ q.T).ravel()
            idxs = np.argsort(-scores)[:k]
        else:
            raise RuntimeError("Index not built. Call build() first.")
        return [(self._ids[int(i)], float(scores[int(n)])) for n, i in enumerate(idxs)]
