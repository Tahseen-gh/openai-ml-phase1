from __future__ import annotations

import pickle
import warnings
from collections.abc import Sequence

import numpy as np

from .base import Document, RetrievalBackend

try:  # optional dependency
    import faiss
except Exception:  # pragma: no cover - fallback
    faiss = None


class EmbeddingModel:
    """Tiny interface for embedding models."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - Protocol
        raise NotImplementedError


class SentenceTransformerModel(EmbeddingModel):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))


class DummyEmbeddingModel(EmbeddingModel):
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            vecs.append(rng.random(self.dim))
        return np.vstack(vecs)


class EmbeddingBackend(RetrievalBackend):
    def __init__(self, model: EmbeddingModel) -> None:
        self.model = model
        self._docs: list[Document] = []
        self._emb: np.ndarray | None = None
        self._index: faiss.IndexFlatIP | None = None

    def build(self, docs: Sequence[Document], random_seed: int | None = None) -> None:
        self._docs = list(docs)
        self._emb = self.model.embed([d.text for d in self._docs])
        if faiss is not None:
            index = faiss.IndexFlatIP(self._emb.shape[1])
            index.add(self._emb.astype(np.float32))
            self._index = index
        else:
            warnings.warn("faiss not available; using brute-force cosine", stacklevel=2)

    def _similarities(self, q: np.ndarray) -> np.ndarray:
        if self._emb is None:
            raise RuntimeError("index not built")
        if self._index is not None:
            q = q.astype(np.float32)
            scores, idx = self._index.search(q[None, :], len(self._docs))
            return scores[0], idx[0]
        # brute force cosine
        emb = self._emb
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        docs_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sims = docs_norm @ q_norm
        order = np.argsort(-sims)
        return sims[order], order

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        q_emb = self.model.embed([query])[0]
        sims, order = self._similarities(q_emb)
        order = order[:k]
        return [(self._docs[i], float(sims[i])) for i in order]

    def save(self, path: str) -> None:
        data = {"docs": self._docs, "emb": self._emb}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> EmbeddingBackend:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(DummyEmbeddingModel())
        obj._docs = data["docs"]
        obj._emb = data["emb"]
        return obj
