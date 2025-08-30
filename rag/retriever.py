from __future__ import annotations

import os
from functools import cache, lru_cache
from pathlib import Path
from typing import Literal

from .backends.base import Document, RetrievalBackend
from .backends.bm25 import BM25Backend
from .backends.embed import (
    DummyEmbeddingModel,
    EmbeddingBackend,
    SentenceTransformerModel,
)
from .backends.hybrid import HybridBackend

BackendName = Literal["bm25", "embed", "hybrid"]


@lru_cache(maxsize=1)
def _load_corpus() -> list[Document]:
    corpus_dir = Path("data/corpus")
    docs = []
    for p in sorted(corpus_dir.glob("*.txt")):
        docs.append(Document(doc_id=p.stem, text=p.read_text(), source=str(p)))
    return docs


@lru_cache(maxsize=1)
def _embed_model() -> DummyEmbeddingModel | SentenceTransformerModel:
    if os.getenv("USE_DUMMY_EMBEDDINGS", "false").lower() == "true":
        return DummyEmbeddingModel()
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformerModel(model_name)


@cache
def get_backend(name: BackendName) -> RetrievalBackend:
    docs = _load_corpus()
    if name == "bm25":
        backend: RetrievalBackend = BM25Backend()
    elif name == "embed":
        backend = EmbeddingBackend(_embed_model())
    elif name == "hybrid":
        alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))
        backend = HybridBackend(BM25Backend(), EmbeddingBackend(_embed_model()), alpha)
    else:  # pragma: no cover
        raise ValueError(f"unknown backend {name}")
    backend.build(docs, random_seed=1337)
    return backend


def search(query: str, backend: BackendName = "bm25", k: int = 5) -> dict:
    b = get_backend(backend)
    results = b.search(query, k)
    return {
        "query": query,
        "backend": backend,
        "results": [
            {
                "doc_id": d.doc_id,
                "chunk": d.text,
                "score": s,
                "source": d.source,
            }
            for d, s in results
        ],
    }
