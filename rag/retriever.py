from __future__ import annotations

from .backends.base import RetrievalBackend
from .backends.bm25 import BM25Backend
from .backends.embed import DummyEmbeddingModel, EmbeddingBackend, EmbeddingModel
from .backends.hybrid import HybridBackend

# Small in-memory corpus for demo purposes
_CORPUS = [
    ("doc1", "the cat sat on the mat"),
    ("doc2", "dogs are great pets"),
    ("doc3", "I love pizza"),
    ("doc4", "the quick brown fox"),
    ("doc5", "fastapi makes apis fast"),
]

_DOC_IDS = [d for d, _ in _CORPUS]
_DOC_TEXTS = [t for _, t in _CORPUS]
DOCS_BY_ID: dict[str, str] = {d: t for d, t in _CORPUS}

_BACKENDS: dict[str, RetrievalBackend] = {}


def _embedding_model(name: str, use_dummy: bool) -> EmbeddingModel:
    if use_dummy:
        return DummyEmbeddingModel()
    from sentence_transformers import SentenceTransformer

    class _STWrapper:
        def __init__(self, name: str):
            self.model = SentenceTransformer(name)
        def encode_texts(self, texts: list[str]):
            return self.model.encode(texts, show_progress_bar=False)

    return _STWrapper(name)


def get_backend(
    name: str,
    *,
    embedding_model: str,
    hybrid_alpha: float,
    use_dummy_embeddings: bool,
) -> RetrievalBackend:
    if name in _BACKENDS:
        return _BACKENDS[name]
    backend: RetrievalBackend
    if name == "bm25":
        backend = BM25Backend()
        backend.build(_DOC_TEXTS, _DOC_IDS)
    elif name == "embed":
        model = _embedding_model(embedding_model, use_dummy_embeddings)
        backend = EmbeddingBackend(model)
        backend.build(_DOC_TEXTS, _DOC_IDS)
    elif name == "hybrid":
        bm = BM25Backend()
        bm.build(_DOC_TEXTS, _DOC_IDS)
        model = _embedding_model(embedding_model, use_dummy_embeddings)
        em = EmbeddingBackend(model)
        em.build(_DOC_TEXTS, _DOC_IDS)
        backend = HybridBackend(bm, em, alpha=hybrid_alpha)
    else:
        raise ValueError(f"unknown backend: {name}")
    _BACKENDS[name] = backend
    return backend
