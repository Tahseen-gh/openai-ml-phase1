from __future__ import annotations

import builtins
import importlib

import pytest

from rag.backends.bm25 import BM25Backend
from rag.backends.embed import DummyEmbeddingModel, EmbeddingBackend
from rag.backends.hybrid import HybridBackend


def test_embed_rank_order_deterministic() -> None:
    docs = ["alpha beta", "beta gamma", "delta"]
    ids = ["d1", "d2", "d3"]
    b1 = EmbeddingBackend(DummyEmbeddingModel())
    b1.build(docs, ids, seed=42)
    r1 = b1.search("beta", k=3)
    b2 = EmbeddingBackend(DummyEmbeddingModel())
    b2.build(docs, ids, seed=42)
    r2 = b2.search("beta", k=3)
    assert r1 == r2


def _fusion_manual(
    query: str, docs: list[str], ids: list[str], alpha: float
) -> list[tuple[str, float]]:
    bm = BM25Backend()
    bm.build(docs, ids)
    em = EmbeddingBackend(DummyEmbeddingModel())
    em.build(docs, ids)
    bm_scores = {i: s for i, s in bm.search(query, k=len(docs))}
    em_scores = {i: s for i, s in em.search(query, k=len(docs))}
    ids_order = sorted(bm_scores.keys() | em_scores.keys())

    def minmax(vals: list[float]) -> list[float]:
        lo, hi = min(vals), max(vals)
        if hi - lo == 0:
            return [0.0 for _ in vals]
        return [(v - lo) / (hi - lo) for v in vals]

    bm_norm = minmax([bm_scores.get(i, 0.0) for i in ids_order])
    em_norm = minmax([em_scores.get(i, 0.0) for i in ids_order])
    expected = [
        (
            ids_order[idx],
            (1 - alpha) * bm_norm[idx] + alpha * em_norm[idx],
            bm_scores.get(ids_order[idx], 0.0),
        )
        for idx in range(len(ids_order))
    ]
    expected.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return [(i, s) for i, s, _ in expected]


def test_hybrid_fusion_math() -> None:
    docs = ["cat", "dog", "cat dog"]
    ids = ["c", "d", "cd"]
    manual = _fusion_manual("cat", docs, ids, 0.5)
    bm = BM25Backend()
    bm.build(docs, ids)
    em = EmbeddingBackend(DummyEmbeddingModel())
    em.build(docs, ids)
    hy = HybridBackend(bm, em, alpha=0.5)
    result = hy.search("cat", k=3)
    assert result == manual[:3]


def test_embed_fallback_without_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("no faiss")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    embed_mod = importlib.reload(importlib.import_module("rag.backends.embed"))
    backend = embed_mod.EmbeddingBackend(embed_mod.DummyEmbeddingModel())
    backend.build(["a", "b"], ["a", "b"])
    assert backend.search("a", k=1)
