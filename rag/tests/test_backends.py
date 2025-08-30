from __future__ import annotations

from rag.backends.base import Document
from rag.backends.bm25 import BM25Backend
from rag.backends.embed import DummyEmbeddingModel, EmbeddingBackend
from rag.backends.hybrid import HybridBackend


def test_embed_backend_rank_order_deterministic() -> None:
    docs = [Document("a", "foo"), Document("b", "bar")]
    backend1 = EmbeddingBackend(DummyEmbeddingModel())
    backend2 = EmbeddingBackend(DummyEmbeddingModel())
    backend1.build(docs, random_seed=1)
    backend2.build(docs, random_seed=1)
    r1 = [d.doc_id for d, _ in backend1.search("foo", k=2)]
    r2 = [d.doc_id for d, _ in backend2.search("foo", k=2)]
    assert r1 == r2


def test_hybrid_beats_or_matches_bm25_on_toy() -> None:
    model = DummyEmbeddingModel()
    docs = [Document("a", "alpha"), Document("b", "beta")]
    bm25 = BM25Backend()
    bm25.build(docs)
    embed = EmbeddingBackend(model)
    embed.build(docs)
    # Determine which doc is closest in embedding space to query
    q = "target"
    q_emb = model.embed([q])[0]
    sims = model.embed([d.text for d in docs]) @ q_emb
    best_doc = docs[int(sims.argmax())].doc_id
    bm25_top = bm25.search(q, k=1)[0][0].doc_id
    hybrid = HybridBackend(bm25, embed, alpha=1.0)
    hybrid_top = hybrid.search(q, k=1)[0][0].doc_id
    assert hybrid_top == best_doc and bm25_top != best_doc
