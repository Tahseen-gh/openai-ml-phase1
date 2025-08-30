from __future__ import annotations

from collections.abc import Sequence

from .base import Document, RetrievalBackend


def _min_max(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class HybridBackend(RetrievalBackend):
    def __init__(self, bm25: RetrievalBackend, embed: RetrievalBackend, alpha: float = 0.5) -> None:
        self.bm25 = bm25
        self.embed = embed
        self.alpha = alpha

    def build(self, docs: Sequence[Document], random_seed: int | None = None) -> None:
        self.bm25.build(docs, random_seed)
        self.embed.build(docs, random_seed)

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        bm25_res = self.bm25.search(query, k)
        embed_res = self.embed.search(query, k)
        all_docs = {d.doc_id: d for d, _ in bm25_res + embed_res}
        bm25_scores = {d.doc_id: s for d, s in bm25_res}
        embed_scores = {d.doc_id: s for d, s in embed_res}
        bm25_norm = {
            doc_id: score
            for doc_id, score in zip(
                bm25_scores.keys(), _min_max(list(bm25_scores.values())), strict=False
            )
        }
        embed_norm = {
            doc_id: score
            for doc_id, score in zip(
                embed_scores.keys(), _min_max(list(embed_scores.values())), strict=False
            )
        }
        combined = {}
        for doc_id in all_docs:
            b = bm25_norm.get(doc_id, 0.0)
            e = embed_norm.get(doc_id, 0.0)
            combined[doc_id] = (1 - self.alpha) * b + self.alpha * e

        # tie-break on bm25 then doc_id
        def sort_key(doc_id: str) -> tuple[float, float, str]:
            return (
                combined[doc_id],
                bm25_scores.get(doc_id, 0.0),
                doc_id,
            )

        top = sorted(all_docs.keys(), key=sort_key, reverse=True)[:k]
        return [(all_docs[doc_id], combined[doc_id]) for doc_id in top]

    def save(self, path: str) -> None:  # pragma: no cover - not used
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> HybridBackend:  # pragma: no cover - not used
        raise NotImplementedError
