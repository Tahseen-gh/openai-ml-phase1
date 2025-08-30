from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .base import RetrievalBackend

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


class BM25Backend(RetrievalBackend):
    """Simple BM25 implementation over in-memory documents."""

    def __init__(self) -> None:
        self._docs: list[str] = []
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build(
        self, docs: list[str], ids: list[str] | None = None, *, seed: int | None = None
    ) -> None:
        self._docs = list(docs)
        self._ids = ids or [str(i) for i in range(len(docs))]
        tok_corpus = [_tokenize(d) for d in self._docs]
        self._bm25 = BM25Okapi(tok_corpus if tok_corpus else [[""]])

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build() first.")
        toks = _tokenize(query)
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: (scores[i], -i), reverse=True)[:k]
        return [(self._ids[i], float(scores[i])) for i in order]
