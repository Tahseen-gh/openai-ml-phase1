from __future__ import annotations

import pickle
import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from .base import Document, RetrievalBackend

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


class BM25Backend(RetrievalBackend):
    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._bm25: BM25Okapi | None = None

    def build(self, docs: Sequence[Document], random_seed: int | None = None) -> None:
        self._docs = list(docs)
        corpus = [_tokenize(d.text) for d in self._docs] or [[""]]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        if self._bm25 is None:
            raise RuntimeError("index not built")
        toks = _tokenize(query)
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._docs[i], float(scores[i])) for i in order]

    def save(self, path: str) -> None:
        data = {"docs": self._docs, "bm25": self._bm25}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> BM25Backend:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._docs = data["docs"]
        obj._bm25 = data["bm25"]
        return obj
