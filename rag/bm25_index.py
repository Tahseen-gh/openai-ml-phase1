from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from rag.chunking import Chunk

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def _tokenize(s: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(s)]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class BM25ChunkIndex:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._tok_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tok_corpus = [_tokenize(c.text) for c in self._chunks]
        corpus = self._tok_corpus if self._tok_corpus else [[""]]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build() first.")
        toks = _tokenize(query)
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [ScoredChunk(self._chunks[i], float(scores[i])) for i in order]
