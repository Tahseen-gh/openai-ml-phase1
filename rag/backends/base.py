from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str | None = None


class RetrievalBackend(Protocol):
    def build(self, docs: Sequence[Document], random_seed: int | None = None) -> None: ...

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> RetrievalBackend: ...
