# Placeholder index that stores chunks in-memory.
from typing import List, Dict


class InMemoryIndex:
    def __init__(self):
        self.docs: List[Dict] = []

    def add(self, docs: List[Dict]):
        self.docs.extend(docs)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        # Extremely naive "search": return first k.
        return self.docs[:k]
