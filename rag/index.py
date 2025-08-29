# Placeholder index that stores chunks in-memory.


class InMemoryIndex:
    def __init__(self):
        self.docs: list[dict] = []

    def add(self, docs: list[dict]):
        self.docs.extend(docs)

    def search(self, query: str, k: int = 5) -> list[dict]:
        # Extremely naive "search": return first k.
        return self.docs[:k]
