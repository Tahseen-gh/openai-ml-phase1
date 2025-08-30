from .index import InMemoryIndex


class Retriever:
    def __init__(self, index: InMemoryIndex):
        self.index = index

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        return self.index.search(query, k=k)
