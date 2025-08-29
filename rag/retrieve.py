from .index import InMemoryIndex


class Retriever:
    def __init__(self, index: InMemoryIndex):
        self.index = index

    def retrieve(self, query: str, k: int = 5):
        return self.index.search(query, k=k)
