from rag.index import InMemoryIndex
from rag.retrieve import Retriever


def test_retriever_returns_k_results():
    docs = [
        {"id": "1", "text": "one"},
        {"id": "2", "text": "two"},
        {"id": "3", "text": "three"},
    ]
    index = InMemoryIndex()
    index.add(docs)
    retriever = Retriever(index)
    results = retriever.retrieve("anything", k=2)
    assert results == docs[:2]
