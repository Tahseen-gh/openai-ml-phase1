from rag.chunking import chunk_text
from rag.bm25_index import BM25ChunkIndex


def test_bm25_ranks_relevant_higher():
    doc_a = "Cats purr softly. Felines are wonderful companions."
    doc_b = "Satellites orbit Earth. Space is vast and cold."
    chunks_a = chunk_text("A", doc_a, max_chars=200, overlap=0)
    chunks_b = chunk_text("B", doc_b, max_chars=200, overlap=0)
    idx = BM25ChunkIndex()
    idx.build(chunks_a + chunks_b)
    top = idx.search("felines purr", k=1)[0]
    assert top.chunk.doc_id == "A"


def test_empty_build_search_returns_empty_list():
    idx = BM25ChunkIndex()
    idx.build([])
    assert idx.search("anything") == []
