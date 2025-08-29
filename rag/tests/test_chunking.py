from rag.ingestion import simple_chunk


def test_simple_chunk_basic():
    txt = "a" * 1000
    chunks = simple_chunk(txt, chunk_size=200, overlap=50)
    # Expect size > 1 and each chunk length <= 200
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
