from rag.chunking import chunk_text


def test_chunking_headings_and_overlap():
    text = (
        "# Intro\n"
        "This is the first paragraph. It sets context.\n\n"
        "Still intro with more detail! Short.\n\n"
        "## Details\n" + ("Alpha beta gamma. " * 80) + "\n\nConclusion sentence."
    )
    chunks = chunk_text("doc1", text, max_chars=500, overlap=80)
    assert all(len(c.text) <= 500 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text[-80:]
        assert tail[:40] in b.text
    assert chunks[0].heading == "Intro"
    assert any(c.heading == "Details" for c in chunks)
