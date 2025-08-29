from pathlib import Path


def simple_chunk(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        overlap = 0
    # Prevent infinite loops & ensure forward progress
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    chunks: list[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end >= n:
            break  # don't loop on the last window
        start = end - overlap
    return chunks


def ingest_files(paths: list[Path]) -> list[dict]:
    docs: list[dict] = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for i, c in enumerate(simple_chunk(text)):
            docs.append({"id": f"{p.name}-{i}", "text": c, "source": str(p)})
    return docs
