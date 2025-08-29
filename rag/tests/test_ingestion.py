from pathlib import Path
import pytest

from rag.ingestion import simple_chunk, ingest_files


def test_simple_chunk_variants(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        simple_chunk("text", chunk_size=0)

    # negative overlap treated as 0
    assert simple_chunk("abcdef", chunk_size=3, overlap=-2) == ["abc", "def"]

    # overlap larger than chunk_size is capped
    chunks = simple_chunk("abcdefgh", chunk_size=4, overlap=10)
    assert chunks[0] == "abcd" and chunks[1].startswith("b")

    file = tmp_path / "doc.txt"
    file.write_text("hello world", encoding="utf-8")
    docs = ingest_files([file])
    assert docs[0]["id"].startswith("doc.txt-0")
    assert docs[0]["source"] == str(file)
