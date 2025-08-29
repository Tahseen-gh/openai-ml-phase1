from _rag_smoke_utils import _smoke_module


def test_smoke_rag_index() -> None:
    _smoke_module("rag.index")


def test_smoke_rag_retrieve() -> None:
    _smoke_module("rag.retrieve")


def test_smoke_rag_ingestion() -> None:
    _smoke_module("rag.ingestion")
