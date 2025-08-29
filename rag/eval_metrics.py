from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from .bm25_index import BM25ChunkIndex
    from .chunking import Chunk, chunk_text
except Exception:  # pragma: no cover
    from rag.bm25_index import BM25ChunkIndex
    from rag.chunking import Chunk, chunk_text
ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "sample_corpus"
FIXTURES = ROOT / "fixtures.json"
REPORT = ROOT / "report.json"
for d in (ROOT / "sample_corpus", ROOT / "eval" / "sample_corpus"):
    if d.exists():
        CORPUS_DIR = d
        break
for f in (ROOT / "fixtures.json", ROOT / "eval" / "fixtures.json"):
    if f.exists():
        FIXTURES = f
        break


def _load_corpus() -> list[Chunk]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    for p in sorted(CORPUS_DIR.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        chunks.extend(chunk_text(p.stem, text, max_chars=500, overlap=80))
    return chunks


def _load_queries() -> list[tuple[str, str]]:
    if FIXTURES.exists():
        data = json.loads(FIXTURES.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and isinstance(data.get("queries"), list):
            qs: list[tuple[str, str]] = []
            for obj in data["queries"]:
                if isinstance(obj, dict):
                    q = obj.get("query") or obj.get("q") or obj.get("text")
                    rel = (
                        obj.get("relevant")
                        or obj.get("doc_id")
                        or obj.get("expected")
                        or obj.get("label")
                    )
                    if isinstance(q, str) and isinstance(rel, str):
                        qs.append((q, rel))
            if qs:
                return qs
        if isinstance(data, dict):
            items = [(k, v) for k, v in data.items() if isinstance(k, str) and isinstance(v, str)]
            if items:
                return items
    stems = [p.stem for p in sorted(CORPUS_DIR.glob("*.txt"))]
    return [(s, s) for s in stems]


def _doc_ids(results: Sequence[Any]) -> list[str]:
    ids: list[str] = []
    for r in results:
        if isinstance(r, tuple) and r:
            c = r[0]
            ids.append(getattr(c, "doc_id", ""))
        else:
            c = getattr(r, "chunk", r)
            ids.append(getattr(c, "doc_id", ""))
    return ids


def main() -> None:
    chunks = _load_corpus()
    if not chunks:
        print(f"[eval] No corpus files found in {CORPUS_DIR}. Add *.txt files and try again.")
        return
    bm25_cls: Any = BM25ChunkIndex
    try:
        index = bm25_cls()
        if hasattr(index, "build"):
            index.build(chunks)
        else:
            fc = getattr(bm25_cls, "from_chunks", None)
            index = fc(chunks) if callable(fc) else bm25_cls(chunks)
    except TypeError:
        fc = getattr(bm25_cls, "from_chunks", None)
        index = fc(chunks) if callable(fc) else bm25_cls(chunks)
    queries = _load_queries()
    if not queries:
        print("[eval] No queries found; nothing to evaluate.")
        return
    k = 3
    total = len(queries)
    recall_hits = 0
    rr_sum = 0.0
    ndcg_sum = 0.0
    for q, rel in queries:
        ids = _doc_ids(index.search(q, k=k))
        try:
            r = ids.index(rel)
        except ValueError:
            r = -1
        if 0 <= r < k:
            recall_hits += 1
            rr_sum += 1.0 / (r + 1)
            ndcg_sum += 1.0 / math.log2(r + 2)
    metrics = {
        "recall@3": round(recall_hits / total, 4),
        "mrr": round(rr_sum / total, 4),
        "ndcg": round(ndcg_sum / total, 4),
        "count": total,
        "k": k,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[eval] wrote {REPORT}:\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
