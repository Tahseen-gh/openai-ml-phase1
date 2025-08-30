from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag.retriever import search


def recall_at_k(rel: set[str], res: Sequence[str], k: int) -> float:
    return len(rel.intersection(res[:k])) / len(rel) if rel else 0.0


def mrr(rel: set[str], res: Sequence[str]) -> float:
    for i, doc_id in enumerate(res, start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel: set[str], res: Sequence[str], k: int) -> float:
    import math

    dcg = 0.0
    for i, doc_id in enumerate(res[:k], start=1):
        if doc_id in rel:
            dcg += 1.0 / math.log2(i + 1)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["bm25", "embed", "hybrid"], default="bm25")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    queries = []
    with open("data/queries.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            queries.append(obj)

    metrics = {"recall@k": 0.0, "mrr": 0.0, "ndcg@10": 0.0}
    backend = args.backend

    for q in queries:
        res = search(q["q"], backend, args.k)["results"]
        ids = [r["doc_id"] for r in res]
        rel = set(q["relevant_ids"])
        metrics["recall@k"] += recall_at_k(rel, ids, args.k)
        metrics["mrr"] += mrr(rel, ids)
        metrics["ndcg@10"] += ndcg_at_k(rel, ids, 10)

    n = len(queries)
    for k in metrics:
        metrics[k] /= n

    Path("evals").mkdir(exist_ok=True)
    out = {"backend": backend, "k": args.k, "metrics": metrics}
    with open("evals/latest.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
