"""Deterministic retrieval evaluation script."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

# ensure repo root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from fastapi_app.app.config import settings
from rag.retriever import get_backend

METRICS_KS = [1, 3, 5, 10]


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def load_corpus(manifest_path: Path) -> tuple[list[str], list[str]]:
    manifest = json.loads(manifest_path.read_text())
    docs_dir = manifest_path.parent / "corpus"
    texts, ids = [], []
    for fname in sorted(manifest["docs"].keys()):
        path = docs_dir / fname
        texts.append(path.read_text().strip())
        ids.append(Path(fname).stem)
    return texts, ids


def load_queries(path: Path) -> list[dict[str, object]]:
    queries: list[dict[str, object]] = []
    with path.open() as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def evaluate(backend_name: str, k: int, seed: int, manifest_path: Path) -> dict[str, object]:
    set_deterministic(seed)
    texts, ids = load_corpus(manifest_path)
    backend = get_backend(
        backend_name,
        embedding_model=settings.embedding_model,
        hybrid_alpha=settings.hybrid_alpha,
        use_dummy_embeddings=settings.use_dummy_embeddings,
    )
    backend.build(texts, ids, seed=seed)
    queries = load_queries(manifest_path.parent / "queries.jsonl")

    recall_counts = {m: 0 for m in METRICS_KS}
    mrr_total = 0.0
    ndcg_total = 0.0

    for q in queries:
        query_text = str(q["q"])
        rel_ids = cast(Sequence[Any], q["relevant_ids"])
        relevant = [str(r) for r in rel_ids]
        results = backend.search(query_text, k=k)
        retrieved_ids = [doc_id for doc_id, _ in results]

        for m in METRICS_KS:
            if any(r in retrieved_ids[:m] for r in relevant):
                recall_counts[m] += 1

        rank = next((i + 1 for i, rid in enumerate(retrieved_ids) if rid in relevant), None)
        if rank is not None:
            mrr_total += 1 / rank

        dcg = 0.0
        for i, rid in enumerate(retrieved_ids[:10], start=1):
            if rid in relevant:
                dcg += 1 / np.log2(i + 1)
        ideal_hits = min(len(relevant), 10)
        idcg = sum(1 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
        if idcg > 0:
            ndcg_total += dcg / idcg

    n = len(queries)
    metrics = {f"recall@{m}": recall_counts[m] / n for m in METRICS_KS}
    metrics["MRR"] = mrr_total / n
    metrics["NDCG@10"] = ndcg_total / n

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    manifest = json.loads(manifest_path.read_text())
    report = {
        "git_sha": git_sha,
        "backend": backend_name,
        "seed": seed,
        "manifest_version": manifest["version"],
        "metrics": metrics,
        "stats": {"queries": n, "docs": manifest["docs_count"]},
        "versions": {"numpy": np.__version__},
    }
    reports_dir = Path("evals/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"{backend_name}-{git_sha}.json"
    out_path.write_text(json.dumps(report, indent=2))
    Path("evals/latest.json").write_text(json.dumps(report, indent=2))
    from rag import retriever as retriever_mod

    retriever_mod._BACKENDS.pop(backend_name, None)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--backend", choices=["bm25", "embed", "hybrid"], required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    args = parser.parse_args()
    evaluate(args.backend, args.k, args.seed, args.manifest)


if __name__ == "__main__":
    main()
