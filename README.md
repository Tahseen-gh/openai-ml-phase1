# Domain RAG & Evals Phase 1 Scaffold

This repo boots a production-leaning LLM/RAG project with:
- **FastAPI** service (`/api/v1/health` endpoint)
- Project layout: `fastapi_app/`, `rag/`, `evals/`, `infra/`
- Tooling: Docker, docker-compose, Makefile, pre-commit (black/ruff), GitHub Actions CI
- Tests: pytest (healthcheck)

## Quickstart
```bash
make install        # create venv + install deps
make dev            # run FastAPI with reload (http://127.0.0.1:8000/api/v1/health)
make test           # run tests
make lint           # ruff
make format         # black + ruff --fix
make up             # docker compose up (prod-like)
```

## Structure
```
fastapi_app/      # API app (FastAPI)
rag/              # (Phase 2+) ingestion, indexing, retrieval
evals/            # (Phase 3+) eval datasets & runners
infra/            # docker-compose and ops bits
ml/               # minimal training loop examples & tests
```

> Next Phases will fill out `rag/` (hybrid retrieval), `evals/` (OpenAI Evals), and safety+observability.


### Health

```bash
curl -s http://127.0.0.1:8000/api/v1/health
# or, if API key enabled:
# export API_KEY=change-me
# curl -H "X-API-KEY: $API_KEY" http://127.0.0.1:8000/api/v1/health
```


### Quickstart

```bash
# 1) Create and activate venv (Windows PowerShell)
python -m venv .venv
. .venv/Scripts/Activate.ps1

# 2) Install
python -m pip install -r requirements.txt -r requirements-dev.txt

# 3) Run API (hot reload for dev)
uvicorn fastapi_app.app.main:app --reload

# 4) Health
curl -s http://127.0.0.1:8000/api/v1/health
# If API key is enabled:
# export API_KEY=change-me
# curl -H "X-API-KEY: $API_KEY" http://127.0.0.1:8000/api/v1/health

# 5) Tests (coverage gate from pytest.ini)
pytest -q

# 6) Pre-commit hooks
pre-commit run --all-files
```

## Retrieval backends

Query the demo corpus via `/api/v1/search?q=...&backend=bm25|embed|hybrid&k=5`.
The hybrid backend combines normalized BM25 and embedding scores:

```
score = (1 - α) * bm25_norm + α * embed_norm
```

Example (toy numbers):

| backend | doc | score |
|---------|-----|-------|
| bm25    | doc1 | 0.82 |
| embed   | doc2 | 0.87 |
| hybrid  | doc2 | 0.90 |

Environment knobs:

- `EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`)
- `HYBRID_ALPHA` (weight for embeddings, default `0.5`)
- `USE_DUMMY_EMBEDDINGS` (set `true` to avoid network calls in tests)

## Evaluation & CI gates

Retrieval quality is measured on a versioned corpus with Recall@1/3/5/10, MRR, and NDCG@10.
The `Retrieval evals (quality gates)` workflow runs deterministic evaluations for
`bm25`, `embed`, and `hybrid` backends and fails if any of `recall@3`, `MRR`, or
`NDCG@10` drop by more than `ALLOWED_REGRESSION_DELTA` (default `0.01`) relative to
`evals/baseline.json`.

Example baseline metrics:

| backend | recall@3 | MRR | NDCG@10 |
|---------|----------|-----|---------|
| bm25    | 1.000    | 1.000 | 1.000 |
| embed   | 1.000    | 1.000 | 1.000 |
| hybrid  | 1.000    | 1.000 | 1.000 |

To intentionally update the baseline:

```bash
python scripts/eval_retrieval.py --backend hybrid --seed 1337
python scripts/update_baseline.py --from evals/reports/hybrid-<git_sha>.json
```

The dataset is pinned via `data/manifest.json`; hashes are checked in tests for determinism.
