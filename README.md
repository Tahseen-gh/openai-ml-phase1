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
