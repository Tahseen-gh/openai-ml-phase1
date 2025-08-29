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

## Security

- **Bandit** runs in pre-commit and CI. Any MEDIUM or HIGH findings fail the build; suppress only with a justified `# nosec` comment.
- **pip-audit** runs in CI to report vulnerable dependencies (currently non-blocking).
- **Dependabot** opens weekly grouped PRs for runtime libraries, dev tools, and GitHub Actions with `dependencies` and `security` labels.

## Operations

- **Readiness**: `GET /api/v1/ready` returns `{ "ready": true, "version": "<v>", "git_sha": "<sha>" }`.
- **Correlation IDs**: requests accept and echo an `X-Request-ID` header (configurable) and include `request_id` in Problem Details.
- **JSON logs**: one line per request, e.g.:

```json
{"ts":"2024-01-01T00:00:00Z","level":"info","logger":"access","request_id":"...","method":"GET","path":"/api/v1/ready","status":200,"duration_ms":1.2,"client_ip":"127.0.0.1","user_agent":"curl"}
```
