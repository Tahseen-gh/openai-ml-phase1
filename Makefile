.PHONY: install dev run test lint format precommit docker-build up down

PY=python3

install:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt

dev:
	. .venv/bin/activate && uvicorn fastapi_app.app.main:app --reload

run:
	uvicorn fastapi_app.app.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check .

format:
	black . && ruff check . --fix

precommit:
	pre-commit install

docker-build:
	docker build -t rag-evals-api:latest .

up:
	docker compose up -d --build

down:
	docker compose down

typecheck:
	python -m mypy fastapi_app rag ml

coverage:
	python -m pytest

audit:
	python -m pip_audit -r requirements.txt
