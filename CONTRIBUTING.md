# Contributing

1. Create a virtualenv and run `make.ps1 setup` (Windows) or `python -m pip install -r requirements.txt`.
2. Pre-commit: `python -m pre_commit install` (hooks run on commit).
3. Run tests: `.\make.ps1 test` (Windows) or `pytest -q` (any).
4. For changes to RAG, update docs/MODEL_CARD.md and run evals: `python rag/eval_metrics.py`.
