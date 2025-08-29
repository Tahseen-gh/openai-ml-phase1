from fastapi import FastAPI
from .api.v1 import router as v1_router


def create_app() -> FastAPI:
    app = FastAPI(title="Domain RAG & Evals", version="0.1.0")
    app.include_router(v1_router, prefix="/api/v1")
    return app


app = create_app()
