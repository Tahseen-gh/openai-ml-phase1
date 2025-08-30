from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # App
    app_name: str = "phase1-api"
    debug: bool = False

    # Auth
    api_key: str | None = None  # header: X-API-KEY
    jwt_secret: str | None = None
    jwt_algorithm: str = "HS256"  # used if jwt_secret is set

    # CORS
    cors_origins: list[str] = Field(default_factory=list)

    # Protection
    rate_limit_qps: float = 5.0
    request_body_max_bytes: int = 100_000

    # Observability
    metrics_enabled: bool = True

    # Retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hybrid_alpha: float = 0.5
    use_dummy_embeddings: bool = True


settings = Settings()
