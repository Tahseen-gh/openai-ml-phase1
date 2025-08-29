from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pydantic v2 settings config
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    app_name: str = "phase1-api"
    debug: bool = False
    api_key: Optional[str] = None
    cors_origins: List[str] = Field(default_factory=list)


settings = Settings()
