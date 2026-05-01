from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    model_name: str = "llama-3.1-8b-instant"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(
        env_prefix="IRATA_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
