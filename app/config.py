from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    cohere_api_key: str = ""
    cohere_chat_model: str = "command-a-03-2025"
    cohere_embedding_model: str = "embed-v4.0"
    database_url: str = "sqlite:///./palm_mind.db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "palm_mind_documents"
    redis_url: str = "redis://localhost:6379/0"
    chat_history_turns: int = Field(default=12, ge=1)
    chat_memory_ttl_seconds: int = Field(default=86400, ge=1)
    max_upload_size_bytes: int = Field(default=10 * 1024 * 1024, ge=1)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
