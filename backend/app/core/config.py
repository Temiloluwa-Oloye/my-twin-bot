from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq / LLM
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model_name: str = Field(
        default="llama-3.3-70b-versatile",
        env="GROQ_MODEL_NAME",
    )

    # Embeddings / Vector store
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL_NAME",
    )
    chroma_db_dir: str = Field(
        default=str(PROJECT_ROOT / "app" / "data" / "chroma_db"),
        env="CHROMA_DB_DIR",
    )
    chroma_collection: str = Field(
        default="twin_knowledge_base",
        env="CHROMA_COLLECTION",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

