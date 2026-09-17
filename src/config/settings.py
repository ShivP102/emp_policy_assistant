from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pdf_dir: Path = Field(default=PROJECT_ROOT / "data" / "company_policies")
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "chroma_db")
    collection_name: str = "company_policies"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    openai_model: str = "gpt-4o-mini"

    top_k_retrieve: int = 10
    top_k_rerank: int = 3
    chunk_max_chars: int = 1000
    chunk_min_chars: int = 500
    chunk_overlap: int = 100
    chunk_config_version: str = "v1"

    openai_api_key: str = ""

    @property
    def manifest_path(self) -> Path:
        return self.chroma_persist_dir / "ingest_manifest.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
