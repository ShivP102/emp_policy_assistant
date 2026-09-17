from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from src.config.settings import Settings


class HuggingFaceEmbedder:
    """Open-source sentence-transformers embeddings for documents and queries."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    @property
    def embedding_function(self) -> HuggingFaceEmbeddings:
        return self._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)
