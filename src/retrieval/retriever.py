from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from src.config.settings import Settings
from src.ingestion.embedder import HuggingFaceEmbedder
from src.vector_store.chroma_store import ChromaStore


@dataclass
class ScoredDocument:
    document: Document
    score: float


class VectorRetriever:
    """Retrieve top-k documents using cosine similarity on embeddings."""

    def __init__(
        self,
        settings: Settings,
        embedder: HuggingFaceEmbedder,
        store: ChromaStore,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredDocument]:
        top_k = k or self.settings.top_k_retrieve
        query_embedding = self.embedder.embed_query(query)
        results = self.store.similarity_search_by_vector(query_embedding, k=top_k)
        return [
            ScoredDocument(document=document, score=score)
            for document, score in results
        ]
