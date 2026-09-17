from __future__ import annotations

from langchain_core.documents import Document

from src.config.settings import Settings
from src.ingestion.embedder import HuggingFaceEmbedder
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import VectorRetriever
from src.vector_store.chroma_store import ChromaStore


class RetrievalPipeline:
    """Two-stage retrieval: vector search then cross-encoder reranking."""

    def __init__(
        self,
        settings: Settings,
        embedder: HuggingFaceEmbedder,
        store: ChromaStore,
    ) -> None:
        self.settings = settings
        self.retriever = VectorRetriever(settings, embedder, store)
        self.reranker = CrossEncoderReranker(settings)

    def retrieve(self, query: str) -> list[Document]:
        candidates = self.retriever.retrieve(query, k=self.settings.top_k_retrieve)
        return self.reranker.rerank(query, candidates, top_k=self.settings.top_k_rerank)
