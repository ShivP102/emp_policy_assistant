from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config.settings import Settings
from src.ingestion.embedder import HuggingFaceEmbedder


class ChromaStore:
    """Persistent ChromaDB vector store wrapper."""

    def __init__(self, settings: Settings, embedder: HuggingFaceEmbedder) -> None:
        self.settings = settings
        self.embedder = embedder
        self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embedder.embedding_function,
            persist_directory=str(settings.chroma_persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    @property
    def vectorstore(self) -> Chroma:
        return self._vectorstore

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self._vectorstore.add_documents(documents)

    def similarity_search_by_vector(
        self,
        query_embedding: list[float],
        k: int,
    ) -> list[tuple[Document, float]]:
        results = self._vectorstore.similarity_search_by_vector_with_relevance_scores(
            embedding=query_embedding,
            k=k,
        )
        return results

    def delete_collection(self) -> None:
        self._vectorstore.delete_collection()

    def get_collection_count(self) -> int:
        return self._vectorstore._collection.count()

    def rebuild(self, documents: list[Document]) -> list[str]:
        try:
            self.delete_collection()
        except Exception:
            pass

        self._vectorstore = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self.embedder.embedding_function,
            persist_directory=str(self.settings.chroma_persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        return self.add_documents(documents)
