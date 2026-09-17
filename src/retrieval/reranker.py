from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config.settings import Settings
from src.retrieval.retriever import ScoredDocument


class CrossEncoderReranker:
    """Rerank retrieved candidates using a cross-encoder model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = CrossEncoder(settings.cross_encoder_model)

    def rerank(
        self,
        query: str,
        candidates: list[ScoredDocument],
        top_k: int | None = None,
    ) -> list[Document]:
        if not candidates:
            return []

        final_k = top_k or self.settings.top_k_rerank
        pairs = [(query, candidate.document.page_content) for candidate in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        reranked_documents: list[Document] = []
        for candidate, score in ranked[:final_k]:
            document = Document(
                page_content=candidate.document.page_content,
                metadata=dict(candidate.document.metadata),
            )
            document.metadata["retrieval_score"] = candidate.score
            document.metadata["rerank_score"] = float(score)
            reranked_documents.append(document)

        return reranked_documents
