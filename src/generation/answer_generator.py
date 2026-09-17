from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config.settings import Settings


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]


class AnswerGenerator:
    """Generate grounded answers from retrieved policy chunks using OpenAI."""

    SYSTEM_PROMPT = (
        "You are an employee policy assistant. Answer the user's question using ONLY "
        "the provided policy excerpts. If the excerpts do not contain enough information, "
        "say you do not know based on the available policies. Cite the source file and "
        "page number inline when referencing specific policy details."
    )

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to .env or Streamlit secrets."
            )
        self.settings = settings
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                (
                    "human",
                    "Policy excerpts:\n\n{context}\n\nQuestion: {question}",
                ),
            ]
        )
        self._chain = self._prompt | self._llm

    def generate(self, question: str, documents: list[Document]) -> AnswerResult:
        context = self._format_context(documents)
        response = self._chain.invoke({"context": context, "question": question})
        answer = response.content if hasattr(response, "content") else str(response)

        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page_label", doc.metadata.get("page", "?")),
                "rerank_score": doc.metadata.get("rerank_score"),
                "retrieval_score": doc.metadata.get("retrieval_score"),
                "excerpt": doc.page_content[:300],
            }
            for doc in documents
        ]

        return AnswerResult(answer=answer, sources=sources)

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        blocks: list[str] = []
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
            blocks.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(blocks)
