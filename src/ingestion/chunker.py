from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import Settings


class PolicyChunker:
    """Two-stage chunker: recursive markdown-aware split, then sentence split."""

    RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    SENTENCE_SEPARATORS = [". ", "? ", "! ", "\n", " "]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_max_chars,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=self.RECURSIVE_SEPARATORS,
            is_separator_regex=False,
        )
        self._sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_max_chars,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=self.SENTENCE_SEPARATORS,
            is_separator_regex=False,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        stage_one = self._recursive_splitter.split_documents(documents)
        final_chunks: list[Document] = []

        for chunk_index, chunk in enumerate(stage_one):
            if len(chunk.page_content) > self.settings.chunk_max_chars:
                sentence_chunks = self._split_oversized_chunk(chunk)
                for sub_index, sentence_chunk in enumerate(sentence_chunks):
                    metadata = dict(sentence_chunk.metadata)
                    metadata.update(
                        {
                            "chunk_index": chunk_index * 1000 + sub_index,
                            "chunk_stage": "sentence",
                            "parent_doc_id": chunk.metadata.get("doc_id"),
                        }
                    )
                    sentence_chunk.metadata = metadata
                    final_chunks.append(sentence_chunk)
            else:
                metadata = dict(chunk.metadata)
                metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_stage": "recursive",
                        "parent_doc_id": chunk.metadata.get("doc_id"),
                    }
                )
                chunk.metadata = metadata
                final_chunks.append(chunk)

        return final_chunks

    def _split_oversized_chunk(self, chunk: Document) -> list[Document]:
        sentence_chunks = self._sentence_splitter.split_documents([chunk])
        merged: list[Document] = []
        buffer = ""
        buffer_metadata = dict(chunk.metadata)

        for sentence_chunk in sentence_chunks:
            candidate = f"{buffer}{sentence_chunk.page_content}".strip()
            if not buffer:
                buffer = sentence_chunk.page_content.strip()
                buffer_metadata = dict(sentence_chunk.metadata)
                continue

            if len(candidate) < self.settings.chunk_min_chars:
                buffer = candidate
                continue

            merged.append(Document(page_content=buffer, metadata=dict(buffer_metadata)))
            buffer = sentence_chunk.page_content.strip()
            buffer_metadata = dict(sentence_chunk.metadata)

        if buffer:
            if merged and len(buffer) < self.settings.chunk_min_chars:
                last = merged[-1]
                merged[-1] = Document(
                    page_content=f"{last.page_content} {buffer}".strip(),
                    metadata=last.metadata,
                )
            else:
                merged.append(Document(page_content=buffer, metadata=buffer_metadata))

        return merged or sentence_chunks
