from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from src.config.settings import Settings
from src.ingestion.chunker import PolicyChunker
from src.ingestion.document_loader import PDFDocumentLoader
from src.ingestion.embedder import HuggingFaceEmbedder
from src.utils.ingest_state import IngestState
from src.vector_store.chroma_store import ChromaStore


@dataclass
class IngestResult:
    files_processed: int
    pages_loaded: int
    chunks_created: int
    chunks_stored: int
    skipped: bool = False


class IngestionPipeline:
    """Orchestrate PDF loading, chunking, embedding, and Chroma persistence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loader = PDFDocumentLoader(settings)
        self.chunker = PolicyChunker(settings)
        self.embedder = HuggingFaceEmbedder(settings)
        self.store = ChromaStore(settings, self.embedder)
        self.ingest_state = IngestState(settings)

    def is_stale(self) -> bool:
        pdf_paths = self.loader.list_pdf_paths()
        if not pdf_paths:
            return True
        if not self.settings.manifest_path.exists():
            return True
        return self.ingest_state.is_stale(pdf_paths)

    def run(self, force: bool = False) -> IngestResult:
        pdf_paths = self.loader.list_pdf_paths()
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in {self.settings.pdf_dir}")

        if not force and not self.is_stale():
            return IngestResult(
                files_processed=len(pdf_paths),
                pages_loaded=0,
                chunks_created=0,
                chunks_stored=self.store.get_collection_count(),
                skipped=True,
            )

        documents = self.loader.load()
        chunks = self.chunker.chunk(documents)
        self.store.rebuild(chunks)

        fingerprints = self.ingest_state.compute_pdf_fingerprints(pdf_paths)
        self.ingest_state.save_manifest(fingerprints, chunks_stored=len(chunks))

        return IngestResult(
            files_processed=len(pdf_paths),
            pages_loaded=len(documents),
            chunks_created=len(chunks),
            chunks_stored=len(chunks),
        )
