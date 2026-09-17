from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from src.config.settings import Settings


class PDFDocumentLoader:
    """Load all PDFs from the configured folder using PyMuPDFLoader."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load(self) -> list[Document]:
        pdf_dir = self.settings.pdf_dir
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

        documents: list[Document] = []
        for pdf_path in pdf_paths:
            documents.extend(self._load_pdf(pdf_path))
        return documents

    def list_pdf_paths(self) -> list[Path]:
        if not self.settings.pdf_dir.exists():
            return []
        return sorted(self.settings.pdf_dir.glob("*.pdf"))

    def _load_pdf(self, pdf_path: Path) -> list[Document]:
        loader = PyMuPDFLoader(str(pdf_path))
        page_docs = loader.load()

        source_name = pdf_path.name
        for doc in page_docs:
            page = doc.metadata.get("page", 0)
            page_label = doc.metadata.get("page_label", str(page + 1))
            doc.metadata.update(
                {
                    "source": source_name,
                    "source_path": str(pdf_path),
                    "page": page,
                    "page_label": str(page_label),
                    "doc_id": f"{source_name}::page_{page}",
                }
            )
        return page_docs
