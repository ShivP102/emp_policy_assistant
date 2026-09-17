from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import Settings


class IngestState:
    """Track PDF fingerprints and ingest configuration in a manifest file."""

    MANIFEST_VERSION = 1

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.manifest_path = settings.manifest_path

    def compute_pdf_fingerprints(self, pdf_paths: list[Path]) -> dict[str, dict]:
        fingerprints: dict[str, dict] = {}
        for pdf_path in pdf_paths:
            stat = pdf_path.stat()
            fingerprints[str(pdf_path.resolve())] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "sha256": self._hash_file(pdf_path),
            }
        return fingerprints

    def load_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_manifest(
        self,
        pdf_fingerprints: dict[str, dict],
        chunks_stored: int,
    ) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "manifest_version": self.MANIFEST_VERSION,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": self.settings.embedding_model,
            "chunk_config_version": self.settings.chunk_config_version,
            "chunk_max_chars": self.settings.chunk_max_chars,
            "chunk_min_chars": self.settings.chunk_min_chars,
            "chunk_overlap": self.settings.chunk_overlap,
            "collection_name": self.settings.collection_name,
            "chunks_stored": chunks_stored,
            "pdfs": pdf_fingerprints,
        }
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    def is_stale(self, pdf_paths: list[Path]) -> bool:
        manifest = self.load_manifest()
        if manifest is None:
            return True

        if manifest.get("embedding_model") != self.settings.embedding_model:
            return True
        if manifest.get("chunk_config_version") != self.settings.chunk_config_version:
            return True
        if manifest.get("chunk_max_chars") != self.settings.chunk_max_chars:
            return True
        if manifest.get("chunk_min_chars") != self.settings.chunk_min_chars:
            return True
        if manifest.get("chunk_overlap") != self.settings.chunk_overlap:
            return True

        current = self.compute_pdf_fingerprints(pdf_paths)
        stored = manifest.get("pdfs", {})
        if set(current.keys()) != set(stored.keys()):
            return True

        for path, fingerprint in current.items():
            if stored.get(path) != fingerprint:
                return True

        return False

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
