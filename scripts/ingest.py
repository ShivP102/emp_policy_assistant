from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.ingestion.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest company policy PDFs into ChromaDB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if the ingest manifest is up to date.",
    )
    args = parser.parse_args()

    settings = get_settings()
    pipeline = IngestionPipeline(settings)

    print(f"PDF directory: {settings.pdf_dir}")
    print(f"Chroma directory: {settings.chroma_persist_dir}")
    print(f"Embedding model: {settings.embedding_model}")

    if not args.force and not pipeline.is_stale():
        count = pipeline.store.get_collection_count()
        print("Index is up to date. Use --force to rebuild.")
        print(f"Chunks in store: {count}")
        return

    print("Starting ingestion...")
    result = pipeline.run(force=args.force)

    if result.skipped:
        print("Ingestion skipped; index already current.")
    else:
        print("Ingestion complete.")
        print(f"  Files processed: {result.files_processed}")
        print(f"  Pages loaded:    {result.pages_loaded}")
        print(f"  Chunks created:  {result.chunks_created}")
        print(f"  Chunks stored:   {result.chunks_stored}")


if __name__ == "__main__":
    main()
