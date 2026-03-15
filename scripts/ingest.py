"""Ingestion script – load, validate, chunk, and index a PDF document.

Usage::

    python scripts/ingest.py [--pdf PATH] [--recreate]
"""

import argparse
import os
import sys

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.chunker import chunk_documents
from app.config import PDF_PATH, PINECONE_API_KEY, PINECONE_INDEX_NAME
from app.document_loader import load_and_validate_pdf
from app.logger import get_logger
from app.vector_store import check_duplicate_source, create_vector_store

logger = get_logger(__name__)


def ingest(pdf_path: str, recreate: bool = False) -> None:
    """Run the full ingestion pipeline.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file to ingest.
    recreate:
        If *True*, delete all vectors from the Pinecone index before indexing.
    """
    logger.info("Indian Cricket RAG Ingestion Script")
    logger.info("PDF: %s", pdf_path)
    logger.info("Pinecone index: %s", PINECONE_INDEX_NAME)
    logger.info("Recreate: %s", recreate)

    if recreate:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True)
        logger.warning("Deleted all vectors from Pinecone index '%s'", PINECONE_INDEX_NAME)

    # Duplicate check (skip if --recreate was used since index was just cleared)
    if not recreate:
        source_file = os.path.basename(pdf_path)
        if check_duplicate_source(source_file):
            logger.error(
                "Duplicate document: '%s' has already been ingested. "
                "Use --recreate to clear the index and re-ingest.",
                source_file,
            )
            return

    # Step 1 – Load and validate
    logger.info("Step 1/3: Loading and validating PDF…")
    documents = load_and_validate_pdf(pdf_path)

    # Step 2 – Chunk
    logger.info("Step 2/3: Chunking documents…")
    chunks = chunk_documents(documents)

    # Step 3 – Create vector store
    logger.info("Step 3/3: Indexing chunks into Pinecone…")
    create_vector_store(chunks)

    logger.info("Ingestion complete!")
    logger.info("Documents loaded: %d", len(documents))
    logger.info("Chunks created: %d", len(chunks))
    logger.info("Pinecone index: %s", PINECONE_INDEX_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG vector store.")
    parser.add_argument("--pdf", default=PDF_PATH, help="Path to the PDF file.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete all vectors from the Pinecone index before re-indexing.",
    )
    args = parser.parse_args()
    ingest(args.pdf, args.recreate)


if __name__ == "__main__":
    main()
