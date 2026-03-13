"""Ingestion script – load, validate, chunk, and index a PDF document.

Usage::

    python scripts/ingest.py [--pdf PATH] [--chroma-dir PATH] [--recreate]
"""

import argparse
import os
import sys

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import shutil

from app.chunker import chunk_documents
from app.config import CHROMA_DB_PATH, PDF_PATH
from app.document_loader import load_and_validate_pdf
from app.logger import get_logger
from app.vector_store import create_vector_store

logger = get_logger(__name__)


def ingest(pdf_path: str, chroma_dir: str, recreate: bool = False) -> None:
    """Run the full ingestion pipeline.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file to ingest.
    chroma_dir:
        Path to the ChromaDB persistence directory.
    recreate:
        If *True*, delete and recreate the ChromaDB directory before indexing.
    """
    logger.info("Indian Cricket RAG – Ingestion Script")
    logger.info("PDF: %s", pdf_path)
    logger.info("ChromaDB: %s", chroma_dir)
    logger.info("Recreate: %s", recreate)

    if recreate and os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
        logger.warning("Deleted existing ChromaDB at '%s'", chroma_dir)

    # Step 1 – Load and validate
    logger.info("Step 1/3 – Loading and validating PDF…")
    documents = load_and_validate_pdf(pdf_path)

    # Step 2 – Chunk
    logger.info("Step 2/3 – Chunking documents…")
    chunks = chunk_documents(documents)

    # Step 3 – Create vector store
    logger.info("Step 3/3 – Creating vector store…")
    create_vector_store(chunks, chroma_dir)

    logger.info("Ingestion complete!")
    logger.info("Documents loaded: %d", len(documents))
    logger.info("Chunks created: %d", len(chunks))
    logger.info("ChromaDB path: %s", chroma_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG vector store.")
    parser.add_argument("--pdf", default=PDF_PATH, help="Path to the PDF file.")
    parser.add_argument(
        "--chroma-dir", default=CHROMA_DB_PATH, help="ChromaDB persistence directory."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the ChromaDB directory.",
    )
    args = parser.parse_args()
    ingest(args.pdf, args.chroma_dir, args.recreate)


if __name__ == "__main__":
    main()
