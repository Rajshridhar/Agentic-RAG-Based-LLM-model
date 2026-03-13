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
from app.vector_store import create_vector_store


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
    print(f"\n{'='*60}")
    print(f"  Indian Cricket RAG – Ingestion Script")
    print(f"{'='*60}")
    print(f"  PDF:       {pdf_path}")
    print(f"  ChromaDB:  {chroma_dir}")
    print(f"  Recreate:  {recreate}")
    print(f"{'='*60}\n")

    if recreate and os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
        print(f"[!] Deleted existing ChromaDB at '{chroma_dir}'")

    # Step 1 – Load and validate
    print("Step 1/3 – Loading and validating PDF…")
    documents = load_and_validate_pdf(pdf_path)

    # Step 2 – Chunk
    print("\nStep 2/3 – Chunking documents…")
    chunks = chunk_documents(documents)

    # Step 3 – Create vector store
    print("\nStep 3/3 – Creating vector store…")
    create_vector_store(chunks, chroma_dir)

    print(f"\n{'='*60}")
    print("  Ingestion complete!")
    print(f"  Documents loaded: {len(documents)}")
    print(f"  Chunks created:   {len(chunks)}")
    print(f"  ChromaDB path:    {chroma_dir}")
    print(f"{'='*60}\n")


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
