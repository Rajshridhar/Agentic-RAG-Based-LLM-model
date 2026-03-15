"""Text chunking using RecursiveCharacterTextSplitter."""

import os

import textstat
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.logger import get_logger

logger = get_logger(__name__)


def chunk_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        raw_source = chunk.metadata.get("source", "Unknown")
        source_file = os.path.basename(raw_source) if raw_source else "unknown"

        # Derive document title from the filename (without extension)
        doc_title = os.path.splitext(source_file)[0] if source_file != "unknown" else "Unknown"

        # Detect section heading: first line if it looks like a title (short, no period)
        first_line = chunk.page_content.strip().split("\n")[0].strip()
        section_name = first_line if len(first_line) < 100 and "." not in first_line else "General"

        chunk.metadata.update(
            {
                "chunk_id": i,
                "page": chunk.metadata.get("page", "N/A"),
                "document_title": doc_title,
                "section": section_name,
                "chunk_size": len(chunk.page_content),
                "has_tables": "|" in chunk.page_content or "─" in chunk.page_content,
                "readability_score": textstat.flesch_reading_ease(chunk.page_content),
                "source": raw_source,
                "source_file": source_file,
            }
        )

    logger.info("Created %d chunk(s) from %d document(s)", len(chunks), len(documents))
    return chunks
