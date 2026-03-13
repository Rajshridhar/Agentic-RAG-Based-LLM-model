"""Text chunking using RecursiveCharacterTextSplitter."""

import textstat
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
                "readability_score": textstat.flesch_reading_ease(chunk.page_content),
            }
        )

    print(f"[✓] Created {len(chunks)} chunk(s) from {len(documents)} document(s)")
    return chunks
