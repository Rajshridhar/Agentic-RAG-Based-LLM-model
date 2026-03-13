"""Text chunking using RecursiveCharacterTextSplitter."""

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
        chunk.metadata.update(
            {
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
                "readability_score": textstat.flesch_reading_ease(chunk.page_content),
            }
        )

    logger.info("Created %d chunk(s) from %d document(s)", len(chunks), len(documents))
    return chunks
