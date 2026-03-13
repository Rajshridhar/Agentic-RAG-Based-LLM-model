from langchain_chroma import Chroma

from app.config import CHROMA_DB_PATH, RETRIEVER_K
from app.embeddings import get_embeddings
from app.chunker import chunk_documents
from app.logger import get_logger

logger = get_logger(__name__)


def create_vector_store(chunks: list, persist_directory: str = CHROMA_DB_PATH) -> Chroma:
    """Build a ChromaDB vector store from *chunks* and persist it to disk.
    """
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_directory,
    )
    logger.info("Vector store created with %d chunk(s) at '%s'", len(chunks), persist_directory)
    return vector_store


def load_vector_store(persist_directory: str = CHROMA_DB_PATH) -> Chroma:
    """Load an existing ChromaDB vector store from *persist_directory*.
    """
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    logger.info("Vector store loaded from '%s'", persist_directory)
    return vector_store


def get_retriever(vector_store: Chroma, k: int = RETRIEVER_K):
    """Return a retriever for the given *vector_store* with top-*k* results."""
    return vector_store.as_retriever(search_kwargs={"k": k})
