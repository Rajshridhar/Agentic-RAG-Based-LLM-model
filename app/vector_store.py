from langchain_community.vectorstores import Chroma

from app.config import CHROMA_DB_PATH, RETRIEVER_K
from app.embeddings import get_embeddings
from app.chunker import chunk_document


def create_vector_store(chunks: list, persist_directory: str = CHROMA_DB_PATH) -> Chroma:
    """Build a ChromaDB vector store from *chunks* and persist it to disk.
    """
    embeddings = get_embeddings()
    chunks = chunk_document()
    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_directory,
    )
    print(f"[✓] Vector store created with {len(chunks)} chunk(s) at '{persist_directory}'")
    return vector_store


def load_vector_store(persist_directory: str = CHROMA_DB_PATH) -> Chroma:
    """Load an existing ChromaDB vector store from *persist_directory*.
    """
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    print(f"[✓] Vector store loaded from '{persist_directory}'")
    return vector_store


def get_retriever(vector_store: Chroma, k: int = RETRIEVER_K):
    """Return a retriever for the given *vector_store* with top-*k* results."""
    return vector_store.as_retriever(search_kwargs={"k": k})
