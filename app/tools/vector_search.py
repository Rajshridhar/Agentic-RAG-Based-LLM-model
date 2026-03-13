"""Vector search tool – wraps ChromaDB retrieval for use in tool-calling agents."""

from app.vector_store import get_retriever, load_vector_store


def search_vector_store(query: str, k: int = 5, vector_store=None) -> list:
    """Search the ChromaDB vector store for *query* and return top-*k* documents.
    """
    if vector_store is None:
        vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=k)
    return retriever.invoke(query)
