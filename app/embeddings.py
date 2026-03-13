"""Embedding model setup using HuggingFace sentence-transformers."""

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL

_embeddings_instance = None


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return a (cached) HuggingFaceEmbeddings instance.

    The model is loaded once and reused on subsequent calls to avoid
    redundant downloads / initialisation overhead.
    """
    global _embeddings_instance  # noqa: PLW0603
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name=model_name)
        print(f"[✓] Embeddings model loaded: {model_name}")
    return _embeddings_instance
