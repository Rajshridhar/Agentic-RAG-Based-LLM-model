"""Embedding model setup using Pinecone's hosted inference API."""

from langchain_pinecone import PineconeEmbeddings

from app.config import EMBEDDING_MODEL, PINECONE_API_KEY
from app.logger import get_logger

logger = get_logger(__name__)

_embeddings_instance = None


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> PineconeEmbeddings:
    """Return a (cached) PineconeEmbeddings instance.

    Uses Pinecone's hosted inference API so the embedding model
    (e.g. ``llama-text-embed-v2``) runs server-side — no local download.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = PineconeEmbeddings(
            model=model_name,
            pinecone_api_key=PINECONE_API_KEY,
        )
        logger.info("Pinecone embeddings model loaded: %s", model_name)
    return _embeddings_instance
