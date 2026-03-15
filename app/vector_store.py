"""Vector store backed by Pinecone – provides create, load, and retriever helpers."""

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
    RETRIEVER_K,
)
from app.embeddings import get_embeddings
from app.logger import get_logger

logger = get_logger(__name__)

# Shared Pinecone client (initialised once)
_pc_client = None


def _get_pinecone_client() -> Pinecone:
    """Return a cached Pinecone client."""
    global _pc_client  # noqa: PLW0603
    if _pc_client is None:
        _pc_client = Pinecone(api_key=PINECONE_API_KEY)
        logger.info("Pinecone client initialised")
    return _pc_client


def _ensure_index_exists() -> None:
    """Create the Pinecone index if it does not already exist."""
    pc = _get_pinecone_client()
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info(
            "Created Pinecone index '%s' (dim=%d)", PINECONE_INDEX_NAME, EMBEDDING_DIMENSION
        )
    else:
        logger.info("Pinecone index '%s' already exists", PINECONE_INDEX_NAME)


def create_vector_store(chunks: list) -> PineconeVectorStore:
    """Index *chunks* (LangChain Document objects) into Pinecone and return the store."""
    _ensure_index_exists()
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )
    logger.info(
        "Vector store created – %d chunk(s) indexed in '%s'",
        len(chunks),
        PINECONE_INDEX_NAME,
    )
    return vector_store


def load_vector_store() -> PineconeVectorStore:
    """Connect to an existing Pinecone index and return a LangChain vector store."""
    _ensure_index_exists()
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )
    logger.info("Vector store loaded from Pinecone index '%s'", PINECONE_INDEX_NAME)
    return vector_store


def check_duplicate_source(source_file: str) -> bool:
    """Return *True* if vectors with the given *source_file* already exist in the index."""
    pc = _get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)

    # Use list() with a metadata filter to see if any vectors have this source_file.
    # We only need to find one match to confirm a duplicate.
    try:
        results = index.list(limit=1)
        # list() doesn't support metadata filtering on all plans,
        # so fall back to a query-based approach.
        from app.embeddings import get_embeddings
        emb = get_embeddings()
        # Create a dummy query vector of the right dimension
        dummy_vector = emb.embed_query(source_file)
        query_result = index.query(
            vector=dummy_vector,
            top_k=1,
            filter={"source_file": {"$eq": source_file}},
            include_metadata=True,
        )
        if query_result.matches:
            logger.info("Duplicate detected: '%s' already exists in index", source_file)
            return True
    except Exception as exc:
        logger.warning("Duplicate check failed: %s — proceeding with ingestion", exc)
        return False
    return False


def get_retriever(vector_store: PineconeVectorStore, k: int = RETRIEVER_K):
    """Return a retriever for the given *vector_store* with top-*k* results."""
    return vector_store.as_retriever(search_kwargs={"k": k})
