"""Simple (non-agentic) LCEL RAG chain – kept as a fallback mode."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.formatter import format_response
from app.llm import get_llm
from app.logger import get_logger
from app.vector_store import get_retriever, load_vector_store

logger = get_logger(__name__)

_TEMPLATE = """You are a helpful assistant answering questions about Indian Cricket.
Answer the question based only on the provided context. If the answer is not in the context, say "I don't have information about this in the provided documents."

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs: list) -> str:
    """Concatenate document page contents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store=None):
    """Build and return a simple LCEL RAG chain.

    Parameters
    ----------
    vector_store:
        Optional pre-loaded Chroma vector store.  If *None* the store is
        loaded from the configured persistence directory.

    Returns
    -------
    Runnable
        A LangChain LCEL chain ready for ``invoke(question)``.
    """
    if vector_store is None:
        vector_store = load_vector_store()

    retriever = get_retriever(vector_store)
    llm = get_llm()
    prompt = PromptTemplate.from_template(_TEMPLATE)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def query_simple(question: str, vector_store=None) -> dict:
    """Run a simple RAG query and return answer + source documents.

    Returns
    -------
    dict
        ``{"answer": str, "sources": list[dict]}``
    """
    logger.info("=== Simple RAG query started: '%s' ===", question[:150])
    chain, retriever = build_rag_chain(vector_store)
    answer = format_response(chain.invoke(question))
    logger.info("Simple RAG answer (%d chars): '%s'", len(answer), answer[:150])
    docs = retriever.invoke(question)
    logger.info("Retrieved %d source document(s)", len(docs))
    sources = [
        {
            "page": doc.metadata.get("page", "N/A"),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            "content_preview": doc.page_content[:300],
        }
        for doc in docs
    ]
    return {"answer": answer, "sources": sources}
