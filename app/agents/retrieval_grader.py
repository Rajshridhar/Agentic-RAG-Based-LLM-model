"""Retrieval Grader Agent – filters retrieved documents by relevance.

For each document the LLM is asked: "Is this document relevant to the
question?".  Only documents that receive a "yes" score are kept.
"""

from langchain_core.prompts import PromptTemplate

from app.llm import get_llm
from app.logger import get_logger

logger = get_logger(__name__)

_GRADE_TEMPLATE = """You are grading document relevance. Reply with exactly one word: "yes" or "no".
Is the document below relevant to the question?

Question: {question}
Document: {document}
Relevant (yes or no):"""


def grade_documents(question: str, documents: list) -> list:
    """Return only the documents from *documents* that are relevant to *question*.

    Parameters
    ----------
    question:
        The user's original question.
    documents:
        Retrieved LangChain Document objects to grade.

    Returns
    -------
    list[Document]
        Subset of *documents* deemed relevant by the LLM.
    """
    llm = get_llm()
    prompt = PromptTemplate.from_template(_GRADE_TEMPLATE)
    chain = prompt | llm

    logger.info("Grading %d retrieved document(s) for relevance", len(documents))
    relevant_docs = []
    for i, doc in enumerate(documents):
        raw = chain.invoke({"question": question, "document": doc.page_content[:500]})
        score = str(raw).strip().lower()
        is_relevant = "yes" in score
        logger.debug("  Doc %d — relevant=%s (raw: '%s')", i + 1, is_relevant, score)
        if is_relevant:
            relevant_docs.append(doc)

    logger.info("Grading complete: %d/%d document(s) relevant", len(relevant_docs), len(documents))
    return relevant_docs
