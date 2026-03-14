"""Retrieval Grader Agent – filters retrieved documents by relevance.

For each document the LLM is asked: "Is this document relevant to the
question?".  Only documents that receive a "yes" score are kept.
"""

import re

from langchain_core.prompts import PromptTemplate

from app.llm import get_llm
from app.logger import get_logger

logger = get_logger(__name__)

_GRADE_TEMPLATE = """You are a grader assessing whether a document is relevant to a question.

Rules:
- If the document contains information that could help answer the question, respond with exactly: yes
- If the document is completely unrelated to the question, respond with exactly: no

Do NOT explain your reasoning. Output ONLY the single word "yes" or "no".

Question: {question}
Document: {document}
Verdict:"""


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
        score = _parse_yes_no(str(raw))
        is_relevant = score == "yes"
        logger.debug("  Doc %d — relevant=%s (raw: '%s')", i + 1, is_relevant, str(raw).strip()[:200])
        if is_relevant:
            relevant_docs.append(doc)

    logger.info("Grading complete: %d/%d document(s) relevant", len(relevant_docs), len(documents))
    return relevant_docs


def _parse_yes_no(text: str) -> str:
    """Robustly extract 'yes' or 'no' from LLM output."""
    cleaned = text.strip().lower()
    first_word = re.split(r'[^a-z]', cleaned)[0]
    if first_word in ("yes", "no"):
        return first_word
    matches = re.findall(r'\b(yes|no)\b', cleaned)
    if matches:
        return matches[-1]
    return "no"
