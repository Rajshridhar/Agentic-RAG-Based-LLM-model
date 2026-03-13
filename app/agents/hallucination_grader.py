"""Hallucination Grader Agent – checks whether the generated answer is
grounded in the supplied source documents.

Returns ``"yes"`` if the answer is grounded (no hallucination) or ``"no"``
if it appears to be hallucinated.
"""

from langchain_core.prompts import PromptTemplate

from app.llm import get_llm

_HALLUCINATION_TEMPLATE = """Check if the answer below is fully supported by the provided documents. Reply with exactly one word: "yes" or "no".
Reply "yes" if the answer is grounded in the documents.
Reply "no" if the answer contains information not found in the documents.

Documents: {documents}
Answer: {answer}
Grounded (yes or no):"""


def grade_hallucination(answer: str, documents: list) -> str:
    """Return ``"yes"`` if *answer* is grounded in *documents*, else ``"no"``."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(_HALLUCINATION_TEMPLATE)
    chain = prompt | llm

    # Truncate context to keep the prompt within model limits
    docs_text = "\n\n".join(doc.page_content[:300] for doc in documents)[:1500]

    raw = chain.invoke({"documents": docs_text, "answer": answer[:500]})
    score = str(raw).strip().lower()
    return "yes" if "yes" in score else "no"
