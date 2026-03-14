"""Hallucination Grader Agent – checks whether the generated answer is
grounded in the supplied source documents.

Returns ``"yes"`` if the answer is grounded (no hallucination) or ``"no"``
if it appears to be hallucinated.
"""

import re

from langchain_core.prompts import PromptTemplate

from app.llm import get_llm
from app.logger import get_logger

logger = get_logger(__name__)

_HALLUCINATION_TEMPLATE = """You are a grader assessing whether an answer is grounded in the provided documents.

Rules:
- If the answer is supported by information in the documents, respond with exactly: yes
- If the answer contains claims or information NOT found in the documents, respond with exactly: no

Do NOT explain your reasoning. Output ONLY the single word "yes" or "no".

Documents: {documents}
Answer: {answer}
Verdict:"""


def grade_hallucination(answer: str, documents: list) -> str:
    """Return ``"yes"`` if *answer* is grounded in *documents*, else ``"no"``."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(_HALLUCINATION_TEMPLATE)
    chain = prompt | llm

    # Truncate context to keep the prompt within model limits
    docs_text = "\n\n".join(doc.page_content[:300] for doc in documents)[:1500]

    logger.info("Checking hallucination for answer (%d chars) against %d document(s)", len(answer), len(documents))
    raw = chain.invoke({"documents": docs_text, "answer": answer[:500]})
    result = _parse_yes_no(str(raw))
    logger.info("Hallucination check result: grounded=%s (raw: '%s')", result, str(raw).strip()[:200])
    return result


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
