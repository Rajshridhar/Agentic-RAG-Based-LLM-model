"""Answer Grader Agent – checks whether the generated answer actually
addresses the user's question.

Returns ``"yes"`` if the answer is useful or ``"no"`` if it is off-topic /
unhelpful.
"""

import re

from langchain_core.prompts import PromptTemplate

from app.llm import get_llm
from app.logger import get_logger

logger = get_logger(__name__)

_ANSWER_GRADE_TEMPLATE = """You are a grader assessing whether an answer addresses a question.

Rules:
- If the answer addresses or attempts to answer the question, respond with exactly: yes
- If the answer is completely off-topic or does not address the question at all, respond with exactly: no

Do NOT explain your reasoning. Output ONLY the single word "yes" or "no".

Question: {question}
Answer: {answer}
Verdict:"""


def grade_answer(question: str, answer: str) -> str:
    """Return ``"yes"`` if *answer* addresses *question*, else ``"no"``."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(_ANSWER_GRADE_TEMPLATE)
    chain = prompt | llm

    logger.info("Grading answer usefulness for question: '%s'", question[:100])
    raw = chain.invoke({"question": question, "answer": answer[:500]})
    result = _parse_yes_no(str(raw))
    logger.info("Answer grade result: useful=%s (raw: '%s')", result, str(raw).strip()[:200])
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
