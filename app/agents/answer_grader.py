"""Answer Grader Agent – checks whether the generated answer actually
addresses the user's question.

Returns ``"yes"`` if the answer is useful or ``"no"`` if it is off-topic /
unhelpful.
"""

from langchain_core.prompts import PromptTemplate

from app.llm import get_llm
from app.logger import get_logger

logger = get_logger(__name__)

_ANSWER_GRADE_TEMPLATE = """Does the answer below address the question? Reply with exactly one word: "yes" or "no".

Question: {question}
Answer: {answer}
Addresses the question (yes or no):"""


def grade_answer(question: str, answer: str) -> str:
    """Return ``"yes"`` if *answer* addresses *question*, else ``"no"``."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(_ANSWER_GRADE_TEMPLATE)
    chain = prompt | llm

    logger.info("Grading answer usefulness for question: '%s'", question[:100])
    raw = chain.invoke({"question": question, "answer": answer[:500]})
    score = str(raw).strip().lower()
    result = "yes" if "yes" in score else "no"
    logger.info("Answer grade result: useful=%s (raw: '%s')", result, score)
    return result
