"""Answer Grader Agent – checks whether the generated answer actually
addresses the user's question.

Returns ``"yes"`` if the answer is useful or ``"no"`` if it is off-topic /
unhelpful.
"""

from langchain_core.prompts import PromptTemplate

from app.llm import get_llm

_ANSWER_GRADE_TEMPLATE = """Does the answer below address the question? Reply with exactly one word: "yes" or "no".

Question: {question}
Answer: {answer}
Addresses the question (yes or no):"""


def grade_answer(question: str, answer: str) -> str:
    """Return ``"yes"`` if *answer* addresses *question*, else ``"no"``."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(_ANSWER_GRADE_TEMPLATE)
    chain = prompt | llm

    raw = chain.invoke({"question": question, "answer": answer[:500]})
    score = str(raw).strip().lower()
    return "yes" if "yes" in score else "no"
