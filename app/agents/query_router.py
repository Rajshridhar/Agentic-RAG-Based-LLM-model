"""Query Router Agent – decides whether a question is relevant to Indian Cricket.

Uses the LLM to classify the query.  Returns ``"vectorstore"`` when the query
is about Indian Cricket history; ``"not_relevant"`` otherwise.
"""

from langchain_core.prompts import PromptTemplate

from app.llm import get_llm
from app.logger import get_logger

logger = get_logger(__name__)

_ROUTER_TEMPLATE = """Classify the question below. Reply with exactly one word.
Reply "yes" if the question is about Indian cricket history, players, matches, tournaments, IPL, or the Indian cricket team.
Reply "no" if the question is about anything else.

Question: {question}
Answer (yes or no):"""


def route_query(question: str) -> str:
    """Classify *question* and return the routing decision.

    Returns
    -------
    str
        ``"vectorstore"`` if the query is cricket-related, ``"not_relevant"``
        otherwise.
    """
    llm = get_llm()
    prompt = PromptTemplate.from_template(_ROUTER_TEMPLATE)
    chain = prompt | llm

    logger.info("Routing query: '%s'", question[:100])
    raw = chain.invoke({"question": question})
    # Normalise model output – small models can be noisy
    decision = str(raw).strip().lower()
    result = "vectorstore" if "yes" in decision else "not_relevant"
    logger.info("Route decision: %s (raw LLM output: '%s')", result, decision)
    return result
