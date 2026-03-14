"""Query Router Agent – decides whether a question is relevant to Indian Cricket.

Uses the LLM to classify the query.  Returns ``"vectorstore"`` when the query
is about Indian Cricket; ``"not_relevant"`` otherwise.
"""

import re

from langchain_core.prompts import PromptTemplate

from app.llm import get_llm
from app.logger import get_logger

logger = get_logger(__name__)

_ROUTER_TEMPLATE = """You are a query classifier. Your ONLY job is to decide if the question is about cricket.

Rules:
- If the question is about cricket (Indian cricket, players, matches, IPL, tournaments, teams, records, scores, cricket history, BCCI, or anything related to the sport of cricket), respond with exactly: yes
- If the question is about something completely unrelated to cricket, respond with exactly: no

Do NOT explain your reasoning. Output ONLY the single word "yes" or "no".

Question: {question}
Answer:"""


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
    decision = _parse_yes_no(str(raw))
    result = "vectorstore" if decision == "yes" else "not_relevant"
    logger.info("Route decision: %s (raw LLM output: '%s')", result, str(raw).strip()[:200])
    return result


def _parse_yes_no(text: str) -> str:
    """Robustly extract 'yes' or 'no' from LLM output.

    Strategy: check the first meaningful word, then fall back to
    whichever keyword appears last (to handle 'No wait, yes').
    """
    cleaned = text.strip().lower()
    # Check if the response starts with yes/no
    first_word = re.split(r'[^a-z]', cleaned)[0]
    if first_word in ("yes", "no"):
        return first_word
    # Fallback: find the last occurrence of yes or no
    matches = re.findall(r'\b(yes|no)\b', cleaned)
    if matches:
        return matches[-1]
    # Default to yes for cricket-related system (safer to retrieve than reject)
    return "yes"
