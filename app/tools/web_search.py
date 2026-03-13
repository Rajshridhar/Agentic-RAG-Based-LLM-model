"""Web search tool – placeholder for a future web-search fallback.

In a production system this would integrate with a real search API
(e.g. Tavily, SerpAPI, DuckDuckGo).  For now it returns a clear message
indicating that web search is not yet configured.
"""


def web_search(query: str) -> str:  # noqa: ARG001
    """Placeholder web search.

    Parameters
    ----------
    query:
        The search query string (currently unused).

    Returns
    -------
    str
        A message indicating web search is not yet available.
    """
    return (
        "Web search is not configured in this deployment. "
        "Please consult an external source for up-to-date information."
    )
