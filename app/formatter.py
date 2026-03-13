"""Response formatting utility for better readability."""

import re
import textwrap


def format_response(text: str, width: int = 100) -> str:
    """Clean up and format raw LLM output for better readability.

    - Strips leading/trailing whitespace
    - Collapses multiple blank lines into one
    - Wraps long lines to the specified width
    - Capitalises the first character of the response
    """
    if not text:
        return text

    # Strip leading/trailing whitespace
    text = text.strip()

    # Collapse 3+ consecutive newlines into 2 (one blank line)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Wrap each paragraph individually (preserve paragraph breaks)
    paragraphs = text.split("\n\n")
    wrapped = []
    for para in paragraphs:
        # Don't re-wrap bullet/numbered list items or very short lines
        lines = para.split("\n")
        if all(len(line) <= width for line in lines):
            wrapped.append(para)
        else:
            wrapped.append(textwrap.fill(para, width=width))

    text = "\n\n".join(wrapped)

    # Capitalise the first character
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text
