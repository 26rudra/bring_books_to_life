"""The one tool that makes this *Agentic* RAG: a book-search the agent calls itself.

The agent (in character) decides when it needs canon details and calls search_book;
the retriever returns the most relevant passages, which the agent then reasons over.
That "the agent chooses when to retrieve" is the difference from plain RAG.
"""
from __future__ import annotations

from langchain_core.tools import tool


def make_search_tool(retriever):
    """Bind a retriever into a search_book tool the agent can call."""

    @tool
    def search_book(query: str) -> str:
        """Search the source book for passages relevant to the query.

        Call this whenever you need a fact, quote, or detail about yourself, other
        characters, or events — so your answer stays true to the book. Pass a short
        topic or question as the query.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant passages found in the book."
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return search_book
