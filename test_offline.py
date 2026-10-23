"""Offline tests — no LLM/API key needed (embeddings run locally).

Validates the RAG layer (chunking + retrieval) and the tool, so we know the
grounding works before spending Groq calls on the full agent.
Run: python test_offline.py   (first run downloads the small embedding model)
"""
from personas import PERSONAS, DEFAULT
from rag import build_retriever
from tools import make_search_tool


def check(name, cond):
    print(f"  {'✅' if cond else '❌'} {name}")
    assert cond, f"FAILED: {name}"


def main():
    print("Offline tests — RAG & tool (downloading embed model on first run)…")
    persona = PERSONAS[DEFAULT]

    retriever = build_retriever(persona.source_path, k=3)

    watson = retriever.invoke("Who is Doctor Watson?")
    check("retrieval returns passages", len(watson) > 0)
    check("Watson query surfaces Watson", any("Watson" in d.page_content for d in watson))

    moriarty = retriever.invoke("Who is the great villain?")
    check("Moriarty query surfaces Moriarty", any("Moriarty" in d.page_content for d in moriarty))

    method = retriever.invoke("how do you solve cases")
    check("method query surfaces deduction", any("deduc" in d.page_content.lower() for d in method))

    tool = make_search_tool(retriever)
    out = tool.invoke("Baker Street")
    check("search_book tool returns text", isinstance(out, str) and len(out) > 0)
    check("Baker Street search finds the address", "221B" in out or "Baker Street" in out)

    print("\n✅ All offline tests passed.")


if __name__ == "__main__":
    main()
