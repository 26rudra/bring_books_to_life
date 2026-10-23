"""Bring Books to Life — the agent.

A LangGraph ReAct agent (on Groq) that plays a character grounded in an uploaded
book. It decides when to search the book (Agentic RAG), stays in canon, and
REMEMBERS the conversation via a LangGraph checkpointer (thread_id per session).

Needs a free Groq API key as GROQ_API_KEY (see .env.example).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

from personas import PERSONAS, DEFAULT
from rag import build_retriever, retriever_from_text
from tools import make_search_tool

load_dotenv(find_dotenv(usecwd=True))

# openai/gpt-oss-120b does the cleanest structured tool-calling on Groq for this
# chat+retrieval pattern. Tested on this workload, not assumed — re-test if you
# swap models: reliability here is task-specific, not a league table.
# Lighter option: GROQ_MODEL=openai/gpt-oss-20b
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# max_retries: the free Groq tier allows 8000 tokens per MINUTE, and this agent's
# context grows with every tool call — by the later steps a single request is ~4k,
# so two of them inside one minute trips the limit on a single run, with nobody
# else on your key. langchain-groq defaults to 2 retries, which is not enough for
# a long run; the Groq SDK honours the retry-after header, so raising this simply
# waits out the window instead of failing the whole run.
def _llm(temperature: float = 0.3) -> ChatGroq:
    # low-ish temperature: enough character warmth, while keeping tool-calling
    # and the persona stable across turns.
    return ChatGroq(model=MODEL, temperature=temperature, max_retries=8)


def _system(persona) -> str:
    # NOTE: describe the book-search in prose, never as tool_name(args) — that keeps
    # Groq/Llama from emitting a malformed tool call (the Week-1 lesson).
    return (
        f"You are {persona.name}, speaking in the first person, fully in character.\n"
        f"Voice: {persona.voice}\n\n"
        "You have the source book about you. Whenever you need a fact, quote, or detail "
        "about yourself, other characters, or events, look it up in the book before you "
        "answer, so everything you say is true to the book. Never invent facts that "
        "contradict it; if the book does not cover something, say so in character rather "
        "than making it up.\n\n"
        "Stay in character at all times. Never say you are an AI or a model, and never "
        "mention searching or retrieving — simply speak as yourself. Keep replies natural "
        "and conversational, a few sentences unless asked for more."
    )


class BookChat:
    """One in-character conversation. Remembers previous turns (checkpointer + thread_id)."""

    def __init__(self, persona_key: str = DEFAULT, source_text: str | None = None,
                 thread_id: str = "session"):
        self.persona = PERSONAS[persona_key]
        retriever = (retriever_from_text(source_text) if source_text
                     else build_retriever(self.persona.source_path))
        self.agent = create_agent(
            _llm(), [make_search_tool(retriever)],
            system_prompt=_system(self.persona), checkpointer=InMemorySaver(),
        )
        self.thread_id = thread_id

    def say(self, message: str, verbose: bool = False) -> str:
        result = self.agent.invoke(
            {"messages": [("user", message)]},
            config={"configurable": {"thread_id": self.thread_id}, "recursion_limit": 12},
        )
        messages = result["messages"]
        if verbose:
            _print_trace(messages[-6:])
        return messages[-1].content


def _print_trace(messages) -> None:
    for m in messages:
        role = m.__class__.__name__.replace("Message", "").upper()
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            for tc in tcs:
                print(f"  · {role} → searches the book: {tc['args'].get('query', '')!r}")
        elif role == "TOOL":
            snippet = str(m.content).replace("\n", " ")[:110]
            print(f"  · retrieved: {snippet}…")
