# Bring Books to Life 🎭

The Week 2 project for **Agentic AI for Developers**. Chat with a character from a book —
they stay **in character**, answer **grounded in the text** (Agentic RAG), and **remember**
your conversation.

Built with **LangGraph** + **Groq** (free tier). Embeddings are **local & free** (fastembed) —
Groq has no embeddings API, so retrieval runs entirely on your machine. Ships with a
public-domain **Sherlock Holmes** source; upload any `.txt` to ground the current character in a
new source (keeps their persona). For a genuinely new character, add a `Persona` in `personas.py`.

## What it demonstrates (Week 2 skills)
- **Agentic RAG** — the agent *decides when* to search the book (a retrieval tool), not a fixed pipeline
- **Just-enough RAG** — chunk → embed → FAISS vector store → top-k retrieval
- **Memory** — a LangGraph checkpointer (thread per session) so it remembers earlier turns
- **Grounding** — the persona answers from retrieved passages and stays true to canon

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then paste your free Groq key from https://console.groq.com/keys
```

## Run
```bash
python test_offline.py    # RAG sanity check (no key; first run downloads a small embed model)

python run.py --demo      # scripted 4-turn convo — shows retrieval + memory
python run.py sherlock -v # interactive chat; -v shows when the character searches the book
streamlit run app.py      # the chat UI (with a book uploader)
```

## Files
| File | Role |
|------|------|
| `sources/` | The book(s) — plain text. Ships with public-domain Sherlock Holmes. |
| `personas.py` | Character definitions (name, source, voice) |
| `rag.py` | Chunk + embed (local fastembed) + FAISS retriever |
| `tools.py` | The `search_book` tool the agent calls (Agentic RAG) |
| `agent.py` | LangGraph ReAct agent on Groq + memory (`BookChat`) |
| `run.py` | CLI (interactive + `--demo`) |
| `app.py` | Streamlit chat UI |
| `test_offline.py` | Offline RAG/tool tests |

## Notes
- **Embeddings are local** (fastembed, ONNX) — no embedding key, runs offline after the one-time model download.
- Swap the book by uploading a `.txt` in the UI, or add a `Persona` in `personas.py`.
