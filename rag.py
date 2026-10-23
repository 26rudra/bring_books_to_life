"""Just-enough RAG: turn a text 'book' into a searchable vector store.

Embeddings are LOCAL and FREE (fastembed — a small ONNX model, downloaded once,
no API key), so this runs offline. Groq has no embeddings API, so the retrieval
layer lives entirely here; Groq is only the LLM (see agent.py).

We talk to fastembed and faiss DIRECTLY rather than through a wrapper library.
That is deliberate, and it is the thing to understand about this file: the whole
of "RAG" is three explicit steps you can read here in full —

    1. embed the chunks   (fastembed -> vectors)
    2. index the vectors  (faiss     -> a searchable index)
    3. embed the query and take the top k nearest  (faiss.search)

A wrapper would collapse all three into one call. Seeing them separately is the
point of the session.
"""
from __future__ import annotations

import functools

import faiss
import numpy as np
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@functools.lru_cache(maxsize=1)
def _embedder() -> TextEmbedding:
    """The local embedding model. Cached so we build it once per process."""
    return TextEmbedding(model_name=MODEL_NAME)


def _embed(texts: list[str]) -> np.ndarray:
    """Embed texts and L2-normalise them.

    Normalising matters: bge models are trained for COSINE similarity, and a
    dot product over unit-length vectors is exactly cosine. That is why the
    index below uses inner product (IndexFlatIP) rather than raw L2 distance —
    on unnormalised vectors the two disagree and you get different neighbours.
    """
    vecs = np.array(list(_embedder().embed(texts)), dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


class BookRetriever:
    """A tiny top-k retriever over a FAISS index. Same surface the agent expects:
    `.invoke(query)` returns a list of Documents with `.page_content`."""

    def __init__(self, chunks: list[str], k: int = 4):
        self.chunks = chunks
        self.k = k
        vecs = _embed(chunks)
        # IndexFlatIP = exact (not approximate) inner-product search. "Flat"
        # means it compares against every vector — perfect recall, and plenty
        # fast for a book-sized corpus.
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def invoke(self, query: str) -> list[Document]:
        """Embed the query, take the k nearest chunks, return them as Documents."""
        scores, ids = self.index.search(_embed([query]), min(self.k, len(self.chunks)))
        return [Document(page_content=self.chunks[i], metadata={"score": float(s)})
                for i, s in zip(ids[0], scores[0]) if i != -1]


def _chunk(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    return splitter.split_text(text)


def build_retriever(source_path: str, k: int = 4) -> BookRetriever:
    """Read a text file, chunk it, embed the chunks, and return a top-k retriever."""
    with open(source_path, "r", encoding="utf-8") as f:
        return BookRetriever(_chunk(f.read()), k=k)


def retriever_from_text(text: str, k: int = 4) -> BookRetriever:
    """Same, but from an in-memory string (used by the upload flow in the UI)."""
    return BookRetriever(_chunk(text), k=k)
