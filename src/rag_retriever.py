"""
rag_retriever.py
================

A minimal, dependency-light RAG retriever you can build on.

Implements the core retrieval loop — chunk -> embed -> store -> search — behind
small interfaces so you can swap the embedder (Anthropic, OpenAI, sentence-
transformers) or the vector store (in-memory, FAISS, Qdrant, pgvector) without
touching the rest. Ships with an in-memory cosine-similarity store so it runs
out of the box with zero infrastructure.

Design goals
------------
- Pluggable ``Embedder`` and ``VectorStore`` via ABCs.
- A working in-memory store using numpy cosine similarity (no FAISS needed).
- Sensible chunking with overlap.
- Clear separation: indexing vs. querying.

Example
-------
    from rag_retriever import RAGRetriever, InMemoryVectorStore, HashingEmbedder

    docs = ["Paris is the capital of France.",
            "Spark runs computations across a cluster.",
            "RAG augments an LLM with retrieved context."]

    rag = RAGRetriever(embedder=HashingEmbedder(dim=256),
                       store=InMemoryVectorStore())
    rag.index(docs)
    for hit in rag.retrieve("What does Spark do?", k=2):
        print(round(hit.score, 3), hit.text)

Swap ``HashingEmbedder`` for a real one (see ``Anthropic-style`` note below) and
this becomes production-shaped.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping character windows.

    Overlap preserves context across chunk boundaries so a sentence split in two
    still retrieves well. Tune ``chunk_size``/``overlap`` per corpus — it's one
    of the biggest levers on RAG quality.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------
class Embedder(ABC):
    """Turns text into fixed-length vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """
    A deterministic, dependency-free embedder for demos and tests.

    Uses hashed token features (the "hashing trick"). NOT semantically strong —
    replace with a real model in production — but it makes the whole pipeline
    runnable with no API key or model download.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
        # L2-normalize so dot product == cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-8, None)


# Sketch of a real embedder — left here as a template, not imported by default:
#
# class AnthropicEmbedder(Embedder):
#     def __init__(self, model="voyage-3"):   # Anthropic recommends Voyage models
#         import voyageai
#         self.client = voyageai.Client()
#         self.model = model
#     def embed(self, texts):
#         res = self.client.embed(texts, model=self.model, input_type="document")
#         return np.array(res.embeddings, dtype=np.float32)


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------
@dataclass
class Hit:
    text: str
    score: float
    metadata: dict


class VectorStore(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, texts: list[str],
            metadatas: list[dict]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine similarity store. Fine up to ~100k vectors."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def add(self, vectors: np.ndarray, texts: list[str],
            metadatas: list[dict]) -> None:
        self._vectors = (vectors if self._vectors is None
                         else np.vstack([self._vectors, vectors]))
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        if self._vectors is None:
            return []
        # vectors are L2-normalized -> dot product is cosine similarity
        scores = self._vectors @ query_vector
        top = np.argsort(scores)[::-1][:k]
        return [Hit(self._texts[i], float(scores[i]), self._metadatas[i])
                for i in top]


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
class RAGRetriever:
    """Ties chunking + embedding + storage + search into one object."""

    def __init__(self, embedder: Embedder, store: VectorStore,
                 chunk_size: int = 500, overlap: int = 50) -> None:
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size
        self.overlap = overlap

    def index(self, documents: list[str],
              metadatas: list[dict] | None = None) -> int:
        """Chunk, embed, and store documents. Returns number of chunks indexed."""
        all_chunks: list[str] = []
        all_meta: list[dict] = []
        for doc_id, doc in enumerate(documents):
            base = (metadatas[doc_id] if metadatas else {}) | {"doc_id": doc_id}
            for ci, chunk in enumerate(chunk_text(doc, self.chunk_size, self.overlap)):
                all_chunks.append(chunk)
                all_meta.append(base | {"chunk_id": ci})
        if not all_chunks:
            return 0
        vectors = self.embedder.embed(all_chunks)
        self.store.add(vectors, all_chunks, all_meta)
        return len(all_chunks)

    def retrieve(self, query: str, k: int = 4) -> list[Hit]:
        """Return the top-k most similar chunks to the query."""
        qvec = self.embedder.embed([query])[0]
        return self.store.search(qvec, k)

    def build_context(self, query: str, k: int = 4,
                      max_chars: int = 2000) -> str:
        """Concatenate retrieved chunks into a prompt-ready context block."""
        hits = self.retrieve(query, k)
        parts, total = [], 0
        for h in hits:
            if total + len(h.text) > max_chars:
                break
            parts.append(h.text)
            total += len(h.text)
        return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    docs = [
        "Paris is the capital of France and sits on the Seine.",
        "Apache Spark runs computations in parallel across a cluster of machines.",
        "RAG augments a language model with externally retrieved context.",
    ]
    rag = RAGRetriever(embedder=HashingEmbedder(dim=256), store=InMemoryVectorStore())
    n = rag.index(docs)
    print(f"indexed {n} chunks\n")
    for hit in rag.retrieve("What does Spark do?", k=2):
        print(f"{hit.score:.3f}  {hit.text}")
