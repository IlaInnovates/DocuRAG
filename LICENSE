"""
vector_store.py
---------------
Embeds document chunks using a sentence-transformer model,
stores them in a FAISS index, and retrieves the top-k most
relevant chunks for any query.

This is the core 'R' (Retrieval) in RAG.
"""

import os
import pickle
import numpy as np
from typing import List, Tuple

from document_loader import Chunk


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Manages embeddings + FAISS index for a collection of Chunks.

    Parameters
    ----------
    model_name : SentenceTransformer model to use for embeddings.
                 'all-MiniLM-L6-v2' is fast, lightweight, and accurate
                 enough for document Q&A tasks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.model_name  = model_name
        self.model       = SentenceTransformer(model_name)
        self.faiss       = faiss
        self.index       = None          # FAISS IndexFlatIP
        self.chunks      : List[Chunk]  = []
        self.embeddings  : np.ndarray   = None

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, chunks: List[Chunk], show_progress: bool = True) -> None:
        """
        Embed all chunks and build the FAISS index.

        Uses cosine similarity via inner product on L2-normalised vectors
        (IndexFlatIP) — this is the standard approach for semantic search.
        """
        if not chunks:
            raise ValueError("No chunks provided. Load a document first.")

        print(f"Embedding {len(chunks)} chunks with '{self.model_name}'...")
        texts = [c.text for c in chunks]

        # Encode — returns (N, D) float32 numpy array
        embeddings = self.model.encode(
            texts,
            show_progress_bar = show_progress,
            convert_to_numpy  = True,
            normalize_embeddings = True,   # L2-normalise for cosine sim
        )

        dim         = embeddings.shape[1]
        self.index  = self.faiss.IndexFlatIP(dim)   # inner-product = cosine on normed vecs
        self.index.add(embeddings.astype("float32"))

        self.chunks     = chunks
        self.embeddings = embeddings
        print(f"Index built. {self.index.ntotal} vectors, dim={dim}.")

    # ── Retrieve ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Embed *query* and return the top-k most similar chunks.

        Returns
        -------
        List of (Chunk, similarity_score) tuples, highest score first.
        """
        if self.index is None:
            raise RuntimeError("Index is empty. Call build() first.")

        q_vec = self.model.encode(
            [query],
            convert_to_numpy     = True,
            normalize_embeddings = True,
        ).astype("float32")

        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:               # FAISS returns -1 for empty slots
                continue
            results.append((self.chunks[idx], float(score)))

        return results                  # already sorted by score desc

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save FAISS index + chunks to *directory*."""
        os.makedirs(directory, exist_ok=True)
        self.faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"VectorStore saved to '{directory}'.")

    def load(self, directory: str) -> None:
        """Load a previously saved VectorStore from *directory*."""
        self.index  = self.faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        print(f"VectorStore loaded. {self.index.ntotal} vectors.")

    # ── Helpers ────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.chunks)

    def __repr__(self):
        n = self.index.ntotal if self.index else 0
        return f"VectorStore(model='{self.model_name}', vectors={n})"


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from document_loader import split_into_chunks

    # Tiny in-memory demo — no files needed
    sample_docs = [
        ("RAG combines retrieval with generation to produce grounded answers. "
         "A retriever finds relevant chunks; a generator reads them to answer.", "doc_rag.txt"),
        ("FAISS (Facebook AI Similarity Search) is a library for efficient "
         "similarity search over dense vectors. It supports exact and approximate search.", "doc_faiss.txt"),
        ("Sentence transformers map sentences to a dense vector space where "
         "semantically similar sentences are close to each other.", "doc_st.txt"),
        ("LangChain is a framework for building LLM-powered applications. "
         "It provides utilities for chaining prompts, memory, and tools.", "doc_lc.txt"),
    ]

    all_chunks = []
    for text, src in sample_docs:
        all_chunks.extend(split_into_chunks(text, src, chunk_size=40, overlap=5))

    store = VectorStore()
    store.build(all_chunks, show_progress=False)

    print("\nQuery: 'How does vector search work?'")
    for chunk, score in store.retrieve("How does vector search work?", top_k=3):
        print(f"  [{score:.3f}] ({chunk.source}) {chunk.text[:80]}")
