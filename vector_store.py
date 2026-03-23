"""
rag_pipeline.py
---------------
End-to-end Retrieval-Augmented Generation (RAG) pipeline.

Ties together:
  DocumentLoader  →  VectorStore  →  Generator

Usage (quick start)
-------------------
    from rag_pipeline import RAGPipeline

    pipe = RAGPipeline()
    pipe.ingest("my_document.pdf")   # or .txt / .md
    answer = pipe.ask("What is the return policy?")
    print(answer)
"""

import os
import time
from typing import List, Tuple, Optional

from document_loader import load_and_chunk, Chunk
from vector_store     import VectorStore
from generator        import get_generator


# ── Pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the full RAG loop:
        1. Ingest  — load document → chunk → embed → index
        2. Retrieve — embed query → FAISS top-k search
        3. Generate — build prompt → LLM → answer

    Parameters
    ----------
    embed_model   : Sentence-transformer model for embeddings.
    gen_mode      : 'hf' (HuggingFace, free) or 'openai'.
    gen_model     : Name of the generative model.
    chunk_size    : Words per chunk.
    chunk_overlap : Overlapping words between chunks.
    top_k         : Number of chunks retrieved per query.
    index_dir     : If set, save/load the FAISS index here.
    """

    def __init__(
        self,
        embed_model   : str = "all-MiniLM-L6-v2",
        gen_mode      : str = "hf",
        gen_model     : str = "google/flan-t5-base",
        chunk_size    : int = 300,
        chunk_overlap : int = 50,
        top_k         : int = 5,
        index_dir     : Optional[str] = None,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k         = top_k
        self.index_dir     = index_dir
        self.sources       : List[str] = []

        print("Initialising RAG Pipeline...")
        self.store     = VectorStore(embed_model)
        self.generator = get_generator(gen_mode, model_name=gen_model)

        # Load pre-built index if it exists
        if index_dir and os.path.exists(os.path.join(index_dir, "index.faiss")):
            self.store.load(index_dir)
            print(f"Loaded existing index from '{index_dir}'.")

    # ── Ingest ────────────────────────────────────────────────────────────

    def ingest(self, path: str) -> int:
        """
        Load a document, chunk it, and add it to the vector index.

        Parameters
        ----------
        path : Path to a .pdf, .txt, or .md file.

        Returns
        -------
        Number of new chunks added.
        """
        print(f"\nIngesting '{path}'...")
        t0     = time.time()
        chunks = load_and_chunk(path, self.chunk_size, self.chunk_overlap)
        print(f"  {len(chunks)} chunks created in {time.time()-t0:.1f}s")

        # Re-build index with all chunks (including previously ingested ones)
        all_chunks = self.store.chunks + chunks
        self.store.build(all_chunks)
        self.sources.append(os.path.basename(path))

        if self.index_dir:
            self.store.save(self.index_dir)

        return len(chunks)

    def ingest_many(self, paths: List[str]) -> None:
        """Ingest multiple documents at once."""
        all_chunks = list(self.store.chunks)   # keep existing
        for path in paths:
            print(f"\nLoading '{path}'...")
            chunks = load_and_chunk(path, self.chunk_size, self.chunk_overlap)
            all_chunks.extend(chunks)
            self.sources.append(os.path.basename(path))
            print(f"  {len(chunks)} chunks")

        print(f"\nBuilding index for {len(all_chunks)} total chunks...")
        self.store.build(all_chunks)
        if self.index_dir:
            self.store.save(self.index_dir)

    # ── Ask ───────────────────────────────────────────────────────────────

    def ask(
        self,
        query         : str,
        top_k         : Optional[int] = None,
        return_sources: bool = False,
    ) -> str:
        """
        Answer a question using the ingested documents.

        Parameters
        ----------
        query          : Natural-language question.
        top_k          : Override the default number of retrieved chunks.
        return_sources : If True, append cited sources to the answer.

        Returns
        -------
        Answer string (+ optional source citations).
        """
        if len(self.store) == 0:
            return "No documents ingested yet. Call ingest() first."

        k       = top_k or self.top_k
        t0      = time.time()
        results = self.store.retrieve(query, top_k=k)
        t_ret   = time.time() - t0

        answer  = self.generator.generate(query, results)
        t_gen   = time.time() - t0 - t_ret

        print(f"  Retrieved in {t_ret:.2f}s | Generated in {t_gen:.2f}s")

        if return_sources:
            sources = list(dict.fromkeys(c.source for c, _ in results))
            answer += f"\n\nSources: {', '.join(sources)}"

        return answer

    # ── Inspect ───────────────────────────────────────────────────────────

    def retrieve_only(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Chunk, float]]:
        """Return raw retrieved chunks without generating an answer."""
        return self.store.retrieve(query, top_k or self.top_k)

    def stats(self) -> dict:
        return {
            "documents_ingested" : len(self.sources),
            "total_chunks"       : len(self.store),
            "sources"            : self.sources,
            "embed_model"        : self.store.model_name,
        }

    def __repr__(self):
        return (f"RAGPipeline("
                f"chunks={len(self.store)}, "
                f"sources={self.sources})")


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # If a file path is passed as argument, use it; otherwise use sample text
    if len(sys.argv) > 1:
        doc_path = sys.argv[1]
        pipe = RAGPipeline(top_k=4)
        pipe.ingest(doc_path)
    else:
        # Create a tiny sample document on the fly
        sample_path = "/tmp/sample_doc.txt"
        with open(sample_path, "w") as f:
            f.write("""
DocuRAG — Document Question Answering System
============================================

Overview
--------
DocuRAG is a Retrieval-Augmented Generation (RAG) system that lets you
ask natural language questions about any document. It works in three steps:

Step 1 — Document Ingestion
The system loads your document (PDF, TXT, or Markdown), splits it into
overlapping chunks of ~300 words, and converts each chunk into a dense
vector embedding using a sentence-transformer model.

Step 2 — Retrieval
When you ask a question, the system converts your question into a vector
and searches the FAISS index for the top-5 most semantically similar chunks.
This is much more powerful than keyword search because it understands meaning.

Step 3 — Generation
The retrieved chunks are passed as context to a language model along with
your question. The model generates a concise, grounded answer.

Key Technologies
----------------
- Sentence Transformers: all-MiniLM-L6-v2 for fast, accurate embeddings
- FAISS: Facebook AI Similarity Search for vector indexing and retrieval
- HuggingFace Transformers: flan-t5-base for answer generation
- LangChain: optional orchestration layer for advanced workflows

Use Cases
---------
DocuRAG can be applied to customer support documentation, internal knowledge
bases, research papers, legal documents, and product manuals. It is especially
useful when documents are too long to fit in a language model's context window.
            """.strip())

        pipe = RAGPipeline(top_k=4)
        pipe.ingest(sample_path)

    print(f"\nPipeline ready: {pipe}")
    print(f"Stats: {pipe.stats()}\n")

    # Interactive Q&A loop
    print("=" * 60)
    print("DocuRAG — Ask questions about your document")
    print("Type 'quit' to exit | 'stats' to see index info")
    print("=" * 60)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() == "quit":
            print("Goodbye!")
            break
        if query.lower() == "stats":
            print(pipe.stats())
            continue

        answer = pipe.ask(query, return_sources=True)
        print(f"\nDocuRAG: {answer}")
