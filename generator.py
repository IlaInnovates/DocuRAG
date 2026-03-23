"""
document_loader.py
------------------
Loads and chunks documents (plain text or PDF) into
overlapping windows ready for embedding.

Designed for DocuRAG — a document Q&A system built on top of
Retrieval-Augmented Generation (RAG).
"""

import re
import os
from typing import List, Dict


# ── Data class ────────────────────────────────────────────────────────────────

class Chunk:
    """A piece of text with its source metadata."""

    def __init__(self, text: str, source: str, chunk_id: int, start_char: int):
        self.text       = text.strip()
        self.source     = source          # filename or URL
        self.chunk_id   = chunk_id
        self.start_char = start_char

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, source='{self.source}', text='{preview}...')"


# ── Core chunker ──────────────────────────────────────────────────────────────

def split_into_chunks(
    text: str,
    source: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Split *text* into overlapping word-level windows.

    Parameters
    ----------
    text        : Full document text.
    source      : Label for this document (e.g. filename).
    chunk_size  : Approximate number of words per chunk.
    overlap     : Number of words shared between consecutive chunks
                  so context is not lost at boundaries.

    Returns
    -------
    List of Chunk objects.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    chunks: List[Chunk] = []
    step = chunk_size - overlap
    chunk_id = 0

    for start in range(0, len(words), step):
        end        = start + chunk_size
        window     = words[start:end]
        chunk_text = " ".join(window)
        start_char = len(" ".join(words[:start]))

        chunks.append(Chunk(
            text       = chunk_text,
            source     = source,
            chunk_id   = chunk_id,
            start_char = start_char,
        ))
        chunk_id += 1

        if end >= len(words):
            break

    return chunks


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_text_file(path: str) -> str:
    """Read a plain-text (.txt / .md) file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF (fitz).
    Install: pip install pymupdf
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF support.\n"
            "Install it with: pip install pymupdf"
        )

    doc   = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def load_document(path: str) -> str:
    """Auto-detect format and load document text."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".txt", ".md", ".rst"):
        return load_text_file(path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use .pdf, .txt, or .md")


def load_and_chunk(
    path: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Chunk]:
    """One-shot: load a document and return chunks."""
    text   = load_document(path)
    source = os.path.basename(path)
    return split_into_chunks(text, source, chunk_size, overlap)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = """
    Retrieval-Augmented Generation (RAG) is a technique that combines
    information retrieval with text generation. A retriever first fetches
    relevant documents from a knowledge base, then a generator uses those
    documents as context to produce a grounded answer.

    RAG was introduced by Lewis et al. (2020) and has since become a
    widely-used pattern for building knowledge-intensive NLP applications
    such as question-answering systems, chatbots, and document assistants.

    The two main components are:
    1. Retriever  — finds the top-k most relevant chunks for a query.
    2. Generator  — synthesises an answer conditioned on those chunks.
    """ * 5  # repeat to make chunking visible

    chunks = split_into_chunks(sample, source="demo.txt", chunk_size=60, overlap=10)
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks[:3]:
        print(c)
        print(f"  Text: {c.text[:80]}...\n")
