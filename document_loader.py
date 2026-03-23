"""
app.py
------
Streamlit web application for DocuRAG.

Run locally:
    streamlit run app.py

Features
--------
- Upload any PDF or TXT document
- Ask unlimited questions in a chat interface
- View retrieved source chunks with similarity scores
- Download Q&A history as CSV
- Supports multiple documents in one session
"""

import os
import time
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title = "DocuRAG — Document Q&A",
    page_icon  = "📄",
    layout     = "wide",
)

# ── Lazy pipeline import (only after st.set_page_config) ─────────────────────
@st.cache_resource(show_spinner="Loading models (first run may take ~60 seconds)...")
def load_pipeline():
    """Load the RAG pipeline once and cache it across sessions."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from rag_pipeline import RAGPipeline
    return RAGPipeline(
        embed_model   = "all-MiniLM-L6-v2",
        gen_mode      = "hf",
        gen_model     = "google/flan-t5-base",
        chunk_size    = 300,
        chunk_overlap = 50,
        top_k         = 5,
    )

# ── Session state initialisation ─────────────────────────────────────────────
if "chat_history"     not in st.session_state: st.session_state.chat_history     = []
if "ingested_files"   not in st.session_state: st.session_state.ingested_files   = []
if "last_sources"     not in st.session_state: st.session_state.last_sources     = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 DocuRAG")
    st.caption("Retrieval-Augmented Generation for Documents")
    st.divider()

    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type    = ["pdf", "txt", "md"],
        accept_multiple_files = True,
        help    = "Supports PDF, plain text, and Markdown files.",
    )

    if uploaded_files:
        pipe = load_pipeline()
        for uf in uploaded_files:
            if uf.name not in st.session_state.ingested_files:
                # Save to temp file then ingest
                tmp_path = f"/tmp/{uf.name}"
                with open(tmp_path, "wb") as f:
                    f.write(uf.read())
                with st.spinner(f"Processing '{uf.name}'..."):
                    n = pipe.ingest(tmp_path)
                st.success(f"✓ '{uf.name}' — {n} chunks indexed")
                st.session_state.ingested_files.append(uf.name)

    st.divider()
    st.subheader("2. Pipeline Settings")

    top_k = st.slider(
        "Chunks retrieved per query (top-k)",
        min_value=1, max_value=10, value=5,
        help="More chunks = more context but slower generation."
    )

    show_sources = st.checkbox("Show retrieved source chunks", value=True)

    st.divider()

    if st.session_state.ingested_files:
        st.subheader("Indexed Documents")
        for fname in st.session_state.ingested_files:
            st.markdown(f"- `{fname}`")

    st.divider()

    if st.session_state.chat_history:
        # Download history as CSV
        df = pd.DataFrame(st.session_state.chat_history, columns=["Question", "Answer"])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label    = "⬇ Download Q&A history (CSV)",
            data     = csv,
            file_name= "docurag_history.csv",
            mime     = "text/csv",
        )

        if st.button("🗑 Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("📄 DocuRAG — Document Question Answering")
st.markdown(
    "Upload a document on the left, then ask any question below. "
    "DocuRAG retrieves the most relevant passages and generates a grounded answer."
)

# Pipeline status banner
if not st.session_state.ingested_files:
    st.info("👈 Upload a document in the sidebar to get started.")
else:
    n_chunks = len(load_pipeline().store)
    st.success(
        f"**{len(st.session_state.ingested_files)} document(s) indexed** — "
        f"{n_chunks} chunks in the knowledge base. Ready to answer questions."
    )

st.divider()

# ── Chat history display ──────────────────────────────────────────────────────
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# ── Query input ───────────────────────────────────────────────────────────────
query = st.chat_input(
    "Ask a question about your document...",
    disabled = not st.session_state.ingested_files,
)

if query:
    pipe = load_pipeline()

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context and generating answer..."):
            t0 = time.time()

            # Retrieve chunks
            retrieved = pipe.retrieve_only(query, top_k=top_k)

            # Generate answer
            answer = pipe.generator.generate(query, retrieved)
            elapsed = time.time() - t0

        st.markdown(answer)
        st.caption(f"Answer generated in {elapsed:.1f}s")

        # Show source chunks
        if show_sources and retrieved:
            with st.expander(f"📎 Retrieved context ({len(retrieved)} chunks)", expanded=False):
                for i, (chunk, score) in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**[{i}]** `{chunk.source}` — similarity: `{score:.3f}`\n\n"
                        f"> {chunk.text[:300]}{'...' if len(chunk.text) > 300 else ''}"
                    )
                    if i < len(retrieved):
                        st.divider()

    # Save to history
    st.session_state.chat_history.append((query, answer))

# ── How it works expander ─────────────────────────────────────────────────────
with st.expander("ℹ️ How DocuRAG works"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Step 1 — Ingestion**")
        st.markdown(
            "Your document is split into overlapping chunks of ~300 words. "
            "Each chunk is embedded into a 384-dimensional vector using "
            "`all-MiniLM-L6-v2` and stored in a FAISS index."
        )
    with col2:
        st.markdown("**Step 2 — Retrieval**")
        st.markdown(
            "Your question is embedded using the same model. "
            "FAISS performs a cosine similarity search and returns "
            "the top-k most relevant chunks in milliseconds."
        )
    with col3:
        st.markdown("**Step 3 — Generation**")
        st.markdown(
            "The retrieved chunks are passed as context to `flan-t5-base`. "
            "The model generates a concise answer grounded in your document — "
            "not from its training data."
        )
