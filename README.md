# DocuRAG — Document Question Answering System

Ask questions about any PDF or text document in plain English and get accurate, grounded answers — powered by Retrieval-Augmented Generation (RAG).

---

## What it does

Most people have experienced asking an AI a question about a document and getting a completely made-up answer. DocuRAG solves that by building a system that **only answers from what is actually in your document**.

You upload a PDF or text file. You ask a question. The system searches through the document to find the most relevant passages, then uses a language model to generate a concise answer from those passages — not from anything else.

---

## How it works

The pipeline has four stages:

**1. Chunking**
The document is split into overlapping chunks of 300 words each, with a 50-word overlap between consecutive chunks. The overlap is important — it makes sure that an answer sitting at the boundary of two chunks is never missed.

**2. Embedding**
Each chunk is passed through `all-MiniLM-L6-v2`, a Sentence Transformer model that converts text into a 384-dimensional vector. Chunks that mean similar things end up with vectors that are close to each other mathematically — even if they use different words.

**3. Retrieval**
When a user asks a question, that question is also converted into a vector using the same model. FAISS (Facebook AI Similarity Search) then finds the top 5 chunks whose vectors are closest to the question vector. This is semantic search — it finds meaning, not just matching keywords.

**4. Generation**
The 5 retrieved chunks are passed as context to `google/flan-t5-base`, a free instruction-tuned language model from Hugging Face. The model reads the context and generates a short, grounded answer. If the answer is not in the document, it says so.

```
Your Document
     │
     ▼
  Chunker  ──────────────────────────────────────────────────────────────┐
     │                                                                    │
     ▼                                                                    │
  Sentence Embeddings (all-MiniLM-L6-v2)                                 │
     │                                                                    │
     ▼                                                                    │
  FAISS Index  ◄── Question Vector                                       │
     │                                                                    │
     ▼                                                                    │
  Top-5 Relevant Chunks ──► Prompt Builder ──► Flan-T5 ──► Answer        │
                                                                          │
  Streamlit Web App  ◄───────────────────────────────────────────────────┘
```

---

## Project structure

```
DocuRAG/
│
├── src/
│   ├── document_loader.py   # Reads PDF and text files, splits into chunks
│   ├── vector_store.py      # Embeds chunks, builds FAISS index, retrieves
│   ├── generator.py         # Generates answers using Flan-T5 or OpenAI
│   └── rag_pipeline.py      # Connects all three stages into one pipeline
│
├── notebooks/
│   └── DocuRAG_Colab.ipynb  # Full walkthrough — run this in Google Colab
│
├── data/                    # Put your documents here (not tracked by git)
├── app.py                   # Streamlit web app
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Getting started

### Run in Google Colab (easiest — no setup needed)

Open `notebooks/DocuRAG_Colab.ipynb` directly in Google Colab. All packages install automatically in the first cell. The notebook walks through every step of the pipeline with explanations.

### Run locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/IlaInnovates/DocuRAG.git
cd DocuRAG
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Launch the web app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload any PDF or text file, then start asking questions.

### Use as a Python library

```python
from src.rag_pipeline import RAGPipeline

pipe = RAGPipeline()
pipe.ingest("my_document.pdf")

answer = pipe.ask("What are the main findings?")
print(answer)
```

---

## Technologies used

| Component | Library | Purpose |
|---|---|---|
| Text chunking | Pure Python | Overlapping word-window splitter |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` | 384-dim semantic vectors |
| Vector search | `faiss-cpu` — `IndexFlatIP` | Cosine similarity retrieval |
| Answer generation | `transformers` — `google/flan-t5-base` | Grounded answer generation |
| PDF reading | `pymupdf` | Extracts text from PDF files |
| Web interface | `streamlit` | Upload, chat, download history |
| Optional LLM | `openai` — GPT-3.5/4 | Higher quality generation |

---

## Web app features

- Upload PDF, TXT, or Markdown files
- Ask unlimited questions in a chat-style interface
- See which source chunks were used for each answer (with similarity scores)
- Ingest multiple documents into one knowledge base
- Download your question and answer history as a CSV file
- Adjust how many chunks are retrieved per question (top-k slider)

---

## Why I built this

I built DocuRAG to understand RAG from the ground up — without relying on a framework to hide the details. Each component (chunker, embedder, retriever, generator) is written separately so it is easy to understand, modify, and swap out.

The same architecture powers enterprise document tools used in production. Building it from scratch gave me a clear understanding of where things can go wrong — chunking strategy, embedding quality, retrieval precision — and how to improve each stage independently.

---

## Author

**Ilamangani S**  
[LinkedIn](https://linkedin.com/in/ilamangani-s-24a882216) · [GitHub](https://github.com/IlaInnovates)
