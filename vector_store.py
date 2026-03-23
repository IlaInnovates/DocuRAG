{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.12",
   "mimetype": "text/x-python",
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "pygments_lexer": "ipython3",
   "file_extension": ".py"
  },
  "colab": {
   "provenance": [],
   "gpuType": "T4",
   "name": "DocuRAG_Colab.ipynb"
  },
  "accelerator": "GPU"
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "attachments": {},
   "source": [
    "# DocuRAG — Document Question Answering with RAG\n",
    "\n",
    "**Author: Ilamangani S** | [GitHub](https://github.com/IlaInnovates) | [LinkedIn](https://linkedin.com/in/ilamangani-s-24a882216)\n",
    "\n",
    "---\n",
    "\n",
    "This notebook walks through a complete **Retrieval-Augmented Generation (RAG)** pipeline built from scratch.\n",
    "\n",
    "Instead of asking a raw LLM a question (which can hallucinate), DocuRAG **retrieves actual passages from your document** and generates an answer grounded only in those passages.\n",
    "\n",
    "### Pipeline overview\n",
    "\n",
    "```\n",
    "Document (.pdf / .txt)\n",
    "       ↓\n",
    "  Text Chunker          ← overlapping 300-word windows\n",
    "       ↓\n",
    "  Sentence Embeddings   ← all-MiniLM-L6-v2 (384-dim vectors)\n",
    "       ↓\n",
    "  FAISS Index           ← cosine similarity search\n",
    "       ↓  ← question vector\n",
    "  Top-5 Retrieval       ← most relevant chunks\n",
    "       ↓\n",
    "  Flan-T5 Generator     ← grounded answer\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "attachments": {},
   "source": [
    "## Step 1 — Install dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q sentence-transformers faiss-cpu transformers torch pymupdf\n",
    "print('All packages installed.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "attachments": {},
   "source": [
    "## Step 2 — Clone the repository"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!git clone https://github.com/IlaInnovates/DocuRAG.git\n",
    "%cd DocuRAG\n",
    "import sys\n",
    "sys.path.insert(0, 'src')\n",
    "print('Repository ready.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "attachments": {},
   "source": [
    "## Step 3 — Create a sample document\n",
    "\n",
    "We will create a short text file about RAG itself to use as our test document.\n",
    "In real use you can replace this with any PDF or TXT file."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_text = \"\"\"\n",
    "Retrieval-Augmented Generation (RAG)\n",
    "=====================================\n",
    "\n",
    "RAG is a technique that improves large language models by grounding their\n",
    "responses in retrieved documents. It was introduced by Lewis et al. (2020)\n",
    "in the paper Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.\n",
    "\n",
    "The pipeline has two main parts:\n",
    "1. Retriever: converts a query and documents into dense vectors and uses\n",
    "   approximate nearest-neighbour search (FAISS) to find the top-k most\n",
    "   relevant passages from the knowledge base.\n",
    "2. Generator: a sequence-to-sequence model (Flan-T5, BART, or GPT) that\n",
    "   receives the query and retrieved passages and generates a grounded answer.\n",
    "\n",
    "Why use RAG instead of fine-tuning?\n",
    "When the knowledge base changes you simply update the document index.\n",
    "There is no need to retrain the model. This makes RAG practical for\n",
    "production systems where documents are added or updated frequently.\n",
    "\n",
    "FAISS (Facebook AI Similarity Search)\n",
    "======================================\n",
    "FAISS is an open-source library from Meta AI for efficient similarity search\n",
    "over large collections of dense vectors. IndexFlatIP with L2-normalised\n",
    "vectors provides cosine similarity search, which works well for semantic\n",
    "document retrieval.\n",
    "\n",
    "Sentence Transformers\n",
    "======================\n",
    "Sentence Transformers (SBERT) map sentences into a dense vector space where\n",
    "semantically similar sentences are close to each other. The model\n",
    "all-MiniLM-L6-v2 produces 384-dimensional embeddings and is 5 times faster\n",
    "than BERT-base while keeping strong retrieval accuracy.\n",
    "\n",
    "Applications of RAG include customer support bots, internal knowledge bases,\n",
    "legal document analysis, medical Q&A, and enterprise document search.\n",
    "\"\"\"\n",
    "\n",
    "with open('/tmp/rag_overview.txt', 'w') as f:\n",
    "    f.write(sample_text)\n",
    "\n",
    "print('Sample document saved to /tmp/rag_overview.txt')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "attachments": {},
   "source": [
    "## Step 4 — Load and chunk the document\n",
    "\n",
    "We split the document into overlapping 300-word chunks.\n",
    "The 50-word overlap makes sure no answer is lost at a chunk boundary."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from document_loader import split_into_chunks, load_and_chunk\n",
    "\n",
    "chunks = load_and_chunk('/tmp/rag_overview.txt', chunk_size=80, overlap=15)\n",
    "\n",
    "print(f'Total chunks created: {len(chunks)}\\n')\n",
    "for c in chunks[:3]:\n",
    "    print(f'Chunk {c.chunk_id}: {c.text[:100]}...')\n",
    "    print()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "attachments": {},
   "source": [
    "## Step 5 — Build the FAISS vector index\n",
    "\n",
    "Each chunk is converted into a 384-dimensional vector using `all-MiniLM-L6-v2`.\n",
    "All vectors are stored in a FAISS `IndexFlatIP` index for cosine similarity search."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from vector_store import VectorStore\n",
    "\n",
    "store = VectorStore(model_name='all-MiniLM-L6-v2')\n",
    "store.build(chunks)\n",
    "\n",
    "print(f'\\nIndex ready: {store}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "attachments": {},
   "source": [
    "## Step 6 — Retrieve relevant chunks for a question\n",
    "\n",
    "The question is converted to a vector using the same model.\n",
    "FAISS finds the top 3 chunks whose vectors are closest to the question vector."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "query = 'What is FAISS and why is it used?'\n",
    "\n",
    "results = store.retrieve(query, top_k=3)\n",
    "\n",
    "print(f'Question: \"{query}\"\\n')\n",
    "for i, (chunk, score) in enumerate(results, 1):\n",
    "    print(f'[{i}] Similarity: {score:.3f}')\n",
    "    print(f'    {chunk.text[:200]}...')\n",
    "    print()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "attachments": {},
   "source": [
    "## Step 7 — Generate a grounded answer\n",
    "\n",
    "The retrieved chunks are passed as context to `google/flan-t5-base`.\n",
    "The model reads only those chunks and generates a concise answer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from generator import HFGenerator\n",
    "\n",
    "# First run downloads the model weights (~250 MB)\n",
    "gen = HFGenerator(model_name='google/flan-t5-base')\n",
    "\n",
    "answer = gen.generate(query, results)\n",
    "\n",
    "print(f'Question : {query}')\n",
    "print(f'Answer   : {answer}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "attachments": {},
   "source": [
    "## Step 8 — Run the full end-to-end pipeline\n",
    "\n",
    "The `RAGPipeline` class connects all three stages — chunker, retriever,\n",
    "and generator — into a single object."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from rag_pipeline import RAGPipeline\n",
    "\n",
    "pipe = RAGPipeline(top_k=4)\n",
    "pipe.ingest('/tmp/rag_overview.txt')\n",
    "\n",
    "print('Pipeline ready.')\n",
    "print(pipe.stats())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "questions = [\n",
    "    'What is Retrieval-Augmented Generation?',\n",
    "    'Why use RAG instead of fine-tuning?',\n",
    "    'What model is used for embeddings?',\n",
    "    'What are some applications of RAG?',\n",
    "]\n",
    "\n",
    "for q in questions:\n",
    "    print(f'Q: {q}')\n",
    "    a = pipe.ask(q)\n",
    "    print(f'A: {a}\\n')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "attachments": {},
   "source": [
    "## Step 9 — Try with your own document\n",
    "\n",
    "Upload any PDF or TXT file from your computer and ask questions about it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import files\n",
    "\n",
    "uploaded = files.upload()\n",
    "\n",
    "for filename in uploaded.keys():\n",
    "    print(f'Ingesting {filename}...')\n",
    "    pipe.ingest(filename)\n",
    "\n",
    "print('Done. Ask your questions below.')\n",
    "print(pipe.stats())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Change this question to anything about your document\n",
    "your_question = 'What is this document about?'\n",
    "\n",
    "answer = pipe.ask(your_question, return_sources=True)\n",
    "print(f'Q: {your_question}')\n",
    "print(f'A: {answer}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "attachments": {},
   "source": [
    "## Summary\n",
    "\n",
    "| Step | Component | Tool |\n",
    "|------|-----------|------|\n",
    "| 1 | Text chunking | Pure Python — overlapping word windows |\n",
    "| 2 | Embeddings | `all-MiniLM-L6-v2` — 384-dim vectors |\n",
    "| 3 | Vector index | FAISS `IndexFlatIP` — cosine similarity |\n",
    "| 4 | Generation | `google/flan-t5-base` — grounded answers |\n",
    "| 5 | Web UI | Streamlit — run `app.py` locally |\n",
    "\n",
    "**Key idea:** RAG = Retrieval + Generation. The retriever finds relevant evidence from the document; the generator synthesises a grounded answer from that evidence only."
   ]
  }
 ]
}