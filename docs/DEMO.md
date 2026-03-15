# Quick Start & Project Overview

This document explains what the project does and how to run it quickly.

---

## What This Project Does

| Area | Description |
|------|--------------|
| **RAG pipeline** | End-to-end: PDF → parse → chunk → embed → vector + keyword search → rerank → LLM with citations. |
| **Retrieval** | Hybrid (vector + BM25), weighted scoring, reranker, section-aware chunking; evaluated with recall@5 / precision@5. |
| **Structure** | Modules: ingestion, embeddings, vector_store, retrieval, rag, evaluation, api, monitoring; config-driven, testable. |
| **Evaluation** | 100-question eval set, retrieval metrics, optional RAGAS (faithfulness, answer relevancy). |
| **Production** | Optional API auth, rate limiting, health checks, retries, timeouts, logging, tests, CI, Docker. |

---

## Quick Start (5–10 minutes)

**Prerequisites:** `.env` with `OPENAI_API_KEY`, `QDRANT_URL` (and `QDRANT_API_KEY` if using Qdrant Cloud), `HUGGINGFACE_TOKEN`. PDFs in `data/` (e.g. `bronze.pdf`, `silver.pdf`).

1. **Run the UI** (if ingestion is already done):
   ```bash
   .venv/bin/python -m scripts.run_pipeline ui
   ```
   Try: *"What is the deductible for the Gold plan?"* or *"Compare prescription coverage between Silver and Platinum."*

2. **Run a quick evaluation**:
   ```bash
   .venv/bin/python -m src.evaluation.evaluation_runner --limit=5
   ```
   With RAG + RAGAS (if installed):
   ```bash
   .venv/bin/python -m src.evaluation.evaluation_runner --rag --ragas --limit=3
   ```

3. **Run tests**:
   ```bash
   .venv/bin/pytest tests/ -v
   ```

4. **Start the API** (optional):
   ```bash
   .venv/bin/python -m scripts.run_pipeline api
   ```
   Then: `POST http://localhost:8000/query` with `{"question": "Does Bronze cover emergency room visits?"}`.

---

## Tech Stack

- **Language:** Python 3.10+
- **Document parsing:** Docling
- **Embeddings:** BAAI/bge-large-en (Hugging Face)
- **Vector DB:** Qdrant (local or Cloud)
- **Keyword search:** BM25 (custom index)
- **Reranker:** bge-reranker-large
- **LLM:** OpenAI (e.g. gpt-4o-mini)
- **API:** FastAPI
- **UI:** Streamlit
- **Evaluation:** Custom recall/precision + optional RAGAS
- **Tests:** pytest; CI: GitHub Actions
- **Deploy:** Dockerfile for API

---

## Design Choices

- **RAG:** Policy PDFs are long and jargon-heavy; RAG keeps answers grounded in actual text with citations (plan, section, page).
- **Hybrid retrieval:** Vector search captures semantics; BM25 captures exact terms (e.g. "deductible", "copay"). Combining both improved recall in evaluation.
- **Rerank:** A cross-encoder (bge-reranker) on a candidate set of 40 gives better precision than using only embedding similarity for the final top-5.
- **Section chunking:** Chunking by section (450 tokens, 120 overlap) keeps context coherent and improves section-level retrieval.
- **Evaluation:** 100-question set with expected section/plan; we track retrieval recall@5 and precision@5 and optionally RAGAS for answer quality.

---

## Where to Look in the Codebase

| Goal | Path |
|------|------|
| End-to-end RAG flow | `src/rag/rag_pipeline.py` |
| Hybrid search (vector + BM25) | `src/retrieval/hybrid_search.py`, `src/retrieval/retriever.py` |
| Chunking strategy | `src/ingestion/chunker.py` |
| Ingestion (parse → embed → Qdrant + BM25) | `scripts/ingest_documents.py`, `src/vector_store/index_builder.py` |
| Evaluation & metrics | `src/evaluation/evaluation_runner.py`, `src/evaluation/metrics.py`, `src/evaluation/ragas_metrics.py` |
| API & middleware | `src/api/app.py`, `src/api/middleware.py` |
| Config | `src/utils/config.py` |
| Full pipeline narrative | `docs/PIPELINE_WORKFLOW.md` |

---

## Full Setup (from scratch)

1. Clone repo, create venv, install: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`; set `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` (if cloud), `HUGGINGFACE_TOKEN`
3. Add PDFs to `data/` (e.g. `bronze.pdf`, `silver.pdf`, `gold.pdf`, `platinum.pdf`)
4. Run Qdrant (e.g. `docker run -p 6333:6333 qdrant/qdrant`) or use Qdrant Cloud
5. Ingest: `python -m scripts.ingest_documents` (15–20 min for 4 PDFs)
6. Run: `python -m scripts.run_pipeline ui` and/or evaluation/tests as above

Or use the setup script: `bash scripts/setup_and_run.sh` (requires `.env` and PDFs in place).

---

## Documentation

- **README.md** — Project overview, setup, usage, metrics
- **docs/DEMO.md** — This file: quick start and code map
- **docs/PIPELINE_WORKFLOW.md** — End-to-end pipeline (ingestion, query, evaluation)
- **src/evaluation/README.md** — Evaluation metrics and how to improve retrieval
- **docs/PRODUCTION_READINESS.md** — Production checklist
