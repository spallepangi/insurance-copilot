# InsuranceCopilot AI

Production-grade RAG system for healthcare insurance policy documents. It supports semantic retrieval, policy reasoning, cross-plan comparison, source citations, and observability (latency, cost, evaluation).

## Why this project / Problem we're solving

**Problem:** Understanding health insurance plans is hard. Policy PDFs are long, dense, and full of jargon. People need quick, accurate answers to questions like “What’s my emergency room copay?” or “How does prescription coverage differ between Silver and Gold?” without reading dozens of pages.

**Why we built this:** We built InsuranceCopilot AI so users can ask natural-language questions and get answers grounded in their actual policy text, with clear citations (plan, section, page). The system improves retrieval quality (recall and precision) via smaller chunks, hybrid vector + keyword search, weighted scoring, and reranking—so the right excerpts reach the LLM and answers stay accurate and trustworthy.

## Overview

InsuranceCopilot AI reads insurance policy PDFs (e.g. Bronze, Silver, Gold, Platinum), chunks them by section, embeds with **BAAI/bge-large-en**, stores vectors in **Qdrant**, and answers questions via hybrid search + **bge-reranker-large** + LLM. Answers include plan name, section title, page number, and policy excerpts.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           InsuranceCopilot AI                             │
├─────────────────────────────────────────────────────────────────────────┤
│  PDFs (data/)  →  Docling  →  Section Chunker  →  BGE Embeddings         │
│       ↓                                                                  │
│  Qdrant (vectors + metadata: plan, section, page)                        │
├─────────────────────────────────────────────────────────────────────────┤
│  User Query  →  Embed  →  Hybrid (vector 20 + BM25 20)  →  Merge 40   │
│       ↓                                                                  │
│  Rerank (bge-reranker-large)  →  Top 5  →  LLM  →  Answer + Citations   │
├─────────────────────────────────────────────────────────────────────────┤
│  Monitoring: latency (p50/p95/p99), token usage, cost per query          │
│  Logs: JSONL + SQLite (logs/query_metrics.jsonl, logs/metrics.db)        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
insurance-copilot/
├── data/                    # PDFs: bronze.pdf, silver.pdf, gold.pdf, platinum.pdf
├── logs/                     # Query metrics (JSONL, SQLite)
├── src/
│   ├── ingestion/           # Docling parser, chunker, metadata
│   ├── embeddings/          # BGE embedder
│   ├── vector_store/        # Qdrant client, index builder
│   ├── retrieval/           # Retriever, hybrid search, reranker
│   ├── rag/                 # Pipeline, plan comparator, answer generator
│   ├── monitoring/          # Latency, cost, metrics logger
│   ├── evaluation/          # evaluation_dataset.json, runner, metrics
│   ├── api/                 # FastAPI app
│   ├── ui/                  # Streamlit app
│   └── utils/               # config, logger
├── scripts/
│   ├── ingest_documents.py  # Parse → chunk → embed → Qdrant
│   ├── run_pipeline.py      # CLI / API / UI / latency-stats
│   └── compute_latency_stats.py
├── .env.example
├── requirements.txt
└── README.md
```

## Installation

1. **Clone / create project and install dependencies**

   ```bash
   cd insurance-copilot
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Add API keys**

   Copy `.env.example` to `.env` and fill in:

   - `OPENAI_API_KEY` — required for answer generation
   - `QDRANT_URL` — default `http://localhost:6333` (run Qdrant locally or use cloud)
   - `QDRANT_API_KEY` — if using Qdrant Cloud
   - `HUGGINGFACE_TOKEN` — if BGE models are gated
   - `LANGCHAIN_API_KEY` — optional, for LangSmith monitoring

3. **Place PDFs in `data/`**

   Add your policy PDFs as:

   - `data/bronze.pdf`, `data/silver.pdf`, `data/gold.pdf`, `data/platinum.pdf`  
   or  
   - `data/bbbronze.pdf`, `data/bbsilver.pdf`, etc. (names are mapped to Bronze/Silver/Gold/Platinum)

4. **Run Qdrant** (if local)

   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Usage

**Do not run ingestion or tests automatically.** Add API keys to `.env` first, then run manually.

1. **Ingest documents** (once)

   ```bash
   python -m scripts.ingest_documents
   ```

   This parses PDFs with Docling, chunks by section (450 tokens, 120 overlap), embeds with BGE, upserts into Qdrant, and builds the BM25 index for hybrid retrieval.

2. **Run the pipeline**

   - **CLI:** `python -m scripts.run_pipeline`  
     Then type questions or `compare &lt;question&gt;`.
   - **API:** `python -m scripts.run_pipeline api`  
     FastAPI at `http://localhost:8000` — `POST /query`, `POST /compare`, `GET /metrics`.
   - **UI:** `python -m scripts.run_pipeline ui`  
     Streamlit with search bar and plan comparison.

3. **Latency statistics**

   ```bash
   python -m scripts.compute_latency_stats
   ```

   Uses `logs/query_metrics.jsonl` or `logs/metrics.db`.

4. **Evaluation** (optional)

   ```bash
   python -m src.evaluation.evaluation_runner
   ```

   Runs queries from `src/evaluation/evaluation_dataset.json` and reports retrieval recall@5, precision@5, latency, cost.

## Example Queries

- Does the Bronze plan cover emergency room visits?
- What is the deductible for the Gold plan?
- Compare prescription drug coverage between Silver and Platinum.
- Which plan has the lowest out-of-pocket maximum?

## System Metrics

- **Latency:** per-query and percentiles (p50, p95, p99) via `LatencyTracker` and `scripts/compute_latency_stats`.
- **Cost:** per-query cost from OpenAI token usage; logged in `MetricsLogger`.
- **Observability:** every query logged to JSONL and SQLite; optional LangSmith via `LANGCHAIN_API_KEY`.

## Future Improvements

- Full LangSmith tracing for retrieval and generation.
- Sparse BM25 is already used (see `src/retrieval/bm25_index.py`).
- Evaluation with answer correctness (e.g. LLM-as-judge).
- Caching for repeated queries and embedding reuse.
