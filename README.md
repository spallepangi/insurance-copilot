# InsuranceCopilot AI

Production-grade RAG system for healthcare insurance policy documents. It supports semantic retrieval, policy reasoning, cross-plan comparison, source citations, and observability (latency, cost, evaluation).

---

## For recruiters & interviewers

This repo is a **portfolio demo** for technical interviews. It implements a full RAG pipeline: PDF ingestion → hybrid retrieval (vector + BM25) → reranking → LLM answers with citations, plus evaluation (recall/precision and optional RAGAS), FastAPI, Streamlit UI, tests, and CI.

- **Quick demo:** If the project is already set up, run **`python -m scripts.run_pipeline ui`** for the Streamlit UI, or **`python -m src.evaluation.evaluation_runner --limit=5`** for a short evaluation. Run **`pytest tests/ -v`** to see tests.
- **Full guide:** See **[docs/DEMO.md](docs/DEMO.md)** for a 5‑minute demo script, tech stack, talking points, and where to look in the codebase.

**Highlights:** Docling PDF parsing · Section-aware chunking (450/120 tokens) · BGE embeddings + Qdrant · Hybrid search (vector top 20 + BM25 top 20 → merge 40) · bge-reranker-large → top 5 · OpenAI answer generation with plan/section/page citations · 100-question evaluation (recall@5, precision@5) · Optional RAGAS (faithfulness, answer relevancy) · FastAPI + Streamlit · pytest + GitHub Actions · Dockerfile · Optional API auth, rate limiting, health checks.

---

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

**Option A – Run everything from the beginning (recommended)**

From the project root, with `.env` already in place and PDFs in `data/`:

```bash
cd insurance-copilot
bash scripts/setup_and_run.sh
```

This will: create/use `.venv`, install dependencies, ingest documents (parse → chunk → embed → Qdrant + BM25), then run a 5-query evaluation. Ingestion can take 15–20 minutes for 4 PDFs.

**Option B – Manual steps**

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
   .venv/bin/python -m src.evaluation.evaluation_runner          # full 100 queries
   .venv/bin/python -m src.evaluation.evaluation_runner --limit=5  # quick test
   .venv/bin/python -m src.evaluation.evaluation_runner --rag --ragas  # add RAGAS (faithfulness, answer_relevancy); requires pip install -r requirements-ragas.txt
   ```

   Runs queries from `src/evaluation/evaluation_dataset.json` and reports retrieval recall@5, precision@5, mean latency, cost. With `--ragas` (and RAGAS installed), also reports **ragas_faithfulness** and **ragas_answer_relevancy**. See **Evaluation metrics** below and `src/evaluation/README.md` for current numbers and the process of improving retrieval.

## Example Queries

- Does the Bronze plan cover emergency room visits?
- What is the deductible for the Gold plan?
- Compare prescription drug coverage between Silver and Platinum.
- Which plan has the lowest out-of-pocket maximum?

## Documentation

| Document | Purpose |
|----------|---------|
| **[docs/DEMO.md](docs/DEMO.md)** | Interview demo & recruiter guide (quick demo, talking points, code map) |
| **[docs/PIPELINE_WORKFLOW.md](docs/PIPELINE_WORKFLOW.md)** | End-to-end pipeline (ingestion, query, evaluation) |
| **[src/evaluation/README.md](src/evaluation/README.md)** | Evaluation metrics and how to improve retrieval |
| **[docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md)** | Production checklist, security, testing, deployment |

## System Metrics

- **Latency:** per-query and percentiles (p50, p95, p99) via `LatencyTracker` and `scripts/compute_latency_stats`.
- **Cost:** per-query cost from OpenAI token usage; logged in `MetricsLogger`.
- **Observability:** every query logged to JSONL and SQLite; optional LangSmith via `LANGCHAIN_API_KEY`.

## Evaluation metrics

All metrics produced by the evaluation runner:

| Metric | When | Description |
|--------|------|-------------|
| **retrieval_recall_at_5** | Always | Fraction of queries where ≥1 of top-5 chunks matches expected section/plan. |
| **retrieval_precision_at_5** | Always | Fraction of top-5 chunks that match expected section/plan. |
| **mean_latency_ms** | Always | Average latency per query (retrieval; + generation if `--rag`). |
| **total_cost** | With `--rag` | Sum of LLM cost over evaluated queries. |
| **num_queries** | Always | Number of queries evaluated. |
| **ragas_faithfulness** | With `--rag --ragas` | RAGAS: answer grounded in context (0–1). |
| **ragas_answer_relevancy** | With `--rag --ragas` | RAGAS: answer addresses the question (0–1). |

Latest **100-query** retrieval-only run:

| Metric | Value |
|--------|--------|
| retrieval_recall_at_5 | 0.23 |
| retrieval_precision_at_5 | 0.056 |
| mean_latency_ms | ~12,000 |
| total_cost | 0 |
| num_queries | 100 |

We improved from an earlier baseline (recall@5 ≈ 0.125, precision@5 ≈ 0.025) by smaller chunking (450/120), hybrid vector + BM25 retrieval (top 20+20 → merge 40), weighted scoring (0.7 vector + 0.3 keyword), and reranking 40 → top 5. For the full improvement process and how to push metrics further, see **`src/evaluation/README.md`**. For end-to-end workflow and pipeline details, see **`docs/PIPELINE_WORKFLOW.md`**.

## Future Improvements

- Full LangSmith tracing for retrieval and generation.
- Sparse BM25 is already used (see `src/retrieval/bm25_index.py`).
- Evaluation with answer correctness (e.g. LLM-as-judge).
- Caching for repeated queries and embedding reuse.

For a checklist of **what’s left for enterprise production** and **what you can do on free tier**, see **`docs/PRODUCTION_READINESS.md`** (security, reliability, observability, testing, deployment, and free-tier options).

### Production features (implemented)

- **Security:** Optional API key auth (`API_KEY` in `.env`; send `X-API-Key` or `Authorization: Bearer <key>`), rate limiting (e.g. `RATE_LIMIT_PER_MINUTE`), request validation (max question length).
- **Reliability:** Health check (`GET /health` checks Qdrant), retries with backoff for Qdrant search, timeouts for Qdrant and OpenAI, graceful shutdown.
- **Observability:** Structured request logging (method, path, status, latency), optional Slack/Discord alerting on 5xx (`SLACK_WEBHOOK_URL`, `DISCORD_WEBHOOK_URL`).
- **Testing:** `pytest tests/` (unit + API tests); see `tests/`.
- **CI:** GitHub Actions (`.github/workflows/ci.yml`) runs tests on push/PR; optional quick evaluation on main when secrets are set.
- **Deployment:** `Dockerfile` for the API; `.env.example` lists all optional env vars.
