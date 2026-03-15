# Pipeline Workflow & In-Depth Explanation

This document describes the end-to-end workflow of InsuranceCopilot AI: from PDF ingestion to answering user questions, plus evaluation and deployment. **For interview demos and recruiter review,** see **[DEMO.md](DEMO.md)** first.

---

## 1. High-Level Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         InsuranceCopilot AI – Full Workflow                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  [1] INGESTION (one-time)                                                         │
│      data/*.pdf → Docling parser → SectionChunker → BGE embed → Qdrant + BM25     │
│                                                                                   │
│  [2] QUERY (per request)                                                          │
│      User question → Embed → Hybrid (vector 20 + BM25 20) → Merge 40              │
│                    → Rerank (bge-reranker) → Top 5 → Context compress → LLM       │
│                    → Answer + citations                                            │
│                                                                                   │
│  [3] EVALUATION (optional)                                                        │
│      evaluation_dataset.json → retrieval/RAG per query → metrics (recall, RAGAS)  │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Ingestion Pipeline (Offline, One-Time)

**Entrypoint:** `python -m scripts.ingest_documents`  
**Purpose:** Turn policy PDFs into searchable chunks in Qdrant and a BM25 index.

### 2.1 Steps

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **PDF discovery** | `get_pdf_paths()` collects all `*.pdf` from `data/` (e.g. `bronze.pdf`, `silver.pdf`, `gold.pdf`, `platinum.pdf`). |
| 2 | **Parse** | `parse_pdf_with_docling(path)` (Docling) extracts text and structure (sections, tables) per PDF. |
| 3 | **Metadata** | `extract_metadata_from_path(path)` infers `plan_name` (Bronze/Silver/Gold/Platinum) and `source_file` from filename. |
| 4 | **Chunk** | `SectionChunker.chunk_document(sections, plan_name, source_file)` splits each section into chunks of **450 tokens** with **120 token overlap**, respecting section boundaries. Output: list of `{ text, plan, section, page, ... }`. |
| 5 | **Embed** | `IndexBuilder` uses **BGE** (`BAAI/bge-large-en`) to embed all chunk texts in batches (default 32). |
| 6 | **Vector store** | Vectors + payloads (plan, section, page, text) are upserted into **Qdrant** (collection created if missing; recreated if vector size mismatches). |
| 7 | **BM25 index** | `build_and_save(ids, payloads, BM25_INDEX_PATH)` builds a BM25 index over chunk texts and saves to `data/bm25_index.pkl` for keyword retrieval. |

### 2.2 Key Configuration (ingestion)

- **Chunk size / overlap:** `CHUNK_SIZE_TOKENS` (450), `CHUNK_OVERLAP_TOKENS` (120) in `src/utils/config.py`.
- **Embedding model:** BGE via `src/embeddings/embedder.py` (dimension 1024).
- **Qdrant:** `QDRANT_URL`, `QDRANT_API_KEY` in `.env`.

---

## 3. Query Pipeline (Per Request)

**Entrypoints:** CLI (`python -m scripts.run_pipeline`), API (`run_pipeline api`), UI (`run_pipeline ui`).  
**Core class:** `RAGPipeline` in `src/rag/rag_pipeline.py`.

### 3.1 Single-Question Flow (`RAGPipeline.query`)

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **Retrieve candidates** | `Retriever.retrieve_candidates(query)` runs **hybrid search**: embed query with BGE → vector search (top 20) + BM25 search (top 20) → merge by chunk id, deduplicate → hybrid score = `VECTOR_WEIGHT`×vector_score + `KEYWORD_WEIGHT`×keyword_score (default 0.7 / 0.3), optional section-metadata boost → take top **40** (RERANK_POOL_SIZE). |
| 2 | **Rerank** | `Retriever.rerank_candidates(query, candidates)` passes the 40 candidates to **bge-reranker-large** in batches; returns **top 5** (TOP_K_AFTER_RERANK). |
| 3 | **Compress context** | `compress_context(chunk_texts)` shortens the top-5 text for the LLM (sentence-level truncation / token estimate) to stay within context limits. |
| 4 | **Generate** | `AnswerGenerator.generate(query, chunks, compressed_context)` builds a system prompt (“answer only from context, cite plan/section/page”) and user message (policy context + question), calls **OpenAI** (or configured LLM), returns `{ answer, citations, usage }`. |
| 5 | **Log & return** | Latency (per-stage and total), token usage, and cost are logged via `MetricsLogger` and `LatencyTracker`; response includes answer, citations, latency_ms, cost. |

### 3.2 Plan Comparison Flow (`RAGPipeline.compare_plans`)

- For each selected plan (default: Bronze, Silver, Gold, Platinum), the pipeline runs **retrieve_candidates → rerank → top 5** (optionally with plan filter).
- Contexts are concatenated with plan labels; then **one** LLM call generates a comparison answer.
- Same logging and cost tracking as single-query.

### 3.3 Key Configuration (query)

- **Retrieval:** `VECTOR_TOP_K`, `KEYWORD_TOP_K` (20 each), `RERANK_POOL_SIZE` (40), `TOP_K_AFTER_RERANK` (5), `VECTOR_WEIGHT`, `KEYWORD_WEIGHT` in `src/utils/config.py`.
- **LLM:** `LLM_MODEL`, `OPENAI_API_KEY`; timeouts in `HTTP_TIMEOUT_SECONDS`.
- **Monitoring:** Logs to `logs/query_metrics.jsonl` and `logs/metrics.db`; optional LangSmith via `LANGCHAIN_API_KEY`.

---

## 4. Evaluation Pipeline

**Entrypoint:** `python -m src.evaluation.evaluation_runner [--limit=N] [--rag] [--ragas]`  
**Purpose:** Measure retrieval quality and, optionally, answer quality using a fixed dataset.

### 4.1 Data

- **Dataset:** `src/evaluation/evaluation_dataset.json` — 100 items with `query`, `expected_section`, `expected_plan` or `expected_plans`.
- **Ground truth:** A chunk is “correct” if its section (and plan when specified) matches the expected section/plan.

### 4.2 Flow

1. Load dataset; optionally restrict to first `N` with `--limit=N`.
2. For each item: run retrieval (and optionally full RAG with `--rag`); record `retrieved_chunks`, `latency_ms`, `cost`, and `answer` if RAG.
3. **Retrieval metrics:** `compute_evaluation_metrics()` computes:
   - **retrieval_recall_at_5** — fraction of queries where ≥1 of top-5 chunks matches expected section/plan.
   - **retrieval_precision_at_5** — fraction of top-5 chunks that match.
   - **mean_latency_ms**, **total_cost**, **num_queries**.
4. If `--ragas` and `--rag`: `compute_ragas_metrics()` runs RAGAS (requires `pip install -r requirements-ragas.txt` and `OPENAI_API_KEY`) and adds **ragas_faithfulness** and **ragas_answer_relevancy** to the reported metrics.

### 4.3 Metrics Summary

| Metric | When | Description |
|--------|------|-------------|
| retrieval_recall_at_5 | Always | Fraction of queries with ≥1 correct chunk in top 5. |
| retrieval_precision_at_5 | Always | Fraction of top-5 chunks that are correct. |
| mean_latency_ms | Always | Average latency per query (retrieval + optional generation). |
| total_cost | With --rag | Sum of LLM cost over queries. |
| num_queries | Always | Number of queries evaluated. |
| ragas_faithfulness | With --rag --ragas | Answer grounded in retrieved context (0–1). |
| ragas_answer_relevancy | With --rag --ragas | Answer addresses the question (0–1). |

---

## 5. API & UI

- **API:** FastAPI app in `src/api/app.py`. Endpoints: `POST /query`, `POST /compare`, `GET /metrics`, `GET /health`. Optional API key (`API_KEY`), rate limiting, and Slack/Discord alerting on 5xx.
- **UI:** Streamlit in `src/ui/`; launched with `python -m scripts.run_pipeline ui` (or `scripts/run_pipeline ui`). Provides search bar and plan comparison.

---

## 6. File Map (Pipeline-Critical)

| Path | Role |
|------|------|
| `scripts/ingest_documents.py` | Ingestion entry: PDFs → chunks → Qdrant + BM25. |
| `scripts/run_pipeline.py` | CLI/API/UI entry; imports `RAGPipeline`. |
| `src/ingestion/pdf_parser.py` | Docling PDF parsing. |
| `src/ingestion/chunker.py` | Section-aware chunking (450/120 tokens). |
| `src/vector_store/index_builder.py` | Embed + upsert Qdrant + build BM25. |
| `src/vector_store/qdrant_client.py` | Qdrant connection, upsert, search. |
| `src/embeddings/embedder.py` | BGE embeddings. |
| `src/retrieval/hybrid_search.py` | Vector + BM25 merge and scoring. |
| `src/retrieval/reranker.py` | bge-reranker-large. |
| `src/rag/rag_pipeline.py` | Orchestrates retrieve → rerank → compress → generate. |
| `src/rag/answer_generator.py` | LLM call with citations. |
| `src/evaluation/evaluation_runner.py` | Run eval; optional RAG and RAGAS. |
| `src/evaluation/metrics.py` | recall@5, precision@5, latency, cost. |
| `src/evaluation/ragas_metrics.py` | RAGAS faithfulness & answer relevancy. |
| `src/utils/config.py` | All config (chunk size, weights, paths, API keys). |

---

## 7. Improving the Pipeline

- **Retrieval:** Tune chunk size/overlap, vector/keyword weights, rerank pool size; align `expected_section` in the eval set with actual section titles from your PDFs.
- **Answer quality:** Improve prompts in `AnswerGenerator`; use RAGAS (and optionally more eval questions) to track faithfulness and relevancy.
- **Production:** See `docs/PRODUCTION_READINESS.md` for security, reliability, observability, testing, and deployment.
