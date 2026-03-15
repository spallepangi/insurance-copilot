# Evaluation

## Metrics reference

All metrics the evaluation runner can report:

| Metric | When | Description |
|--------|------|-------------|
| **retrieval_recall_at_5** | Always | Fraction of queries where ≥1 of top-5 chunks matches expected section/plan. |
| **retrieval_precision_at_5** | Always | Fraction of top-5 chunks that match expected section/plan. |
| **mean_latency_ms** | Always | Average latency per query (retrieval; + generation if `--rag`). |
| **total_cost** | With `--rag` | Sum of LLM cost over evaluated queries. |
| **num_queries** | Always | Number of queries evaluated. |
| **ragas_faithfulness** | With `--rag --ragas` | RAGAS: answer grounded in retrieved context (0–1). |
| **ragas_answer_relevancy** | With `--rag --ragas` | RAGAS: answer addresses the question (0–1). |

## Current metrics (100-query run, retrieval only)

Latest full run on the 100-question set (retrieval only; no LLM):

| Metric | Value |
|--------|--------|
| **retrieval_recall_at_5** | 0.23 |
| **retrieval_precision_at_5** | 0.056 |
| **mean_latency_ms** | ~12,000 ms (~12 s per query) |
| **total_cost** | 0 |
| **num_queries** | 100 |

Recall@5 = fraction of queries where at least one of the top-5 retrieved chunks matches the expected section (and plan when specified). Precision@5 = fraction of the top-5 chunks that match. Latency includes embedding, hybrid search, and reranking.

## Process of improving retrieval

We improved retrieval from an initial baseline (recall@5 ≈ 0.125, precision@5 ≈ 0.025) by:

1. **Chunking** – Reduced chunk size from ~800 to **450 tokens** with **120 token overlap**. Smaller chunks improve embedding quality and section-level relevance; overlap preserves context at boundaries. Section boundaries are respected in `chunk_document`.
2. **Candidate pool** – **Vector top 20** + **BM25 top 20** → merge and deduplicate → up to **40 candidates** before reranking. This gives the reranker a larger, diverse pool (dense + sparse).
3. **Hybrid scoring** – Combined scores with **vector_weight = 0.7**, **keyword_weight = 0.3** (`hybrid_score = 0.7×vector + 0.3×keyword`), with min-max–normalized scores. Candidates are sorted by `hybrid_score` before reranking.
4. **Reranking** – Rerank the merged **40** with bge-reranker-large in batches, then select **top 5** for context. This focuses the final context on the most relevant chunks.
5. **Optional metadata boost** – If the query contains terms like *emergency*, *deductible*, *copay*, *prescription*, *coverage*, chunks whose section metadata contains the same term get a small score boost so section-specific questions favor the right sections.

**Ongoing improvements** (to push recall and precision higher):

- Align **expected_section** in the evaluation set with the exact section titles in your PDFs (parser output), so ground truth matches chunk metadata.
- Tune **VECTOR_WEIGHT** / **KEYWORD_WEIGHT** (e.g. in `src/utils/config.py`) for your query mix.
- Consider **query expansion** or **hybrid query** (e.g. add keywords from the query for BM25).
- Re-run evaluation after any change:  
  `.venv/bin/python -m src.evaluation.evaluation_runner`

---

## Test set (`evaluation_dataset.json`)

**100 questions** based on real insurance policy topics that appear in typical plan PDFs (e.g. Summary of Benefits, SBC). Each item has:

- **query** – Natural-language question.
- **expected_section** – Section title substring that should appear in at least one retrieved chunk (e.g. `"Deductible"`, `"Emergency"`, `"Prescription"`). Matches the `section` / `section_title` metadata from ingested chunks.
- **expected_plan** – Plan name (`"Bronze"`, `"Silver"`, `"Gold"`, `"Platinum"`) or `null` for cross-plan or “any plan” questions.
- **expected_plans** – Optional list of plans for comparison questions (e.g. `["Silver", "Platinum"]`). Recall/precision count a chunk as correct if its plan is in this list and section matches.

Topics covered: Deductible, Out-of-Pocket, Emergency, Primary Care, Specialist, Prescription, Preventive, Hospital, Urgent Care, Mental Health, Lab, Imaging, Eligibility, Exclusions, Appeals, Coverage, Summary, Network, Coinsurance, Maternity, Rehabilitation.

## Running evaluation

From the project root, with `.env` set and Qdrant + ingestion done. **Use the project venv** (e.g. `.venv/bin/python`) so NumPy &lt; 2 is used and the embedding model loads correctly:

```bash
# Quick test (first 10 queries only)
.venv/bin/python -m src.evaluation.evaluation_runner --limit=10

# Full 100 queries – retrieval only (recall@5, precision@5, mean latency; no LLM cost)
.venv/bin/python -m src.evaluation.evaluation_runner

# Full RAG (retrieval + answer generation; adds cost and generation latency)
.venv/bin/python -m src.evaluation.evaluation_runner --rag
```

Output: JSON with `retrieval_recall_at_5`, `retrieval_precision_at_5`, `mean_latency_ms`, `total_cost`, `num_queries`. Full 100-query run can take ~15–20 minutes. After changing retrieval (chunking, weights, rerank pool, etc.), re-run evaluation to track metrics.

### RAGAS (optional)

[RAGAS](https://github.com/explodinggradients/ragas) adds **answer-level** metrics using an LLM:

- **Faithfulness** – Is the answer grounded in the retrieved context? (no hallucination)
- **Answer relevancy** – Does the answer address the question?

Install and run:

```bash
pip install -r requirements-ragas.txt
.venv/bin/python -m src.evaluation.evaluation_runner --rag --ragas
```

Requires `OPENAI_API_KEY` (RAGAS uses the LLM to score). Output will include `ragas_faithfulness` and `ragas_answer_relevancy` in the metrics JSON. Use `--limit=5` or `--limit=10` for a quick RAGAS run (it calls the LLM per sample).
