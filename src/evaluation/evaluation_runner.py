"""
Run evaluation: load dataset, run retrieval (and optional RAG) for each query, compute metrics.
Do NOT run automatically; user runs this script after setting API keys.
"""

import json
from pathlib import Path
from typing import Any, List

from src.evaluation.metrics import compute_evaluation_metrics
from src.utils.config import EVAL_DATASET_PATH
from src.utils.logger import get_logger
from src.rag.rag_pipeline import RAGPipeline

logger = get_logger(__name__)


def load_evaluation_dataset(path: Path | None = None) -> List[dict]:
    path = path or EVAL_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    dataset_path: Path | None = None,
    run_full_rag: bool = True,
    limit: int | None = None,
    use_ragas: bool = False,
) -> dict[str, Any]:
    """
    For each item in evaluation_dataset.json:
    - Run retrieval once, then optionally run full RAG (generate answer).
    - Record retrieved chunks, latency, cost.
    Then compute recall@5, precision@5, mean latency, total cost.
    If use_ragas and run_full_rag, also compute RAGAS metrics (faithfulness, answer_relevancy) when ragas is installed.
    If limit is set, only run the first limit queries (for quick tests).
    """
    import time
    dataset = load_evaluation_dataset(dataset_path)
    if limit is not None and limit > 0:
        dataset = dataset[:limit]
        logger.info("Running evaluation on first %d queries", len(dataset))
    pipeline = RAGPipeline()
    results: List[dict] = []
    for i, item in enumerate(dataset):
        query = item.get("query", "")
        if not query:
            continue
        try:
            start = time.perf_counter()
            chunks = pipeline.retriever.retrieve(query=query, use_rerank=True)
            latency_ms = (time.perf_counter() - start) * 1000
            cost = 0.0
            answer = ""
            if run_full_rag and chunks:
                gen_start = time.perf_counter()
                out = pipeline.answer_generator.generate(query=query, chunks=chunks)
                answer = out.get("answer", "")
                usage = out.get("usage", {})
                from src.monitoring.cost_tracker import CostTracker
                cost = CostTracker.compute_cost(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )
                latency_ms += (time.perf_counter() - gen_start) * 1000
            results.append({
                "query": query,
                "retrieved_chunks": chunks,
                "latency_ms": latency_ms,
                "cost": cost,
                "answer": answer,
            })
        except Exception as e:
            logger.exception("Evaluation query failed: %s", query)
            results.append({
                "query": query,
                "retrieved_chunks": [],
                "latency_ms": 0,
                "cost": 0,
                "error": str(e),
            })
    # Align with dataset (only queries that were in dataset)
    eval_entries = [item for item in dataset if item.get("query")]
    while len(results) < len(eval_entries):
        results.append({"query": "", "retrieved_chunks": [], "latency_ms": 0, "cost": 0})
    results = results[: len(eval_entries)]
    metrics = compute_evaluation_metrics(results, eval_entries)
    if use_ragas and run_full_rag:
        try:
            from src.evaluation.ragas_metrics import compute_ragas_metrics
            ragas_scores = compute_ragas_metrics(results)
            if ragas_scores:
                metrics.update(ragas_scores)
                logger.info("RAGAS metrics: %s", ragas_scores)
        except Exception as e:
            logger.warning("RAGAS metrics skipped: %s (install with: pip install -r requirements-ragas.txt)", e)
    return {
        "metrics": metrics,
        "results": results,
    }


def main():
    """Entrypoint for scripts; prints metrics. Do not run automatically."""
    import sys
    run_rag = "--rag" in sys.argv
    use_ragas = "--ragas" in sys.argv
    limit = None
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            try:
                limit = int(arg.split("=", 1)[1])
            except ValueError:
                pass
            break
    out = run_evaluation(run_full_rag=run_rag, limit=limit, use_ragas=use_ragas)
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
