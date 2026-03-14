"""
End-to-end RAG pipeline: embed -> parallel retrieval -> batch rerank -> compress -> generate.
Per-stage timing: retrieval_ms, rerank_ms, compression_ms, generation_ms.
"""

import time
from typing import Any, Optional

from src.monitoring.cost_tracker import CostTracker
from src.monitoring.latency_tracker import LatencyTracker
from src.monitoring.metrics_logger import MetricsLogger
from src.rag.answer_generator import AnswerGenerator
from src.rag.context_compressor import compress_context, estimate_tokens
from src.rag.plan_comparator import PlanComparator
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _log_stage_timings(stages: dict[str, float]) -> None:
    """Print [Latency] stage timings in seconds."""
    parts = [f"{k}: {v/1000:.2f}s" for k, v in stages.items()]
    logger.info("[Latency] %s", " | ".join(parts))


class RAGPipeline:
    """
    User Query -> embed -> retrieval -> rerank (batch) -> compress -> generate.
    Tracks per-stage latency and total.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        answer_generator: AnswerGenerator | None = None,
        plan_comparator: PlanComparator | None = None,
        metrics_logger: MetricsLogger | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.answer_generator = answer_generator or AnswerGenerator()
        self.plan_comparator = plan_comparator or PlanComparator(retriever=self.retriever)
        self.metrics_logger = metrics_logger or MetricsLogger()

    def query(
        self,
        question: str,
        plan_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run RAG with per-stage timing: retrieval -> rerank -> compression -> generation.
        """
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve_candidates(query=question, plan_filter=plan_filter)
        t1 = time.perf_counter()
        retrieval_ms = (t1 - t0) * 1000

        chunks = self.retriever.rerank_candidates(query=question, candidates=candidates)
        t2 = time.perf_counter()
        rerank_ms = (t2 - t1) * 1000

        retrieval_stats = {
            **getattr(self.retriever, "_last_retrieval_stats", {}),
            "final_context_chunks": len(chunks),
        }
        logger.info(
            "[Retrieval] vector_candidates=%s keyword_candidates=%s reranked_candidates=%s final_context_chunks=%s",
            retrieval_stats.get("vector_candidates", 0),
            retrieval_stats.get("keyword_candidates", 0),
            retrieval_stats.get("reranked_candidates", 0),
            retrieval_stats.get("final_context_chunks", 0),
        )

        chunk_texts = [(c.get("text") or "").strip() for c in chunks]
        original_context = "\n\n".join(chunk_texts)
        compressed_context = compress_context(chunk_texts)
        t3 = time.perf_counter()
        compression_ms = (t3 - t2) * 1000
        original_token_estimate = estimate_tokens(original_context)
        compressed_token_estimate = estimate_tokens(compressed_context)
        logger.info(
            "context_compression original_token_estimate=%s compressed_token_estimate=%s",
            original_token_estimate,
            compressed_token_estimate,
        )

        result = self.answer_generator.generate(
            query=question, chunks=chunks, compressed_context=compressed_context
        )
        t4 = time.perf_counter()
        generation_ms = (t4 - t3) * 1000
        total_ms = (t4 - t0) * 1000

        stages = {
            "retrieval": retrieval_ms,
            "rerank": rerank_ms,
            "compression": compression_ms,
            "generation": generation_ms,
            "total": total_ms,
        }
        _log_stage_timings(stages)

        usage = result.get("usage", {})
        cost = CostTracker.compute_cost(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
        result["latency_ms"] = round(total_ms, 2)
        result["retrieval_latency_ms"] = round(retrieval_ms, 2)
        result["generation_latency_ms"] = round(generation_ms, 2)
        result["cost"] = cost
        result["usage"] = usage
        result["stage_timings_ms"] = {k: round(v, 2) for k, v in stages.items()}

        self.metrics_logger.log_query(
            query=question,
            latency_ms=total_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cost=cost,
            original_token_estimate=original_token_estimate,
            compressed_token_estimate=compressed_token_estimate,
            rerank_ms=rerank_ms,
            compression_ms=compression_ms,
            retrieval_stats=retrieval_stats,
        )
        LatencyTracker.record(total_ms, stage_timings=stages)
        return result

    def compare_plans(
        self,
        question: str,
        plans: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Compare plans: parallel retrieval -> batch rerank -> compress -> generate.
        """
        t0 = time.perf_counter()
        comparison = self.plan_comparator.compare(query=question, plans=plans)
        t1 = time.perf_counter()
        # compare() does parallel retrieval + single batch rerank; we don't split retrieval/rerank inside it
        retrieval_rerank_ms = (t1 - t0) * 1000
        all_chunks = comparison.get("all_chunks", [])

        chunk_texts = [(c.get("text") or "").strip() for c in all_chunks]
        original_context = "\n\n".join(chunk_texts)
        compressed_context = compress_context(chunk_texts)
        t2 = time.perf_counter()
        compression_ms = (t2 - t1) * 1000
        original_token_estimate = estimate_tokens(original_context)
        compressed_token_estimate = estimate_tokens(compressed_context)
        logger.info(
            "context_compression original_token_estimate=%s compressed_token_estimate=%s",
            original_token_estimate,
            compressed_token_estimate,
        )

        result = self.answer_generator.generate(
            query=question, chunks=all_chunks, compressed_context=compressed_context
        )
        t3 = time.perf_counter()
        generation_ms = (t3 - t2) * 1000
        total_ms = (t3 - t0) * 1000

        stages = {
            "retrieval": retrieval_rerank_ms,
            "rerank": 0,
            "compression": compression_ms,
            "generation": generation_ms,
            "total": total_ms,
        }
        _log_stage_timings(stages)

        usage = result.get("usage", {})
        cost = CostTracker.compute_cost(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        result["latency_ms"] = round(total_ms, 2)
        result["cost"] = cost
        result["comparison_data"] = comparison
        result["stage_timings_ms"] = {k: round(v, 2) for k, v in stages.items()}

        self.metrics_logger.log_query(
            query=question,
            latency_ms=total_ms,
            retrieval_latency_ms=retrieval_rerank_ms,
            generation_latency_ms=generation_ms,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cost=cost,
            original_token_estimate=original_token_estimate,
            compressed_token_estimate=compressed_token_estimate,
            rerank_ms=0,
            compression_ms=compression_ms,
        )
        LatencyTracker.record(total_ms, stage_timings=stages)
        return result
