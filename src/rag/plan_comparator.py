"""
Cross-plan comparison: parallel retrieval and batch reranking for multiple plans.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

PLANS_ORDER = ["Bronze", "Silver", "Gold", "Platinum"]


class PlanComparator:
    """
    Compare coverage across plans. Uses parallel retrieval (all plans at once)
    then a single batch rerank over merged candidates.
    """

    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()

    def compare(
        self,
        query: str,
        plans: Optional[List[str]] = None,
        top_k_per_plan: int = 3,
    ) -> dict[str, Any]:
        """
        Parallel retrieval per plan, merge candidates, batch rerank, then top_k_per_plan per plan.
        Returns: { "query", "plans", "chunks_by_plan", "all_chunks" }.
        """
        plans = plans or list(PLANS_ORDER)
        # Parallel retrieval: one hybrid search per plan (no rerank yet)
        def get_candidates(plan: str) -> List[dict]:
            return self.retriever.retrieve_candidates(query=query, plan_filter=plan)

        all_candidates: List[dict] = []
        with ThreadPoolExecutor(max_workers=len(plans)) as executor:
            future_to_plan = {executor.submit(get_candidates, p): p for p in plans}
            for future in as_completed(future_to_plan):
                try:
                    all_candidates.extend(future.result())
                except Exception as e:
                    logger.warning("Parallel retrieval failed for plan %s: %s", future_to_plan[future], e)

        if not all_candidates:
            return {
                "query": query,
                "plans": plans,
                "chunks_by_plan": {p: [] for p in plans},
                "all_chunks": [],
            }

        # Single batch rerank over all candidates; then take top_k_per_plan per plan
        top_total = len(plans) * top_k_per_plan
        reranked = self.retriever.rerank_candidates(
            query=query,
            candidates=all_candidates,
            top_k=min(top_total, len(all_candidates)),
        )
        chunks_by_plan: dict[str, List[dict]] = {p: [] for p in plans}
        for c in reranked:
            plan = c.get("plan") or c.get("plan_name", "")
            if plan in chunks_by_plan and len(chunks_by_plan[plan]) < top_k_per_plan:
                chunks_by_plan[plan].append(c)

        return {
            "query": query,
            "plans": plans,
            "chunks_by_plan": chunks_by_plan,
            "all_chunks": [c for chunks in chunks_by_plan.values() for c in chunks],
        }
