"""
Evaluation metrics: retrieval recall@k, precision@k, latency, cost per query.
"""

from typing import Any, List


def _plan_matches(
    chunk: dict,
    expected_plan: str | None,
    expected_plans: List[str] | None,
) -> bool:
    """True if chunk's plan matches expected_plan or is in expected_plans."""
    if expected_plans:
        plan = (chunk.get("plan") or chunk.get("plan_name") or "").lower()
        return any(p.lower() in plan for p in expected_plans if p)
    if expected_plan:
        plan = (chunk.get("plan") or chunk.get("plan_name") or "").lower()
        return expected_plan.lower() in plan
    return True


def retrieval_recall_at_k(
    retrieved_chunks: List[dict],
    expected_section: str | None,
    expected_plan: str | None = None,
    expected_plans: List[str] | None = None,
    k: int = 5,
) -> float:
    """
    Recall@k: did we retrieve at least one chunk matching expected_section (and optionally plan)?
    Returns 1.0 if any of top-k has matching section (and plan if provided), else 0.0.
    """
    if not expected_section:
        return 1.0
    top = retrieved_chunks[:k]
    for c in top:
        section = (c.get("section") or c.get("section_title") or "").lower()
        if expected_section.lower() in section and _plan_matches(c, expected_plan, expected_plans):
            return 1.0
    return 0.0


def retrieval_precision_at_k(
    retrieved_chunks: List[dict],
    expected_section: str | None,
    expected_plan: str | None = None,
    expected_plans: List[str] | None = None,
    k: int = 5,
) -> float:
    """
    Precision@k: among top-k, what fraction match expected_section (and plan if provided)?
    """
    if not expected_section or not retrieved_chunks:
        return 1.0
    top = retrieved_chunks[:k]
    matches = 0
    for c in top:
        section = (c.get("section") or c.get("section_title") or "").lower()
        if expected_section.lower() in section and _plan_matches(c, expected_plan, expected_plans):
            matches += 1
    return matches / len(top) if top else 0.0


def compute_evaluation_metrics(
    results: List[dict[str, Any]],
    dataset: List[dict],
) -> dict:
    """
    results: list of { "recall_at_5", "precision_at_5", "latency_ms", "cost", "retrieved_chunks" } per item.
    dataset: evaluation_dataset entries with expected_section, expected_plan/expected_plans.
    """
    if len(results) != len(dataset):
        raise ValueError("results and dataset length must match")
    recalls = []
    precisions = []
    latencies = []
    costs = []
    for r, d in zip(results, dataset):
        chunks = r.get("retrieved_chunks", [])
        exp_sec = d.get("expected_section")
        exp_plan = d.get("expected_plan")
        exp_plans = d.get("expected_plans")
        recalls.append(retrieval_recall_at_k(chunks, exp_sec, exp_plan, exp_plans, k=5))
        precisions.append(retrieval_precision_at_k(chunks, exp_sec, exp_plan, exp_plans, k=5))
        latencies.append(r.get("latency_ms", 0))
        costs.append(r.get("cost", 0))
    n = len(recalls)
    return {
        "retrieval_recall_at_5": sum(recalls) / n if n else 0,
        "retrieval_precision_at_5": sum(precisions) / n if n else 0,
        "mean_latency_ms": sum(latencies) / n if n else 0,
        "total_cost": sum(costs),
        "num_queries": n,
    }
