"""
Unit tests for evaluation metrics.
"""

import pytest

from src.evaluation.metrics import (
    compute_evaluation_metrics,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)


def test_recall_at_k_match():
    chunks = [
        {"section": "Deductible", "plan": "Gold"},
        {"section": "Emergency", "plan": "Silver"},
    ]
    assert retrieval_recall_at_k(chunks, "Deductible", "Gold", k=5) == 1.0
    assert retrieval_recall_at_k(chunks, "Emergency", "Silver", k=5) == 1.0


def test_recall_at_k_no_match():
    chunks = [
        {"section": "Deductible", "plan": "Gold"},
        {"section": "Emergency", "plan": "Silver"},
    ]
    assert retrieval_recall_at_k(chunks, "Prescription", "Gold", k=5) == 0.0
    assert retrieval_recall_at_k(chunks, "Deductible", "Silver", k=5) == 0.0


def test_recall_at_k_no_expected_section():
    assert retrieval_recall_at_k([], None, None, k=5) == 1.0


def test_recall_at_k_expected_plans():
    chunks = [{"section": "Prescription", "plan": "Platinum"}]
    assert retrieval_recall_at_k(chunks, "Prescription", None, expected_plans=["Silver", "Platinum"], k=5) == 1.0
    assert retrieval_recall_at_k(chunks, "Prescription", None, expected_plans=["Bronze"], k=5) == 0.0


def test_precision_at_k():
    chunks = [
        {"section": "Deductible", "plan": "Gold"},
        {"section": "Emergency", "plan": "Silver"},
        {"section": "Deductible", "plan": "Gold"},
    ]
    assert retrieval_precision_at_k(chunks, "Deductible", "Gold", k=3) == pytest.approx(2 / 3)
    assert retrieval_precision_at_k(chunks, "Deductible", None, k=3) == pytest.approx(2 / 3)


def test_compute_evaluation_metrics():
    results = [
        {"retrieved_chunks": [{"section": "Deductible", "plan": "Gold"}], "latency_ms": 100, "cost": 0},
        {"retrieved_chunks": [{"section": "Emergency", "plan": "Silver"}], "latency_ms": 200, "cost": 0},
    ]
    dataset = [
        {"query": "q1", "expected_section": "Deductible", "expected_plan": "Gold"},
        {"query": "q2", "expected_section": "Emergency", "expected_plan": "Silver"},
    ]
    out = compute_evaluation_metrics(results, dataset)
    assert out["num_queries"] == 2
    assert out["retrieval_recall_at_5"] == 1.0
    assert out["retrieval_precision_at_5"] == 1.0
    assert out["mean_latency_ms"] == 150.0
    assert out["total_cost"] == 0
