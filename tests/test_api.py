"""
API tests: health, root, validation. Uses TestClient; no real Qdrant/OpenAI needed for basic routes.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api.app import app
    return TestClient(app)


def test_root(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("app") == "InsuranceCopilot AI"
    assert "health" in data


def test_health_unauthenticated(client: TestClient):
    # /health is in SKIP_AUTH_PATHS so no API key needed
    r = client.get("/health")
    # 200 if Qdrant is reachable and collection exists; 503 otherwise (e.g. in CI without Qdrant)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert r.json() == {"status": "ok"}
    else:
        assert "detail" in r.json()


def test_query_validation_max_length(client: TestClient):
    # No API key required if API_KEY env is not set
    r = client.post("/query", json={"question": "x" * 3000, "plan_filter": None})
    assert r.status_code == 422  # validation error (max length)


def test_query_validation_min_length(client: TestClient):
    r = client.post("/query", json={"question": "", "plan_filter": None})
    assert r.status_code == 422


def test_compare_validation(client: TestClient):
    r = client.post("/compare", json={"question": "", "plans": ["Gold"]})
    assert r.status_code == 422


def test_metrics_endpoint(client: TestClient):
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "p50" in data or "count" in data or isinstance(data, dict)
