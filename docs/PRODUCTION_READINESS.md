# Production Readiness

## Is the evaluation done?

- **Ingestion:** Done. 2,260 chunks (4 plans) are in Qdrant and the BM25 index is built (`data/bm25_index.pkl`).
- **Evaluation dataset:** 100 questions with `expected_section` / `expected_plan` ground truth in `src/evaluation/evaluation_dataset.json`.
- **Latest full run (100 queries, retrieval only):**

  | Metric | Value |
  |--------|--------|
  | retrieval_recall_at_5 | 0.23 |
  | retrieval_precision_at_5 | 0.056 |
  | mean_latency_ms | ~12,000 |
  | total_cost | 0 |
  | num_queries | 100 |

  With `--rag --ragas`, the runner also reports **ragas_faithfulness** and **ragas_answer_relevancy**. See **Metrics reference** in `src/evaluation/README.md` for all metrics. We improved from an earlier baseline (recall@5 ≈ 0.125, precision@5 ≈ 0.025) via smaller chunking, hybrid vector + BM25, weighted scoring, and rerank-40→top-5. The process of improving retrieval and how to push metrics further is documented in **`src/evaluation/README.md`**. End-to-end pipeline workflow: **`docs/PIPELINE_WORKFLOW.md`**.
- **Running evaluation:** From project root:
  ```bash
  .venv/bin/python -m src.evaluation.evaluation_runner          # full 100 queries (~15–20 min)
  .venv/bin/python -m src.evaluation.evaluation_runner --limit=5  # quick check
  ```
  Output: `retrieval_recall_at_5`, `retrieval_precision_at_5`, `mean_latency_ms`, `total_cost`, `num_queries`.

## Is the project complete?

Yes, for the intended scope:

| Component | Status |
|-----------|--------|
| PDF ingestion (Docling, chunking 450/120) | Done |
| Vector store (Qdrant) + BM25 index | Done |
| Hybrid retrieval (vector 20 + BM25 20 → merge → rerank 40 → top 5) | Done |
| Reranker (bge-reranker-large), context compression | Done |
| RAG pipeline (query + plan comparison) | Done |
| FastAPI (POST /query, /compare, GET /metrics) | Done |
| Streamlit UI | Done |
| Monitoring (latency, cost, JSONL + SQLite) | Done |
| Evaluation (100-question set, recall@5, precision@5) | Done |
| Setup script (`scripts/setup_and_run.sh`) | Done |

## Production-ready RAG?

**Suitable for:** Demos, internal tools, MVP, and single-team deployment with controlled access.

**Not yet “enterprise production”** without extra work:

| Area | Current state | For stricter production |
|------|----------------|-------------------------|
| **Security** | API keys in `.env`, no secrets in repo | Add API auth (e.g. API key or OAuth), rate limiting, input validation/sanitization |
| **Reliability** | Basic error handling | Retries, timeouts, circuit breakers; health checks for Qdrant and embedder |
| **Observability** | Logs, latency, cost per query | Structured logging, tracing (e.g. OpenTelemetry), alerting, dashboards |
| **Scaling** | Single process | Optional: separate embed/rerank services, async workers, horizontal scaling |
| **Testing** | Evaluation script only | Unit tests, integration tests, CI runs evaluation on PRs |
| **Deployment** | Run via `uvicorn`, `streamlit` | Dockerfile, optional Kubernetes/ECS, env-based config, secrets manager |

**Bottom line:** The RAG pipeline is **feature-complete and runnable in production-like environments** (e.g. a single server or small deployment). For high-availability or regulated environments, add the security, reliability, and operational pieces below.

---

## What’s remaining for enterprise production 
### Security

| Item | How |
|------|-----|
| **API authentication** (API key or OAuth) | Code: validate `X-API-Key` or `Authorization` header against a secret from env. No external service required. |
| **OAuth / SSO** (optional) | Auth0, Clerk, or Keycloak (self-hosted). |
| **Rate limiting** | Code: e.g. `slowapi` (pip) or in-memory/Redis per-IP or per-key limits. |
| **Input validation & sanitization** | You already use Pydantic; add max length on `question`, strip/escape if you render user input in UI. |
| **Secrets in env / vault** | Keep using `.env`; for “vault” you can use GitHub Actions secrets or a secrets manager (e.g. Doppler, HashiCorp Vault). |

### Reliability

| Item | How |
|------|-----|
| **Retries with backoff** (Qdrant, embedder, LLM) | Code: `tenacity` or a small retry loop with exponential backoff. |
| **Timeouts** (all outbound calls) | Code: set `timeout` on Qdrant client, HTTP calls, and LLM client. |
| **Health check endpoint** | Code: e.g. `GET /health` that pings Qdrant (and optionally embedder); return 503 if unhealthy. |
| **Circuit breaker** (optional) | Code: e.g. `pybreaker` or simple “fail after N errors, then skip for T seconds”. |
| **Graceful shutdown** | Code: FastAPI lifespan or signal handler to drain in-flight requests before exit. |

### Observability

| Item | How |
|------|-----|
| **Structured logging** (JSON, levels) | Code: use `structlog` or ensure log lines are JSON with a consistent schema. |
| **Tracing** (OpenTelemetry) | Code: `opentelemetry-api` + instrument FastAPI and HTTP/DB; export to Jaeger or Zipkin. |
| **Alerting** (e.g. latency spike, errors) | Webhook to Slack/Discord/email; or Uptime Kuma (self-hosted); or PagerDuty/Opsgenie. |
| **Dashboards** (latency, error rate, cost) | Grafana (Cloud or self-hosted) + SQLite/JSONL, or export metrics to Prometheus. |

### Testing & CI

| Item | How |
|------|-----|
| **Unit tests** (pytest) | Code: pytest; mock embedder/Qdrant/LLM; run locally and in CI. |
| **Integration tests** (one full query path) | Code: pytest with real .env or test containers; optional `testcontainers` (Docker). |
| **CI** (run tests + lint on push/PR) | GitHub Actions: run pytest and `evaluation_runner --limit=5` (or full eval on schedule). |
| **Evaluation in CI** | Full 100-query run is slow (~20 min); use `--limit=5` or `--limit=10` on PR and full run on main/nightly. |

### Deployment & scaling

| Item | How |
|------|-----|
| **Dockerfile** (API + UI) | Code: multi-stage Dockerfile. |
| **Container registry** | GitHub Container Registry (GHCR) or Docker Hub. |
| **Hosting** (run the app) | Railway, Render, Fly.io, Hugging Face Spaces (Streamlit), or self-hosted. |
| **Secrets in production** | Env vars on the host or in the platform (Railway/Render/Fly.io support env secrets). |
| **Horizontal scaling** (multiple API replicas) | Multiple instances; Docker Compose or Kubernetes. |
| **Caching** (e.g. repeated queries) | In-memory dict; or Redis (e.g. Upstash, Redis Cloud). |

### Cost / limits to consider

| Item | Note |
|------|------|
| **Qdrant** | Cloud or self-hosted; size/throughput limits apply. |
| **OpenAI** | Pay-per-token. |
| **Embedding/reranker** | Self-hosted (BGE) = your compute; API-based = paid. |
| **24/7 uptime SLA** | Typically requires paid hosting. |
| **Dedicated support** | Enterprise support = paid. |

---
