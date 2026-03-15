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

## What’s remaining for enterprise production (and free-tier options)

Everything below can be implemented in code or with free-tier services unless noted. **Free** = $0 and implementable now with open-source or free-tier offerings.

### Security

| Item | Free tier? | How |
|------|------------|-----|
| **API authentication** (API key or OAuth) | ✅ Free | Code: validate `X-API-Key` or `Authorization` header against a secret from env. No external service required. |
| **OAuth / SSO** (optional) | ✅ Free tier | Auth0, Clerk, or Keycloak (self-hosted) have free tiers for limited MAU. |
| **Rate limiting** | ✅ Free | Code: e.g. `slowapi` (pip) or in-memory/Redis per-IP or per-key limits. |
| **Input validation & sanitization** | ✅ Free | You already use Pydantic; add max length on `question`, strip/escape if you render user input in UI. |
| **Secrets in env / vault** | ✅ Free | Keep using `.env`; for “vault” you can use GitHub Actions secrets or free tier of Doppler/HashiCorp Vault (limits apply). |

### Reliability

| Item | Free tier? | How |
|------|------------|-----|
| **Retries with backoff** (Qdrant, embedder, LLM) | ✅ Free | Code: `tenacity` or a small retry loop with exponential backoff. |
| **Timeouts** (all outbound calls) | ✅ Free | Code: set `timeout` on Qdrant client, HTTP calls, and LLM client. |
| **Health check endpoint** | ✅ Free | Code: e.g. `GET /health` that pings Qdrant (and optionally embedder); return 503 if unhealthy. |
| **Circuit breaker** (optional) | ✅ Free | Code: e.g. `pybreaker` or simple “fail after N errors, then skip for T seconds”. |
| **Graceful shutdown** | ✅ Free | Code: FastAPI lifespan or signal handler to drain in-flight requests before exit. |

### Observability

| Item | Free tier? | How |
|------|------------|-----|
| **Structured logging** (JSON, levels) | ✅ Free | Code: use `structlog` or ensure log lines are JSON with a consistent schema. |
| **Tracing** (OpenTelemetry) | ✅ Free | Code: `opentelemetry-api` + instrument FastAPI and HTTP/DB; export to Jaeger (self-hosted or free tier) or Zipkin. |
| **Alerting** (e.g. latency spike, errors) | ✅ Free tier | Free: webhook to Slack/Discord/email; or Uptime Kuma (self-hosted). Paid: PagerDuty/Opsgenie free tiers (limits). |
| **Dashboards** (latency, error rate, cost) | ✅ Free tier | Grafana Cloud free tier, or self-hosted Grafana + SQLite/JSONL (or export metrics to Prometheus). |

### Testing & CI

| Item | Free tier? | How |
|------|------------|-----|
| **Unit tests** (pytest) | ✅ Free | Code: pytest; mock embedder/Qdrant/LLM; run locally and in CI. |
| **Integration tests** (one full query path) | ✅ Free | Code: pytest with real .env or test containers; optional `testcontainers` (Docker). |
| **CI** (run tests + lint on push/PR) | ✅ Free | GitHub Actions: 2,000 min/month free for private repos; run pytest and `evaluation_runner --limit=5` (or full eval on schedule). |
| **Evaluation in CI** | ✅ Free | Same as above; full 100-query run is slow (~20 min) so use `--limit=5` or `--limit=10` on PR and full run on main/nightly. |

### Deployment & scaling

| Item | Free tier? | How |
|------|------------|-----|
| **Dockerfile** (API + UI) | ✅ Free | Code: multi-stage Dockerfile; no cost. |
| **Container registry** | ✅ Free | GitHub Container Registry (GHCR) or Docker Hub free tier. |
| **Hosting** (run the app) | ✅ Free tier | Railway, Render, Fly.io, Hugging Face Spaces (Streamlit) have free tiers; limits on RAM/hours. Your Qdrant/OpenAI/HF usage is separate. |
| **Secrets in production** | ✅ Free | Env vars on the host or in the platform (Railway/Render/Fly.io all support env secrets for free). |
| **Horizontal scaling** (multiple API replicas) | ⚠️ Depends | Free tiers usually allow 1 instance; scaling out = paid or self-hosted (e.g. Docker Compose or K8s on your own machine). |
| **Caching** (e.g. repeated queries) | ✅ Free tier | In-memory dict; or Redis via Upstash Redis free tier / Redis Cloud free tier. |

### What usually isn’t free (or is limited)

| Item | Note |
|------|------|
| **Qdrant** | You’re already on cloud; they have a free tier with size limits. |
| **OpenAI** | Pay-per-token; no “free production” tier for high volume. |
| **Embedding/reranker** | Self-hosted (BGE) = free compute; API-based = paid. |
| **24/7 uptime SLA** | Free hosting tiers typically don’t guarantee SLA; enterprise = paid. |
| **Dedicated support** | Enterprise support = paid. |

---

## How much can you implement on free tier now?

Rough split:

- **Fully free (code + free-tier tools):** Most of **security** (API key auth, rate limiting, validation), **reliability** (retries, timeouts, health check, graceful shutdown), **observability** (structured logs, optional tracing, webhook alerts, Grafana free tier), **testing** (pytest, CI with GitHub Actions), and **deployment** (Dockerfile, GHCR, one app on Railway/Render/Fly.io/HF Spaces). That’s the majority of “enterprise-style” hardening.
- **Free tier with limits:** Hosting (one instance, sleep after idle), Redis (small storage), OAuth (limited MAU), alerting (limited incidents). Enough for a serious MVP or internal tool.
- **Paid or self-hosted when you need more:** Multiple regions, many replicas, 24/7 SLA, large Qdrant/OpenAI usage, dedicated support.

**Practical order to implement on free tier:** (1) API key auth + rate limiting, (2) health check + retries/timeouts, (3) Dockerfile + deploy to Railway or Render, (4) pytest + GitHub Actions CI, (5) structured logging + optional tracing/alerting.
