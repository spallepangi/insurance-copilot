"""
FastAPI backend for InsuranceCopilot AI.
Endpoints: query, compare plans, health, metrics.
Production: API key auth (optional), rate limiting, validation, health check, alerting.
"""

from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.middleware import APIKeyMiddleware, StructuredLoggingMiddleware
from src.monitoring.alerting import send_alert
from src.monitoring.latency_tracker import LatencyTracker
from src.rag.rag_pipeline import RAGPipeline
from src.utils.config import QUERY_MAX_LENGTH, RATE_LIMIT_PER_MINUTE
from src.utils.logger import get_logger
from src.vector_store.qdrant_client import QdrantStore

logger = get_logger(__name__)

# Global pipeline (lazy init to avoid loading keys at import)
_pipeline: RAGPipeline | None = None
_shutting_down: bool = False


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def _check_qdrant() -> bool:
    """Return True if Qdrant is reachable and collection exists."""
    try:
        store = QdrantStore()
        return store.collection_exists()
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _shutting_down
    _shutting_down = False
    yield
    _shutting_down = True
    logger.info("Shutting down; in-flight requests will complete.")


app = FastAPI(
    title="InsuranceCopilot AI",
    description="Production RAG API for healthcare insurance policy Q&A and plan comparison",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting (slowapi)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _HAS_SLOWAPI = True
    _RATE_LIMIT_STR = f"{RATE_LIMIT_PER_MINUTE}/minute"
    def _rate_limit(f):
        return limiter.limit(_RATE_LIMIT_STR)(f)
except ImportError:
    limiter = None
    _HAS_SLOWAPI = False
    def _rate_limit(f):
        return f

# Middleware: API key (when API_KEY set), structured request logging
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(APIKeyMiddleware)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=QUERY_MAX_LENGTH, description="User question")
    plan_filter: Optional[str] = None


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=QUERY_MAX_LENGTH, description="User question")
    plans: Optional[List[str]] = None


@app.get("/")
def root():
    """Welcome and API info. Use /docs for interactive API documentation."""
    return {
        "app": "InsuranceCopilot AI",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "query": "POST /query",
        "compare": "POST /compare",
        "metrics": "GET /metrics",
    }


@app.get("/health")
def health():
    """Liveness: ok. Readiness: 503 if shutting down or Qdrant unreachable."""
    global _shutting_down
    if _shutting_down:
        raise HTTPException(status_code=503, detail="Server shutting down")
    if not _check_qdrant():
        raise HTTPException(status_code=503, detail="Qdrant unreachable or collection missing")
    return {"status": "ok"}


def _handle_query_error(e: Exception, endpoint: str) -> None:
    logger.exception("%s failed", endpoint)
    send_alert(
        "InsuranceCopilot API Error",
        f"{endpoint}: {type(e).__name__}: {str(e)[:200]}",
        payload={"endpoint": endpoint},
    )


@app.post("/query", response_model=dict)
@_rate_limit
def query(request: Request, req: QueryRequest) -> dict[str, Any]:
    """Run RAG query; returns answer, citations, latency, cost."""
    try:
        pipeline = get_pipeline()
        result = pipeline.query(question=req.question.strip(), plan_filter=req.plan_filter)
        return result
    except ValueError as e:
        if "OPENAI_API_KEY" in str(e) or "environment variable" in str(e).lower():
            raise HTTPException(status_code=503, detail="API key not configured. Add OPENAI_API_KEY to .env")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _handle_query_error(e, "POST /query")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=dict)
@_rate_limit
def compare_plans(request: Request, req: CompareRequest) -> dict[str, Any]:
    """Compare coverage across plans; returns answer and comparison data."""
    try:
        pipeline = get_pipeline()
        result = pipeline.compare_plans(question=req.question.strip(), plans=req.plans)
        return result
    except ValueError as e:
        if "OPENAI_API_KEY" in str(e):
            raise HTTPException(status_code=503, detail="API key not configured. Add OPENAI_API_KEY to .env")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _handle_query_error(e, "POST /compare")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Return latency stats (p50, p95, p99)."""
    return LatencyTracker.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
