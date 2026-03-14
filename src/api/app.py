"""
FastAPI backend for InsuranceCopilot AI.
Endpoints: query, compare plans, health, metrics.
"""

from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.monitoring.latency_tracker import LatencyTracker
from src.rag.rag_pipeline import RAGPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global pipeline (lazy init to avoid loading keys at import)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: optional LangSmith etc.
    yield
    # Shutdown
    pass


app = FastAPI(
    title="InsuranceCopilot AI",
    description="Production RAG API for healthcare insurance policy Q&A and plan comparison",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str
    plan_filter: Optional[str] = None


class CompareRequest(BaseModel):
    question: str
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
    return {"status": "ok"}


@app.post("/query", response_model=dict)
def query(req: QueryRequest) -> dict[str, Any]:
    """Run RAG query; returns answer, citations, latency, cost."""
    try:
        pipeline = get_pipeline()
        result = pipeline.query(question=req.question, plan_filter=req.plan_filter)
        return result
    except ValueError as e:
        if "OPENAI_API_KEY" in str(e) or "environment variable" in str(e).lower():
            raise HTTPException(status_code=503, detail="API key not configured. Add OPENAI_API_KEY to .env")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=dict)
def compare_plans(req: CompareRequest) -> dict[str, Any]:
    """Compare coverage across plans; returns answer and comparison data."""
    try:
        pipeline = get_pipeline()
        result = pipeline.compare_plans(question=req.question, plans=req.plans)
        return result
    except ValueError as e:
        if "OPENAI_API_KEY" in str(e):
            raise HTTPException(status_code=503, detail="API key not configured. Add OPENAI_API_KEY to .env")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Compare failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Return latency stats (p50, p95, p99)."""
    return LatencyTracker.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
