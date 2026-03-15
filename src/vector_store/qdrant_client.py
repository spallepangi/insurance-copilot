"""
Qdrant vector store client for insurance policy chunks.
Supports filtering by plan.
"""

from typing import Any, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.utils.config import (
    HTTP_TIMEOUT_SECONDS,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    VECTOR_SIZE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _retry_search(func):
    """Retry Qdrant search up to 3 times with exponential backoff."""
    from tenacity import retry, stop_after_attempt, wait_exponential
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )(func)


class QdrantStore:
    """Qdrant client wrapper for storing and querying policy chunks."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = QDRANT_COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE,
    ):
        self.url = url or QDRANT_URL or "http://localhost:6333"
        self.api_key = api_key or QDRANT_API_KEY
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            kwargs: dict = {"url": self.url, "timeout": int(HTTP_TIMEOUT_SECONDS)}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = QdrantClient(**kwargs)
            logger.info(
                "Qdrant connected to %s (API key %s)",
                self.url,
                "set" if self.api_key else "not set",
            )
        return self._client

    def ensure_collection(self, recreate: bool = False):
        """Create collection if not exists; optionally recreate. Creates payload index for 'plan' for filtering."""
        exists = self.collection_exists()
        if recreate and exists:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            exists = False
        if not exists or recreate:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=10000,
                ),
            )
            logger.info("Collection %s ready", self.collection_name)
        self._ensure_plan_index()

    def _ensure_plan_index(self):
        """Create payload index on 'plan' so filter by plan works. Idempotent (ignores if exists)."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="plan",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            logger.info("Payload index for 'plan' ready")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                pass
            else:
                logger.warning("Could not create payload index for 'plan': %s", e)

    def collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ):
        """Insert or update points."""
        points = [
            qmodels.PointStruct(
                id=uid,
                vector=vec,
                payload=payload,
            )
            for uid, vec, payload in zip(ids, vectors, payloads)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info("Upserted %d points", len(points))

    def _search_impl(
        self,
        vector: list[float],
        limit: int = 10,
        plan_filter: Optional[str] = None,
    ) -> list[dict]:
        """Internal: one attempt at vector search."""
        if plan_filter:
            self._ensure_plan_index()
        query_filter = None
        if plan_filter:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="plan",
                        match=qmodels.MatchValue(value=plan_filter),
                    )
                ]
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            query_filter=query_filter,
        )
        points = getattr(response, "points", []) or []
        return [
            {
                "id": str(getattr(r, "id", r)),
                "score": getattr(r, "score", 0.0),
                "payload": getattr(r, "payload", None) or {},
            }
            for r in points
        ]

    def search(
        self,
        vector: list[float],
        limit: int = 10,
        plan_filter: Optional[str] = None,
    ) -> list[dict]:
        """Vector similarity search with optional plan filter. Retries up to 3 times on failure."""
        return _retry_search(lambda: self._search_impl(vector, limit, plan_filter))()

    def search_with_filter(
        self,
        vector: list[float],
        limit: int,
        filter_conditions: Optional[qmodels.Filter] = None,
    ) -> list[dict]:
        """Search with custom Qdrant filter."""
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            query_filter=filter_conditions,
        )
        points = getattr(response, "points", []) or []
        return [
            {
                "id": str(getattr(r, "id", r)),
                "score": getattr(r, "score", 0.0),
                "payload": getattr(r, "payload", None) or {},
            }
            for r in points
        ]
