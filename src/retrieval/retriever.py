"""
Unified retriever: hybrid search (vector 20 + BM25 20 → merge → top 40) then rerank → top 5.
Supports retrieve_candidates (search only) and rerank_candidates (batch rerank) for pipeline timing.
"""

from typing import Any, List, Optional

from src.embeddings.embedder import BGEEmbedder
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import BGEReranker
from src.utils.config import RERANK_POOL_SIZE, TOP_K_AFTER_RERANK, TOP_K_RETRIEVAL
from src.utils.logger import get_logger
from src.vector_store.qdrant_client import QdrantStore

logger = get_logger(__name__)


class Retriever:
    """
    Retrieve pool via hybrid (vector top 20 + keyword top 20 → merge → top 40),
    then rerank (batched) and return top 5.
    """

    def __init__(
        self,
        embedder: BGEEmbedder | None = None,
        vector_store: QdrantStore | None = None,
        hybrid_search: HybridSearch | None = None,
        reranker: BGEReranker | None = None,
        top_k: int = TOP_K_RETRIEVAL,
        top_k_after_rerank: int = TOP_K_AFTER_RERANK,
    ):
        self.embedder = embedder or BGEEmbedder()
        self.vector_store = vector_store or QdrantStore()
        self.hybrid_search = hybrid_search or HybridSearch(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        self.reranker = reranker or BGEReranker()
        self.top_k = top_k
        self.top_k_after_rerank = top_k_after_rerank
        self._last_retrieval_stats: dict[str, int] = {}

    def retrieve_candidates(
        self,
        query: str,
        plan_filter: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        """Hybrid search only (vector 20 + keyword 20 → merge → top pool). No reranking."""
        candidates, stats = self.hybrid_search.search(
            query=query,
            top_k=RERANK_POOL_SIZE,
            plan_filter=plan_filter,
        )
        self._last_retrieval_stats = stats
        return candidates

    def rerank_candidates(
        self,
        query: str,
        candidates: List[dict[str, Any]],
        top_k: int | None = None,
    ) -> List[dict[str, Any]]:
        """Rerank candidates in batch. Returns top_k items."""
        if not candidates:
            return []
        k = top_k if top_k is not None else self.top_k_after_rerank
        return self.reranker.rerank(
            query=query,
            documents=[c.get("text", "") or "" for c in candidates],
            top_k=k,
            doc_metadata=candidates,
        )

    def retrieve(
        self,
        query: str,
        plan_filter: Optional[str] = None,
        use_rerank: bool = True,
    ) -> List[dict[str, Any]]:
        """
        Run hybrid retrieval, optionally rerank, return top_k_after_rerank chunks.
        Each item: { "text", "plan", "section", "page", "score", ... }
        """
        candidates = self.retrieve_candidates(query=query, plan_filter=plan_filter)
        if not candidates:
            return []
        if use_rerank and self.reranker:
            return self.rerank_candidates(query=query, candidates=candidates)
        return candidates[: self.top_k_after_rerank]
