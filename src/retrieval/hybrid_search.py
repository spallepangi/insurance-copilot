"""
Hybrid search: vector (top 20) + BM25 keyword (top 20) → merge & deduplicate → weighted scoring → top 40.
Optional metadata boost when query terms match section (e.g. emergency, deductible, copay).
"""

from typing import Any, List, Optional, Tuple

from src.embeddings.embedder import BGEEmbedder
from src.retrieval.bm25_index import load as bm25_load
from src.utils.config import (
    BM25_INDEX_PATH,
    KEYWORD_TOP_K,
    KEYWORD_WEIGHT,
    RERANK_POOL_SIZE,
    VECTOR_TOP_K,
    VECTOR_WEIGHT,
)
from src.utils.logger import get_logger
from src.vector_store.qdrant_client import QdrantStore

logger = get_logger(__name__)

# Query terms that trigger section-metadata boost when section title matches
SECTION_BOOST_TERMS = ("emergency", "deductible", "copay", "prescription", "coverage")
SECTION_BOOST_DELTA = 0.1


def _normalize_scores_min_max(items: List[dict], score_key: str = "score") -> None:
    """In-place: scale scores to [0, 1] by min-max over the list."""
    if not items:
        return
    scores = [x.get(score_key, 0.0) for x in items]
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        for x in items:
            x[score_key] = 1.0
        return
    for x in items:
        x[score_key] = (x.get(score_key, 0.0) - lo) / (hi - lo)


def _apply_metadata_boost(query: str, candidates: List[dict[str, Any]]) -> None:
    """If query contains SECTION_BOOST_TERMS, boost candidates whose section contains that term."""
    q_lower = query.lower()
    for term in SECTION_BOOST_TERMS:
        if term not in q_lower:
            continue
        for c in candidates:
            section = (c.get("section") or "").lower()
            if term in section:
                c["hybrid_score"] = c.get("hybrid_score", 0.0) + SECTION_BOOST_DELTA


class HybridSearch:
    """
    Vector search (top 20) + BM25 search (top 20) → merge by id → hybrid_score = 0.7*vec + 0.3*kw
    → optional section boost → sort → top RERANK_POOL_SIZE (40).
    """

    def __init__(
        self,
        embedder: BGEEmbedder | None = None,
        vector_store: QdrantStore | None = None,
    ):
        self.embedder = embedder or BGEEmbedder()
        self.vector_store = vector_store or QdrantStore()
        self._bm25: Any = None

    def _get_bm25(self):
        if self._bm25 is None:
            if not BM25_INDEX_PATH.exists():
                logger.warning("BM25 index not found at %s; keyword retrieval disabled", BM25_INDEX_PATH)
                return None
            self._bm25 = bm25_load(BM25_INDEX_PATH)
        return self._bm25

    def search(
        self,
        query: str,
        top_k: int = 10,
        plan_filter: Optional[str] = None,
    ) -> Tuple[List[dict[str, Any]], dict[str, int]]:
        """
        Run vector (top 20) + keyword (top 20), merge, hybrid scoring, optional metadata boost.
        Returns (candidates up to RERANK_POOL_SIZE, stats_dict with vector_candidates, keyword_candidates).
        """
        # Vector search
        query_vector = self.embedder.embed_query(query)
        raw_vector = self.vector_store.search(
            vector=query_vector,
            limit=VECTOR_TOP_K,
            plan_filter=plan_filter,
        )
        vector_results: List[dict[str, Any]] = []
        for r in raw_vector:
            pid = r.get("id")
            payload = r.get("payload") or {}
            vector_results.append({
                "id": pid,
                "text": payload.get("text", ""),
                "plan": payload.get("plan", ""),
                "section": payload.get("section", ""),
                "page": payload.get("page", 0),
                "score": r.get("score", 0.0),
            })
        n_vector = len(vector_results)

        # Keyword (BM25) search
        keyword_results: List[dict[str, Any]] = []
        bm25 = self._get_bm25()
        if bm25:
            keyword_results = bm25.search(query=query, top_k=KEYWORD_TOP_K, plan_filter=plan_filter)
        n_keyword = len(keyword_results)

        # Normalize vector scores to [0,1] (Qdrant cosine can be in [0,1] or [-1,1]; we clamp to 0-1)
        _normalize_scores_min_max(vector_results, "score")
        for r in vector_results:
            r["vector_score"] = r["score"]
        _normalize_scores_min_max(keyword_results, "score")
        for r in keyword_results:
            r["keyword_score"] = r["score"]

        # Merge by id; combine scores
        by_id: dict[str, dict[str, Any]] = {}
        for r in vector_results:
            by_id[r["id"]] = {
                "id": r["id"],
                "text": r["text"],
                "plan": r["plan"],
                "section": r["section"],
                "page": r["page"],
                "vector_score": r["vector_score"],
                "keyword_score": 0.0,
            }
        for r in keyword_results:
            uid = r["id"]
            if uid in by_id:
                by_id[uid]["keyword_score"] = r["keyword_score"]
            else:
                by_id[uid] = {
                    "id": uid,
                    "text": r["text"],
                    "plan": r["plan"],
                    "section": r["section"],
                    "page": r["page"],
                    "vector_score": 0.0,
                    "keyword_score": r["keyword_score"],
                }

        # Hybrid score
        for c in by_id.values():
            c["hybrid_score"] = (
                VECTOR_WEIGHT * c["vector_score"] + KEYWORD_WEIGHT * c["keyword_score"]
            )
            c["score"] = c["hybrid_score"]

        candidates = list(by_id.values())
        _apply_metadata_boost(query, candidates)
        candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        pool = candidates[:RERANK_POOL_SIZE]

        stats = {
            "vector_candidates": n_vector,
            "keyword_candidates": n_keyword,
            "reranked_candidates": len(pool),
        }
        return pool, stats