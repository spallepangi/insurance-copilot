"""
Rerank retrieved chunks using BAAI/bge-reranker-large.
Processes documents in batches for GPU/CPU efficiency.
"""

from typing import Any, List

from src.utils.config import RERANK_BATCH_SIZE, RERANKER_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BGEReranker:
    """Rerank documents with bge-reranker-large. Uses batch inference (batch_size=16 by default)."""

    def __init__(self, model_name: str = RERANKER_MODEL, batch_size: int = RERANK_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(self.model_name, use_fp16=True)
            except Exception as e:
                logger.error("Failed to load reranker %s: %s", self.model_name, e)
                raise RuntimeError(
                    f"Could not load reranker {self.model_name}. "
                    "Ensure HUGGINGFACE_TOKEN is set in .env if the model is gated."
                ) from e
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        doc_metadata: List[dict] | None = None,
    ) -> List[dict[str, Any]]:
        """
        Rerank documents by relevance to query. Processes in batches of batch_size for efficiency.
        If doc_metadata is provided, each element is merged with the reranked result.
        """
        if not documents:
            return []
        model = self._get_model()
        meta_list = doc_metadata or [{}] * len(documents)
        all_scores: List[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch_docs = documents[start : start + self.batch_size]
            pairs = [[query, d] for d in batch_docs]
            scores = model.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(float(s) for s in scores)
        indexed = list(zip(range(len(documents)), all_scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        out = []
        for idx, score in indexed[:top_k]:
            item = dict(meta_list[idx]) if idx < len(meta_list) else {}
            item["score"] = float(score)
            item["text"] = documents[idx]
            out.append(item)
        return out
