"""
BM25 keyword index for hybrid retrieval.
Built at ingestion; loaded at query time. Returns top-k by BM25 score.
"""

import re
from pathlib import Path
from typing import Any, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, keep words 2+ chars."""
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
    return tokens


def build_and_save(
    ids: List[str],
    payloads: List[dict],
    path: Path,
) -> None:
    """
    Build BM25 index from chunk ids and payloads (each must have 'text').
    Saves to path as pickle: { "ids", "payloads", "tokenized_corpus", "bm25" }.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("rank_bm25 is required for BM25 retrieval. Install with: pip install rank_bm25")

    corpus_texts = [p.get("text") or "" for p in payloads]
    tokenized_corpus = [_tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(path, "wb") as f:
        pickle.dump({
            "ids": ids,
            "payloads": payloads,
            "tokenized_corpus": tokenized_corpus,
            "bm25": bm25,
        }, f)
    logger.info("BM25 index saved to %s (%d documents)", path, len(ids))


def load(path: Path) -> "BM25Search":
    """Load BM25 index from path. Returns BM25Search instance."""
    import pickle
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found: {path}. Run ingestion to build it.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return BM25Search(
        ids=data["ids"],
        payloads=data["payloads"],
        bm25=data["bm25"],
    )


class BM25Search:
    """Query a pre-built BM25 index. Returns hits with id, text, plan, section, page, score."""

    def __init__(self, ids: List[str], payloads: List[dict], bm25: Any):
        self.ids = ids
        self.payloads = payloads
        self.bm25 = bm25

    def search(
        self,
        query: str,
        top_k: int = 20,
        plan_filter: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        """
        Return top_k hits as list of { id, text, plan, section, page, score }.
        If plan_filter is set, fetch more then filter and take top_k.
        """
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        # Index -> (id, payload, score)
        indexed = [(i, self.ids[i], self.payloads[i], float(scores[i])) for i in range(len(self.ids))]
        indexed.sort(key=lambda x: x[3], reverse=True)
        if plan_filter:
            filtered = [
                (idx, uid, pl, sc)
                for idx, uid, pl, sc in indexed
                if (pl.get("plan") or pl.get("plan_name") or "") == plan_filter
            ]
            indexed = filtered
        out = []
        for _, uid, pl, sc in indexed[:top_k]:
            out.append({
                "id": uid,
                "text": pl.get("text", ""),
                "plan": pl.get("plan") or pl.get("plan_name", ""),
                "section": pl.get("section") or pl.get("section_title", ""),
                "page": pl.get("page") if pl.get("page") is not None else pl.get("page_number", 0),
                "score": sc,
            })
        return out
