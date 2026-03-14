"""
Build and populate the Qdrant index from chunked documents.
Also builds and saves the BM25 index for hybrid retrieval.
"""

from typing import Any, List
from uuid import uuid4

from src.embeddings.embedder import BGEEmbedder
from src.retrieval.bm25_index import build_and_save as bm25_build_and_save
from src.utils.config import BM25_INDEX_PATH
from src.utils.logger import get_logger
from src.vector_store.qdrant_client import QdrantStore

logger = get_logger(__name__)


class IndexBuilder:
    """Build vector index from chunks using BGE embeddings and Qdrant."""

    def __init__(
        self,
        embedder: BGEEmbedder | None = None,
        vector_store: QdrantStore | None = None,
    ):
        self.embedder = embedder or BGEEmbedder()
        self.vector_store = vector_store or QdrantStore()

    def index_chunks(self, chunks: List[dict[str, Any]], batch_size: int = 32):
        """
        Embed all chunks and upsert into Qdrant.
        Each chunk must have 'text' and metadata: plan, section, page, etc.
        """
        if not chunks:
            logger.warning("No chunks to index")
            return

        texts = [c.get("text", "") or "" for c in chunks]
        # Align payloads for Qdrant: plan, section, page, text
        payloads = []
        for c in chunks:
            payloads.append({
                "plan": c.get("plan") or c.get("plan_name", ""),
                "section": c.get("section") or c.get("section_title", ""),
                "page": c.get("page") if c.get("page") is not None else c.get("page_number", 0),
                "text": c.get("text", ""),
            })

        self.vector_store.ensure_collection(recreate=False)
        if not self.vector_store.collection_exists():
            self.vector_store.ensure_collection(recreate=True)

        # Ensure collection has correct vector size
        try:
            coll = self.vector_store.client.get_collection(self.vector_store.collection_name)
            if coll.config.params.vectors.size != self.embedder.dimension:
                self.vector_store.ensure_collection(recreate=True)
        except Exception:
            self.vector_store.ensure_collection(recreate=True)

        ids = [str(uuid4()) for _ in chunks]
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_vectors = self.embedder.embed_documents(batch_texts)
            batch_ids = ids[i : i + batch_size]
            batch_payloads = payloads[i : i + batch_size]
            self.vector_store.upsert(batch_ids, batch_vectors, batch_payloads)

        logger.info("Indexed %d chunks into Qdrant", len(chunks))
        try:
            bm25_build_and_save(ids=ids, payloads=payloads, path=BM25_INDEX_PATH)
        except Exception as e:
            logger.warning("BM25 index build/save failed (keyword retrieval will be disabled): %s", e)
