"""
BAAI/bge-large-en embeddings for insurance policy chunks.
"""

from typing import List

from src.utils.config import EMBEDDING_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BGEEmbedder:
    """Generate embeddings using BAAI/bge-large-en."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error("Failed to load embedding model %s: %s", self.model_name, e)
                raise RuntimeError(
                    f"Could not load embedding model {self.model_name}. "
                    "Ensure HUGGINGFACE_TOKEN is set in .env if the model is gated."
                ) from e
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts; returns list of vectors."""
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query (e.g. with instruction prefix for retrieval)."""
        model = self._get_model()
        # bge recommends prefix "Represent this sentence for searching: " for retrieval
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        vec = model.encode([prefixed], normalize_embeddings=True)
        return vec[0].tolist()

    @property
    def dimension(self) -> int:
        """Embedding dimension (1024 for bge-large-en)."""
        model = self._get_model()
        return model.get_sentence_embedding_dimension()
