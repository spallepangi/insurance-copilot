"""
Context compression: reduce tokens sent to the LLM by keeping only the first N sentences per chunk.
"""

import re
from typing import List

from src.utils.config import CONTEXT_COMPRESSION_SENTENCES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sentences_from_text(text: str) -> List[str]:
    """Split text into sentences (period, question mark, exclamation). Trim whitespace."""
    if not (text or "").strip():
        return []
    # Split on sentence boundaries, keep delimiters for reconstruction
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def compress_context(chunks: List[str], max_sentences: int | None = None) -> str:
    """
    For each chunk, keep only the first max_sentences sentences, then join.
    Reduces prompt size (e.g. from ~4000 to ~600-900 tokens).
    """
    max_sentences = max_sentences if max_sentences is not None else CONTEXT_COMPRESSION_SENTENCES
    compressed_parts: List[str] = []
    for chunk in chunks:
        if not (chunk or "").strip():
            continue
        sentences = _sentences_from_text(chunk)
        selected = sentences[:max_sentences]
        if selected:
            compressed_parts.append(" ".join(selected))
    return "\n\n".join(compressed_parts)


def estimate_tokens(text: str) -> int:
    """Simple token estimate: len(words) * 1.3."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)
