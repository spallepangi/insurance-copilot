"""
Smart chunking by section boundaries.
Chunk size 450 tokens, overlap 120 tokens; preserves metadata (plan, section, page, text).
Section boundaries are respected (one section at a time in chunk_document).
"""

import re
from typing import Any

from src.utils.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Approximate tokens per character for English (e.g. tiktoken-like)
CHARS_PER_TOKEN_APPROX = 4


def _estimate_tokens(text: str) -> int:
    """Rough token count (words + punctuation)."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN_APPROX)


def _split_into_sentences(text: str) -> list[str]:
    """Simple sentence split for overlap boundary."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


class SectionChunker:
    """
    Chunk documents by section boundaries.
    Target size 450 tokens, overlap 120 tokens; section boundaries respected per chunk_document.
    """

    def __init__(
        self,
        chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
        overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ):
        self.chunk_size = chunk_size_tokens
        self.overlap = overlap_tokens

    def chunk_section(
        self,
        content: str,
        table_content: str,
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Chunk a single section's content (and optional table_content)
        into chunks of ~chunk_size tokens with overlap, preserving metadata.
        """
        combined = content
        if table_content:
            combined = combined + "\n\n" + table_content if combined else table_content

        if not combined.strip():
            return []

        chunks: list[dict[str, Any]] = []
        sentences = _split_into_sentences(combined)
        if not sentences:
            # Fallback: split by fixed character length
            chunk_chars = self.chunk_size * CHARS_PER_TOKEN_APPROX
            overlap_chars = self.overlap * CHARS_PER_TOKEN_APPROX
            start = 0
            while start < len(combined):
                end = min(start + chunk_chars, len(combined))
                chunk_text = combined[start:end]
                chunks.append({
                    **metadata,
                    "text": chunk_text,
                })
                start = end - overlap_chars
                if start >= len(combined) - overlap_chars:
                    break
            return chunks

        current: list[str] = []
        current_tokens = 0
        overlap_buffer: list[str] = []

        for sent in sentences:
            sent_tokens = _estimate_tokens(sent)
            if current_tokens + sent_tokens > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append({
                    **metadata,
                    "text": chunk_text,
                })
                # Keep overlap
                overlap_tokens_so_far = 0
                overlap_buffer = []
                for s in reversed(current):
                    t = _estimate_tokens(s)
                    if overlap_tokens_so_far + t <= self.overlap:
                        overlap_buffer.append(s)
                        overlap_tokens_so_far += t
                    else:
                        break
                overlap_buffer.reverse()
                current = list(overlap_buffer)
                current_tokens = sum(_estimate_tokens(s) for s in current)
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunk_text = " ".join(current)
            chunks.append({
                **metadata,
                "text": chunk_text,
            })

        return chunks

    def chunk_document(
        self,
        sections: list[dict[str, Any]],
        plan_name: str,
        source_file: str,
    ) -> list[dict[str, Any]]:
        """
        Chunk a full document (list of sections from pdf_parser) with plan metadata.
        """
        all_chunks: list[dict[str, Any]] = []
        for sec in sections:
            meta = {
                "plan": plan_name,
                "section": sec.get("section_title", "Unknown"),
                "page": sec.get("page_number", 0),
                "plan_name": plan_name,
                "section_title": sec.get("section_title", "Unknown"),
                "page_number": sec.get("page_number", 0),
                "source_file": source_file,
            }
            sub = self.chunk_section(
                sec.get("content", "") or "",
                sec.get("table_content", "") or "",
                meta,
            )
            all_chunks.extend(sub)
        logger.info("Chunked into %d chunks (plan=%s)", len(all_chunks), plan_name)
        return all_chunks
