"""
Generate answers from retrieved chunks using an LLM, with citations.
"""

from typing import Any, List

from src.utils.logger import get_logger

logger = get_logger(__name__)

CITATION_TEMPLATE = (
    "According to the {plan} plan policy ({section} section, page {page}), {excerpt}"
)


def _build_context(chunks: List[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        plan = c.get("plan", "Unknown")
        section = c.get("section", "Unknown")
        page = c.get("page", 0)
        text = (c.get("text") or "").strip()
        if text:
            parts.append(f"[{i}] Plan: {plan} | Section: {section} | Page: {page}\n{text}")
    return "\n\n---\n\n".join(parts)


def _format_citations(chunks: List[dict]) -> List[dict]:
    """Build citation dicts: plan, section, page, policy_excerpt."""
    citations = []
    for c in chunks:
        citations.append({
            "plan": c.get("plan", ""),
            "section": c.get("section", ""),
            "page": c.get("page", 0),
            "policy_excerpt": (c.get("text") or "")[:500],
        })
    return citations


class AnswerGenerator:
    """Generate answer using OpenAI (or other LLM) with source citations."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from src.utils.config import require_openai_key, LLM_MODEL
            import openai
            self._api_key = self._api_key or require_openai_key()
            self._client = openai.OpenAI(api_key=self._api_key)
            if not self.model:
                self.model = LLM_MODEL
        return self._client

    def generate(
        self,
        query: str,
        chunks: List[dict[str, Any]],
        compressed_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate answer and citations.
        If compressed_context is provided, use it as the policy context sent to the LLM; otherwise build from chunks.
        Returns: { "answer", "citations": [ { "plan", "section", "page", "policy_excerpt" } ], "usage" }
        """
        if not chunks:
            return {
                "answer": "I couldn't find relevant policy sections for your question. Please rephrase or specify a plan.",
                "citations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        context = compressed_context if compressed_context is not None else _build_context(chunks)
        system = (
            "You are an expert in healthcare insurance policy documents. "
            "Answer the user's question using ONLY the provided policy excerpts. "
            "Cite the plan name, section title, and page number when you reference information. "
            "If the information is not in the context, say so. Be precise and concise."
        )
        user = f"Policy context:\n\n{context}\n\nQuestion: {query}"

        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        choice = resp.choices[0] if resp.choices else None
        answer = choice.message.content if choice else ""
        usage = {
            "prompt_tokens": getattr(resp, "usage", None) and resp.usage.prompt_tokens or 0,
            "completion_tokens": getattr(resp, "usage", None) and resp.usage.completion_tokens or 0,
        }
        citations = _format_citations(chunks)
        return {
            "answer": answer,
            "citations": citations,
            "usage": usage,
        }
