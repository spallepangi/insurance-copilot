"""
Cost per query from token usage (OpenAI pricing).
"""

from src.utils.config import OPENAI_INPUT_COST_PER_1K, OPENAI_OUTPUT_COST_PER_1K


class CostTracker:
    """Compute cost from prompt and completion tokens."""

    @staticmethod
    def compute_cost(
        prompt_tokens: int,
        completion_tokens: int,
        input_cost_per_1k: float = OPENAI_INPUT_COST_PER_1K,
        output_cost_per_1k: float = OPENAI_OUTPUT_COST_PER_1K,
    ) -> float:
        """Cost in USD."""
        return (prompt_tokens / 1000 * input_cost_per_1k) + (
            completion_tokens / 1000 * output_cost_per_1k
        )
