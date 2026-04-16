"""
Query latency tracking: p50, p95, p99 and optional per-stage timings.
"""

import statistics
from collections import deque
from typing import Deque, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# In-memory store; last N latencies (ms)
MAX_SAMPLES = 10_000
_latencies: Deque[float] = deque(maxlen=MAX_SAMPLES)
# Last stage timings for inspection (retrieval_ms, rerank_ms, compression_ms, generation_ms, total_ms)
_last_stage_timings: Optional[dict[str, float]] = None
# Cached sorted snapshot — invalidated whenever a new sample is recorded
_stats_cache: Optional[dict] = None


class LatencyTracker:
    """Track query latencies and compute percentiles. Optionally store stage timings."""

    @classmethod
    def record(cls, latency_ms: float, stage_timings: Optional[dict[str, float]] = None) -> None:
        global _last_stage_timings, _stats_cache
        _latencies.append(latency_ms)
        _stats_cache = None  # invalidate cache on new sample
        if stage_timings is not None:
            _last_stage_timings = stage_timings

    @classmethod
    def get_stats(cls) -> dict:
        """Return p50, p95, p99, count, mean; and last stage timings if available.
        Result is cached between calls and invalidated whenever a new sample is recorded."""
        global _stats_cache
        if _stats_cache is not None:
            return _stats_cache
        out: dict = {}
        if not _latencies:
            out = {
                "count": 0,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "mean_ms": None,
            }
        else:
            data = sorted(_latencies)
            n = len(data)
            out = {
                "count": n,
                "p50_ms": round(_percentile(data, 50), 2),
                "p95_ms": round(_percentile(data, 95), 2),
                "p99_ms": round(_percentile(data, 99), 2),
                "mean_ms": round(statistics.mean(data), 2),
            }
        if _last_stage_timings:
            out["last_stage_timings_ms"] = {k: round(v, 2) for k, v in _last_stage_timings.items()}
        _stats_cache = out
        return out

    @classmethod
    def reset(cls) -> None:
        global _last_stage_timings, _stats_cache
        _latencies.clear()
        _last_stage_timings = None
        _stats_cache = None


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
