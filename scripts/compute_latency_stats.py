"""
Compute latency statistics from stored metrics (JSONL or SQLite).
Run after queries have been logged.
  python -m scripts.compute_latency_stats
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import LOGS_DIR

METRICS_JSONL = LOGS_DIR / "query_metrics.jsonl"
METRICS_DB = LOGS_DIR / "metrics.db"


def from_jsonl():
    if not METRICS_JSONL.exists():
        return []
    latencies = []
    with open(METRICS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                latencies.append(rec.get("latency_ms", 0))
            except json.JSONDecodeError:
                continue
    return latencies


def from_sqlite():
    if not METRICS_DB.exists():
        return []
    import sqlite3
    conn = sqlite3.connect(str(METRICS_DB))
    cur = conn.execute("SELECT latency_ms FROM query_metrics")
    latencies = [row[0] for row in cur.fetchall()]
    conn.close()
    return latencies


def percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def main():
    latencies = from_sqlite() or from_jsonl()
    if not latencies:
        print("No metrics found. Run some queries first (API or UI).")
        return
    latencies.sort()
    n = len(latencies)
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)
    mean = sum(latencies) / n
    print("Latency statistics (ms)")
    print("  count:", n)
    print("  mean: ", round(mean, 2))
    print("  p50:  ", round(p50, 2))
    print("  p95:  ", round(p95, 2))
    print("  p99:  ", round(p99, 2))


if __name__ == "__main__":
    main()
