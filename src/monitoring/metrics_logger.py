"""
Log query metrics (latency, tokens, cost) to local JSON/SQLite for observability.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from src.utils.config import LOGS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

METRICS_FILE = LOGS_DIR / "query_metrics.jsonl"
SQLITE_DB = LOGS_DIR / "metrics.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    latency_ms REAL,
    retrieval_latency_ms REAL,
    generation_latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_INSERT_SQL = """
INSERT INTO query_metrics
(query, latency_ms, retrieval_latency_ms, generation_latency_ms, prompt_tokens, completion_tokens, cost)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


class MetricsLogger:
    """Append each query's metrics to a JSONL file and optionally SQLite.

    SQLite connection is kept alive for the lifetime of this instance to avoid
    per-query open/close overhead. The schema is created once on first write.
    """

    def __init__(self, jsonl_path: Path | None = None, use_sqlite: bool = True):
        self.jsonl_path = jsonl_path or METRICS_FILE
        self.use_sqlite = use_sqlite
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._table_ready: bool = False

    def _ensure_log_dir(self) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        """Return the persistent SQLite connection, creating it and the schema on first call."""
        if self._sqlite_conn is None:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            self._sqlite_conn = sqlite3.connect(str(SQLITE_DB), check_same_thread=False)
            # WAL mode for better concurrent write performance
            self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
        if not self._table_ready:
            self._sqlite_conn.execute(_CREATE_TABLE_SQL)
            self._sqlite_conn.commit()
            self._table_ready = True
        return self._sqlite_conn

    def log_query(
        self,
        query: str,
        latency_ms: float,
        retrieval_latency_ms: float = 0,
        generation_latency_ms: float = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0,
        original_token_estimate: int | None = None,
        compressed_token_estimate: int | None = None,
        rerank_ms: float | None = None,
        compression_ms: float | None = None,
        retrieval_stats: dict[str, int] | None = None,
    ) -> None:
        record = {
            "query": query[:500],
            "latency_ms": latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        }
        if original_token_estimate is not None:
            record["original_token_estimate"] = original_token_estimate
        if compressed_token_estimate is not None:
            record["compressed_token_estimate"] = compressed_token_estimate
        if rerank_ms is not None:
            record["rerank_ms"] = round(rerank_ms, 2)
        if compression_ms is not None:
            record["compression_ms"] = round(compression_ms, 2)
        if retrieval_stats:
            record["vector_candidates"] = retrieval_stats.get("vector_candidates", 0)
            record["keyword_candidates"] = retrieval_stats.get("keyword_candidates", 0)
            record["reranked_candidates"] = retrieval_stats.get("reranked_candidates", 0)
            record["final_context_chunks"] = retrieval_stats.get("final_context_chunks", 0)
        self._ensure_log_dir()
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.use_sqlite:
            self._write_sqlite(record)

    def _write_sqlite(self, record: dict[str, Any]) -> None:
        try:
            conn = self._get_conn()
            conn.execute(
                _INSERT_SQL,
                (
                    record.get("query", ""),
                    record.get("latency_ms", 0),
                    record.get("retrieval_latency_ms", 0),
                    record.get("generation_latency_ms", 0),
                    record.get("prompt_tokens", 0),
                    record.get("completion_tokens", 0),
                    record.get("cost", 0),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.debug("SQLite metrics write failed: %s", e)
