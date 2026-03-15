"""
Configuration management for InsuranceCopilot AI.
Loads from environment variables; never hardcodes API keys.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root (insurance-copilot/)
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path)


def _get_required(key: str, env_name: str) -> str:
    value = os.getenv(key)
    if not value or not value.strip():
        raise ValueError(
            f"Missing required environment variable: {env_name}. "
            f"Add it to your .env file. See .env.example for reference."
        )
    return value.strip()


def _get_optional(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return value.strip()


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
EVAL_DATASET_PATH = PROJECT_ROOT / "src" / "evaluation" / "evaluation_dataset.json"

# --- API Keys (required for full functionality) ---
OPENAI_API_KEY: Optional[str] = _get_optional("OPENAI_API_KEY")
COHERE_API_KEY: Optional[str] = _get_optional("COHERE_API_KEY")
QDRANT_URL: Optional[str] = _get_optional("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: Optional[str] = _get_optional("QDRANT_API_KEY")
HUGGINGFACE_TOKEN: Optional[str] = _get_optional("HUGGINGFACE_TOKEN")
LANGCHAIN_API_KEY: Optional[str] = _get_optional("LANGCHAIN_API_KEY")

# --- Embedding & Reranker ---
EMBEDDING_MODEL = "BAAI/bge-large-en"
RERANKER_MODEL = "BAAI/bge-reranker-large"

# --- Chunking ---
CHUNK_SIZE_TOKENS = 450
CHUNK_OVERLAP_TOKENS = 120

# --- Retrieval ---
TOP_K_RETRIEVAL = 10  # legacy; pool size driven by vector + keyword
TOP_K_AFTER_RERANK = 5
RERANK_BATCH_SIZE = 16
# Hybrid retrieval pool: vector top_k + keyword top_k → merge → rerank pool
VECTOR_TOP_K = 20
KEYWORD_TOP_K = 20
RERANK_POOL_SIZE = 40
VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
# BM25 index built at ingestion; path under data/
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"

# --- Context compression (before LLM) ---
CONTEXT_COMPRESSION_SENTENCES = 2

# --- Qdrant ---
QDRANT_COLLECTION_NAME = "insurance_policy_chunks"
VECTOR_SIZE = 1024  # bge-large-en dimension

# --- API (production) ---
API_KEY: Optional[str] = _get_optional("API_KEY")  # If set, requests must send X-API-Key or Authorization: Bearer <key>
_rate_limit = _get_optional("RATE_LIMIT_PER_MINUTE")
RATE_LIMIT_PER_MINUTE: int = int(_rate_limit) if _rate_limit and _rate_limit.isdigit() else 100
HTTP_TIMEOUT_SECONDS: float = 60.0  # Timeout for Qdrant and outbound HTTP
# Optional alerting (if set, 5xx and high-latency events can be sent)
SLACK_WEBHOOK_URL: Optional[str] = _get_optional("SLACK_WEBHOOK_URL")
DISCORD_WEBHOOK_URL: Optional[str] = _get_optional("DISCORD_WEBHOOK_URL")
# Query validation
QUERY_MAX_LENGTH: int = 2000  # Max character length for question field

# --- LLM (OpenAI by default) ---
LLM_MODEL = _get_optional("LLM_MODEL", "gpt-4o-mini")

# --- Cost (OpenAI pricing per 1K tokens, USD) ---
# gpt-4o-mini: $0.15/1M input, $0.60/1M output -> per 1K
OPENAI_INPUT_COST_PER_1K = 0.00015
OPENAI_OUTPUT_COST_PER_1K = 0.0006


def require_openai_key() -> str:
    """Return OpenAI API key or raise with clear message."""
    return _get_required("OPENAI_API_KEY", "OPENAI_API_KEY")


def require_qdrant_url() -> str:
    """Return Qdrant URL (defaults to local)."""
    return _get_optional("QDRANT_URL", "http://localhost:6333") or "http://localhost:6333"


def require_hf_token() -> str:
    """Return HuggingFace token for gated models or raise."""
    return _get_required("HUGGINGFACE_TOKEN", "HUGGINGFACE_TOKEN")
