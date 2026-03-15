#!/usr/bin/env bash
# Set up environment and run all required commands from the beginning.
# Usage: from project root, run:  bash scripts/setup_and_run.sh
# Ensure .env exists with OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, HUGGINGFACE_TOKEN.

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

echo "=== 1. Virtual environment ==="
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "Created .venv"
fi
# Activate for this script (optional; we call .venv/bin/python explicitly)
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

echo "=== 2. Install dependencies ==="
.venv/bin/pip install -q -r requirements.txt
echo "Dependencies installed."

echo "=== 3. Check .env ==="
if [ ! -f ".env" ]; then
  echo "ERROR: .env not found. Copy .env.example to .env and add API keys."
  exit 1
fi
echo ".env present."

echo "=== 4. Check data/ PDFs ==="
PDF_COUNT=$(find data -maxdepth 1 -name "*.pdf" 2>/dev/null | wc -l)
if [ "$PDF_COUNT" -eq 0 ]; then
  echo "ERROR: No PDFs in data/. Add bronze.pdf, silver.pdf, gold.pdf, platinum.pdf (or bbbronze.pdf, etc.)."
  exit 1
fi
echo "Found $PDF_COUNT PDF(s) in data/."

echo "=== 5. Ingest documents (parse → chunk → embed → Qdrant + BM25) ==="
.venv/bin/python -m scripts.ingest_documents
echo "Ingestion complete."

echo "=== 6. Quick evaluation (first 5 queries) ==="
.venv/bin/python -m src.evaluation.evaluation_runner --limit=5
echo ""
echo "=== Done. To run full evaluation (100 queries): .venv/bin/python -m src.evaluation.evaluation_runner ==="
echo "=== To start API: .venv/bin/python -m scripts.run_pipeline api ==="
echo "=== To start UI:  .venv/bin/python -m scripts.run_pipeline ui ==="
