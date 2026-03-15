# Multi-stage: build not required for Python; single stage for API.
# Usage: docker build -t insurance-copilot-api . && docker run -p 8000:8000 --env-file .env insurance-copilot-api

FROM python:3.10-slim

WORKDIR /app

# System deps if needed (e.g. for sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
# Optional: copy data/ if you embed PDFs at build time; usually mount at run
# COPY data/ ./data/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
