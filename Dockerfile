FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

ARG PRELOAD_MODELS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/
COPY assets/ ./assets/
COPY data/benchmark_queries.json data/financebench_ground_truth.json ./data/
COPY data/corpus_index.json data/corpus_embeddings.npz ./data/
COPY data/corpus_pdfs/ ./data/corpus_pdfs/

RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"; \
    fi \
    && mkdir -p /app/data/uploads /app/data/pdfs /app/.cache/huggingface \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
