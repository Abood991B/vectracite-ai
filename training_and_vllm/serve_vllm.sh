#!/usr/bin/env bash
# Launches high-throughput vLLM OpenAI-compatible inference server with PagedAttention,
# 4-bit AWQ quantization, Continuous Batching, and dynamic LoRA adapter serving.

set -euo pipefail

MODEL_ID="${MODEL_ID:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
LORA_PATH="${LORA_PATH:-./qlora_checkpoints/vectracite-llama3.1-8b-lora}"
PORT="${PORT:-8000}"

echo "Starting vLLM Inference Engine for ${MODEL_ID} on port ${PORT}..."

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --quantization awq \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-lora \
  --lora-modules vectracite-lora="${LORA_PATH}" \
  --port "${PORT}"
