#!/usr/bin/env bash
set -euo pipefail

exec vllm serve "${MODEL_PATH:-./artifacts/models/bge-reranker-v2-m3}" \
    --served-model-name "${SERVED_MODEL_NAME:-survey-cross-encoder}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --runner pooling \
    --pooler-config '{"use_activation": true}' \
    --max-model-len "${MAX_MODEL_LEN:-256}" \
    --host "${VLLM_HOST:-0.0.0.0}" \
    --port "${VLLM_PORT:-8000}"
