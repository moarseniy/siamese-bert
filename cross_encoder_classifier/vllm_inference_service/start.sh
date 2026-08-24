#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ! -f venv/bin/activate ]]; then
    echo "Virtual environment not found: ./venv" >&2
    exit 1
fi

. venv/bin/activate

MODEL_NAME="${MODEL_NAME:-bge-reranker-v2-m3}"
MODEL_PATH="./artifacts/models/${MODEL_NAME}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-survey-cross-encoder}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-256}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVICE_HOST="${SERVICE_HOST:-0.0.0.0}"
SERVICE_PORT="${SERVICE_PORT:-8080}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Trained model not found: ${MODEL_PATH}" >&2
    exit 1
fi

VLLM_API_KEY="${VLLM_API_KEY:-$(python -c 'import secrets; print(secrets.token_urlsafe(32))')}"
export VLLM_API_KEY
export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}"
export VLLM_MODEL="${SERVED_MODEL_NAME}"
export CLASSIFIER_CONFIG_PATH="${CLASSIFIER_CONFIG_PATH:-./artifacts/classifier_config.json}"
export MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${MODEL_PATH}/config.json}"
export CODEBOOK_PATH="${CODEBOOK_PATH:-./artifacts/codebook.xlsx}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer.json}"

VLLM_PID=""
GATEWAY_PID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    [[ -z "${GATEWAY_PID}" ]] || kill "${GATEWAY_PID}" 2>/dev/null || true
    [[ -z "${VLLM_PID}" ]] || kill "${VLLM_PID}" 2>/dev/null || true
    [[ -z "${GATEWAY_PID}" ]] || wait "${GATEWAY_PID}" 2>/dev/null || true
    [[ -z "${VLLM_PID}" ]] || wait "${VLLM_PID}" 2>/dev/null || true
    exit "${status}"
}
trap cleanup EXIT INT TERM

vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --runner pooling \
    --pooler-config '{"use_activation": true}' \
    --max-model-len "${MAX_MODEL_LEN}" \
    --api-key "${VLLM_API_KEY}" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" &
VLLM_PID=$!

uvicorn app.main:app \
    --host "${SERVICE_HOST}" \
    --port "${SERVICE_PORT}" \
    --workers 1 &
GATEWAY_PID=$!

set +e
wait -n "${VLLM_PID}" "${GATEWAY_PID}"
status=$?
set -e
exit "${status}"
