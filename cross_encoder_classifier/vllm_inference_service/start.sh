#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ! -f venv/bin/activate ]]; then
    echo "Virtual environment not found: ./venv" >&2
    exit 1
fi

. venv/bin/activate

MODEL_NAME="${MODEL_NAME:-bge-reranker-v2-m3}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-./artifacts}"
MODEL_PATH="${MODEL_PATH:-${ARTIFACTS_DIR}/models/${MODEL_NAME}}"
DVC_FILE="${DVC_FILE:-${MODEL_PATH}.dvc}"
DVC_REMOTE="${DVC_REMOTE:-}"
DVC_REMOTE_URL="${DVC_REMOTE_URL:-}"
DVC_S3_ENDPOINT_URL="${DVC_S3_ENDPOINT_URL:-}"
DVC_JOBS="${DVC_JOBS:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-survey-cross-encoder}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-256}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVICE_HOST="${SERVICE_HOST:-0.0.0.0}"
SERVICE_PORT="${SERVICE_PORT:-8080}"

pull_dvc_model() {
    if [[ ! -f "${DVC_FILE}" ]]; then
        echo "Neither trained model nor DVC file was found: ${DVC_FILE}" >&2
        exit 1
    fi
    if ! command -v dvc >/dev/null 2>&1; then
        echo "DVC file found, but dvc is not installed." >&2
        exit 1
    fi

    local repo_dir target remote_name
    if [[ -d "${ARTIFACTS_DIR}/.dvc" ]]; then
        repo_dir="${ARTIFACTS_DIR}"
        target="${DVC_FILE#${ARTIFACTS_DIR}/}"
    elif [[ -d .dvc ]]; then
        repo_dir="."
        target="${DVC_FILE}"
    elif [[ -n "${DVC_REMOTE_URL}" ]]; then
        repo_dir="${ARTIFACTS_DIR}"
        target="${DVC_FILE#${ARTIFACTS_DIR}/}"
        mkdir -p "${repo_dir}"
        (cd "${repo_dir}" && dvc init --no-scm)
    else
        echo "DVC repository config is missing. Add ${ARTIFACTS_DIR}/.dvc/config or set DVC_REMOTE_URL." >&2
        exit 1
    fi

    if [[ -n "${DVC_REMOTE_URL}" ]]; then
        remote_name="${DVC_REMOTE:-models}"
        (
            cd "${repo_dir}"
            dvc remote add --force --default "${remote_name}" "${DVC_REMOTE_URL}"
            if [[ -n "${DVC_S3_ENDPOINT_URL}" ]]; then
                dvc remote modify "${remote_name}" endpointurl "${DVC_S3_ENDPOINT_URL}"
            fi
        )
        DVC_REMOTE="${remote_name}"
    elif [[ -n "${DVC_S3_ENDPOINT_URL}" ]]; then
        if [[ -z "${DVC_REMOTE}" ]]; then
            echo "DVC_REMOTE is required when DVC_S3_ENDPOINT_URL is set." >&2
            exit 1
        fi
        (
            cd "${repo_dir}"
            dvc remote modify "${DVC_REMOTE}" endpointurl "${DVC_S3_ENDPOINT_URL}"
        )
    fi

    local dvc_args=(pull "${target}")
    if [[ -n "${DVC_REMOTE}" ]]; then
        dvc_args+=(--remote "${DVC_REMOTE}")
    fi
    if [[ -n "${DVC_JOBS}" ]]; then
        dvc_args+=(--jobs "${DVC_JOBS}")
    fi
    echo "Materializing model from ${DVC_FILE}"
    (cd "${repo_dir}" && dvc "${dvc_args[@]}")
}

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    pull_dvc_model
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "DVC pull completed, but trained model was not created: ${MODEL_PATH}" >&2
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
