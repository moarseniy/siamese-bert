#!/usr/bin/env bash
set -euo pipefail

module="${1:-main.py}"
exec uvicorn "src.${module%.py}:app" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8800}" \
    --workers 1
