#!/usr/bin/env bash
# Build the stable FeLLM host. cuda_kernels is excluded from this workspace.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WITH_CUDA=0
for arg in "$@"; do
  case "$arg" in
    --cuda) WITH_CUDA=1 ;;
    *) printf 'error: unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

if [[ "$WITH_CUDA" -eq 1 ]]; then
  cargo build -q -p fellm-cli --release --features backend-cuda
  cargo build -q -p fellm-server --release --features backend-cuda
else
  cargo build -q -p fellm-cli --release
  cargo build -q -p fellm-server --release
fi
