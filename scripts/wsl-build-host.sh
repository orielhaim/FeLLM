#!/usr/bin/env bash
# Pipeline A — FeLLM host (stable Rust from repo-root rust-toolchain.toml).
# Never pulls cuda-oxide nightly. Builds CLI/server with optional backend-cuda
# (cudarc host only); kernel .so must come from wsl-build-plugin.sh.
#
# Usage: wsl -e bash scripts/wsl-build-host.sh [--cuda]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WITH_CUDA=0
for arg in "$@"; do
  case "$arg" in
    --cuda) WITH_CUDA=1 ;;
  esac
done

echo "==> FeLLM host build (stable)"
echo "    toolchain: $(rustc --version 2>/dev/null || true)"
echo "    root rust-toolchain.toml pins host; plugins/cuda_kernels is excluded"

if [[ "$WITH_CUDA" -eq 1 ]]; then
  cargo build -p fellm-cli --release --features backend-cuda
  cargo build -p fellm-server --release --features backend-cuda
else
  cargo build -p fellm-cli --release
  cargo build -p fellm-server --release
fi

echo "==> Host done"
echo "    binary: $ROOT/target/release/fellm"
if [[ "$WITH_CUDA" -eq 1 ]]; then
  echo "    CUDA host ready; load kernels from plugins/dist/ after:"
  echo "      bash scripts/wsl-build-plugin.sh"
fi
