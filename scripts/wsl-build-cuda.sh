#!/usr/bin/env bash
# Build FeLLM with CUDA under WSL2 (cudarc host + cuda-oxide plugin).
# Usage: wsl -e bash scripts/wsl-build-cuda.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> FeLLM CUDA build (WSL)"
echo "    root: $ROOT"

echo "==> Host binary with backend-cuda feature"
# Package is fellm-cli; binary name is fellm.
cargo build -p fellm-cli --release --features backend-cuda
cargo build -p fellm-server --release --features backend-cuda

PLUGIN_DIR="$ROOT/plugins/cuda_kernels"
mkdir -p "$ROOT/plugins/dist"
# Remove any stale broken plugin that registered incorrect Q4_K stubs.
rm -f "$ROOT/plugins/dist/libcuda_kernels.so"

if [[ -d "$PLUGIN_DIR" ]]; then
  echo "==> Kernel plugin (cuda-oxide)"
  if cargo oxide --help >/dev/null 2>&1; then
    (cd "$PLUGIN_DIR" && cargo oxide build --release)
  else
    echo "    cargo-oxide not installed; building ABI-only plugin (0 ops → CPU kernels)"
    echo "    install: cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide"
    (cd "$PLUGIN_DIR" && cargo build --release)
  fi
  for cand in \
    "$PLUGIN_DIR/target/release/libcuda_kernels.so" \
    "$ROOT/target/release/libcuda_kernels.so"; do
    if [[ -f "$cand" ]]; then
      cp -f "$cand" "$ROOT/plugins/dist/"
      echo "    installed plugins/dist/libcuda_kernels.so"
      break
    fi
  done
fi

echo "==> Done"
echo "    Default: CUDA device + CPU kernels (correct). Enable oxide ops with FELLM_PLUGIN_KERNELS=1"
echo "    Force CPU binary path: FELLM_BACKEND=cpu ./target/release/fellm ..."
