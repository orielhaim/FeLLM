#!/usr/bin/env bash
# Orchestrator: host (stable) + oxide plugin (nightly) as two separate pipelines.
#
#   Pipeline A  scripts/wsl-build-host.sh --cuda   → fellm with cudarc
#   Pipeline B  scripts/wsl-build-plugin.sh        → plugins/dist/libcuda_kernels.so
#
# Usage: wsl -e bash scripts/wsl-build-cuda.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> FeLLM CUDA dual-pipeline build"
echo "    A) host  = stable ($(cat "$ROOT/rust-toolchain.toml" | tr '\n' ' '))"
echo "    B) plugin = oxide nightly (plugins/cuda_kernels/rust-toolchain.toml)"

bash "$ROOT/scripts/wsl-build-host.sh" --cuda
bash "$ROOT/scripts/wsl-build-plugin.sh"

echo "==> Done"
echo "    Default: CUDA device + CPU kernels (correct)."
echo "    Enable oxide ops when registered: FELLM_PLUGIN_KERNELS=1"
echo "    Force CPU: FELLM_BACKEND=cpu ./target/release/fellm ..."
