#!/usr/bin/env bash
# Pipeline B — cuda-oxide kernel plugin ONLY.
# Uses plugins/cuda_kernels/rust-toolchain.toml (nightly-2026-04-03).
# Does not build fellm-cli / fellm-server.
#
# Prerequisites: bash scripts/wsl-setup-oxide.sh  (and --apt or user-local CUDA/clang)
#
# Usage: wsl -e bash scripts/wsl-build-plugin.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGIN_DIR="$ROOT/plugins/cuda_kernels"
DIST="$ROOT/plugins/dist"
mkdir -p "$DIST"

# Prefer committed env snippet, then ~/.fellm-oxide-env, then heuristics.
if [[ -f "$ROOT/scripts/fellm-oxide-env.sh" ]]; then
  # shellcheck source=/dev/null
  . "$ROOT/scripts/fellm-oxide-env.sh"
elif [[ -f "$HOME/.fellm-oxide-env" ]]; then
  # shellcheck source=/dev/null
  . "$HOME/.fellm-oxide-env"
fi

# Prefer user-local CUDA toolkit if present (no sudo install).
if [[ -z "${CUDA_TOOLKIT_PATH:-}" ]]; then
  for cand in \
    "$HOME/.local/cuda-12.8" \
    "$HOME/.local/cuda" \
    /usr/local/cuda; do
    if [[ -f "$cand/include/cuda.h" ]]; then
      export CUDA_TOOLKIT_PATH="$cand"
      break
    fi
  done
fi
if [[ -n "${CUDA_TOOLKIT_PATH:-}" ]]; then
  export CUDA_HOME="$CUDA_TOOLKIT_PATH"
  export PATH="$CUDA_TOOLKIT_PATH/bin:${PATH:-}"
fi

# User-local clang/llvm + libffi symlink for rustc-codegen-cuda link
if [[ -d "$HOME/.local/llvm/bin" ]]; then
  export PATH="$HOME/.local/llvm/bin:${PATH:-}"
fi
if [[ -d /usr/lib/llvm-21/bin ]]; then
  export PATH="/usr/lib/llvm-21/bin:${PATH:-}"
fi
if [[ -d "$HOME/.local/lib" ]]; then
  export LIBRARY_PATH="$HOME/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$HOME/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "==> FeLLM oxide plugin build"
echo "    plugin: $PLUGIN_DIR"
echo "    CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_PATH:-<unset>}"
echo "    rustc (via plugin rust-toolchain.toml):"

if [[ ! -d "$PLUGIN_DIR" ]]; then
  echo "error: missing $PLUGIN_DIR" >&2
  exit 1
fi

if ! command -v cargo-oxide >/dev/null 2>&1 && ! cargo oxide --help >/dev/null 2>&1; then
  echo "error: cargo-oxide not installed. Run: bash scripts/wsl-setup-oxide.sh" >&2
  exit 1
fi

cd "$PLUGIN_DIR"
# Ensure codegen backend is cached (first run is slow).
# cargo-oxide may warn that Cargo didn't report the .so even when it exists
# under ~/.cargo/cuda-oxide/ — treat that as soft-fail if doctor sees the backend.
if ! cargo oxide setup; then
  if [[ -f "${HOME}/.cargo/cuda-oxide/librustc_codegen_cuda.so" ]]; then
    echo "    note: cargo oxide setup warned, but codegen .so is present; continuing"
  else
    echo "error: cargo oxide setup failed and codegen backend is missing" >&2
    exit 1
  fi
fi
# Fail fast if env is incomplete.
cargo oxide doctor

echo "==> cargo oxide build -- --release"
cargo oxide build -- --release

SO=""
for cand in \
  "$PLUGIN_DIR/target/release/libcuda_kernels.so" \
  "$ROOT/target/release/libcuda_kernels.so"; do
  if [[ -f "$cand" ]]; then
    SO="$cand"
    break
  fi
done
if [[ -z "$SO" ]]; then
  echo "error: libcuda_kernels.so not found after build" >&2
  find "$PLUGIN_DIR/target" -name 'libcuda_kernels.so' 2>/dev/null || true
  exit 1
fi

cp -f "$SO" "$DIST/libcuda_kernels.so"
echo "==> installed $DIST/libcuda_kernels.so"
echo "    Host loads this via FELLM_PLUGIN_DIR / plugins/dist"
echo "    Enable ops: FELLM_PLUGIN_KERNELS=1 FELLM_BACKEND=cuda"
