#!/usr/bin/env bash
# Build the cuda-oxide kernel plugin only. The host is a separate stable build.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGIN_DIR="$ROOT/plugins/cuda_kernels"
DIST="$ROOT/plugins/dist"
OXIDE_NIGHTLY="nightly-2026-04-03"
CODEGEN="$HOME/.cargo/cuda-oxide/librustc_codegen_cuda.so"
diagnostics="$(mktemp)"
trap 'rm -f "$diagnostics"' EXIT

error() {
  printf 'error: %s\n' "$*" >&2
}

run_quiet_or_report() {
  local label="$1"
  shift
  : >"$diagnostics"
  if ! "$@" >"$diagnostics" 2>&1; then
    error "$label"
    cat "$diagnostics" >&2
    return 1
  fi
}

[[ -d "$PLUGIN_DIR" ]] || { error "missing $PLUGIN_DIR"; exit 1; }
# shellcheck source=/dev/null
[[ -f "$ROOT/scripts/fellm-oxide-env.sh" ]] && . "$ROOT/scripts/fellm-oxide-env.sh"

if [[ -z "${CUDA_TOOLKIT_PATH:-}" ]]; then
  for candidate in "$HOME/.local/cuda-12.8" "$HOME/.local/cuda" /usr/local/cuda /usr; do
    if [[ -f "$candidate/include/cuda.h" ]]; then
      export CUDA_TOOLKIT_PATH="$candidate"
      break
    fi
  done
fi
if [[ -n "${CUDA_TOOLKIT_PATH:-}" ]]; then
  export CUDA_HOME="$CUDA_TOOLKIT_PATH"
  export PATH="$CUDA_TOOLKIT_PATH/bin:${PATH:-}"
fi
[[ -d "$HOME/.local/llvm/bin" ]] && export PATH="$HOME/.local/llvm/bin:${PATH:-}"
[[ -d /usr/lib/llvm-21/bin ]] && export PATH="/usr/lib/llvm-21/bin:${PATH:-}"
if [[ -d "$HOME/.local/lib" ]]; then
  export LIBRARY_PATH="$HOME/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$HOME/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if ! command -v cargo-oxide >/dev/null 2>&1 && ! cargo oxide --help >/dev/null 2>&1; then
  error 'cargo-oxide is missing from PATH; run scripts/wsl-setup-oxide.sh'
  exit 1
fi

cd "$PLUGIN_DIR"
: >"$diagnostics"
if ! cargo +"$OXIDE_NIGHTLY" oxide setup >"$diagnostics" 2>&1 && [[ ! -f "$CODEGEN" ]]; then
  error 'cargo oxide setup failed and the codegen backend is missing'
  cat "$diagnostics" >&2
  exit 1
fi
run_quiet_or_report 'cargo oxide doctor failed' cargo +"$OXIDE_NIGHTLY" oxide doctor
cargo +"$OXIDE_NIGHTLY" oxide build -- --release

SO="$PLUGIN_DIR/target/release/libcuda_kernels.so"
if [[ ! -f "$SO" ]]; then
  error "libcuda_kernels.so was not produced at $SO"
  exit 1
fi
mkdir -p "$DIST"
cp -f "$SO" "$DIST/libcuda_kernels.so"
