#!/usr/bin/env bash
# Read-only FeLLM CUDA/cuda-oxide environment validation for WSL2.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGIN="$ROOT/plugins/cuda_kernels"
OXIDE_NIGHTLY="nightly-2026-04-03"
failures=0

ok() { printf 'ok:    %s\n' "$*"; }
warn() { printf 'error: %s\n' "$*" >&2; failures=$((failures + 1)); }
note() { printf '       %s\n' "$*"; }

printf 'FeLLM WSL CUDA environment doctor\n'
printf 'repository: %s\n\n' "$ROOT"

if [[ ! -r /proc/sys/kernel/osrelease ]] || ! grep -qi microsoft /proc/sys/kernel/osrelease; then
  warn 'this is not WSL2; cuda-oxide integration for FeLLM must run in Ubuntu on WSL2'
  note 'from PowerShell: wsl --install -d Ubuntu; wsl --set-version Ubuntu 2'
else
  ok "WSL2 ($(uname -r))"
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  ok "NVIDIA WSL driver ($(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1))"
else
  warn 'NVIDIA GPU/WSL driver is unavailable'
  note 'install/update the Windows NVIDIA driver with WSL CUDA support; do not install a Linux display driver in WSL'
fi

CUDA_ROOT="${CUDA_TOOLKIT_PATH:-${CUDA_HOME:-}}"
if command -v nvcc >/dev/null 2>&1; then
  ok "CUDA toolkit ($(nvcc --version | tail -1 | sed 's/^ *//'))"
  [[ -n "$CUDA_ROOT" ]] || CUDA_ROOT="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
else
  warn 'nvcc is missing from PATH (CUDA Toolkit 12.x or newer is required)'
  note 'run: bash scripts/wsl-setup-oxide.sh        # user-local toolkit, no sudo'
  note 'or:  bash scripts/wsl-setup-oxide.sh --apt  # Ubuntu packages, requires sudo'
fi
if [[ -n "$CUDA_ROOT" && -f "$CUDA_ROOT/include/cuda.h" ]]; then
  ok "CUDA headers ($CUDA_ROOT/include/cuda.h)"
  export CUDA_TOOLKIT_PATH="$CUDA_ROOT"
  export CUDA_HOME="$CUDA_ROOT"
elif [[ -n "$CUDA_ROOT" ]]; then
  warn "CUDA_TOOLKIT_PATH/CUDA_HOME points to '$CUDA_ROOT', but include/cuda.h is absent"
fi

if rustup toolchain list | grep -q "^${OXIDE_NIGHTLY}"; then
  ok "cuda-oxide Rust toolchain ($OXIDE_NIGHTLY)"
  for component in rust-src rustc-dev llvm-tools; do
    if (cd "$PLUGIN" && RUSTUP_TOOLCHAIN="$OXIDE_NIGHTLY" rustup component list --installed 2>/dev/null) | grep -q "^${component}"; then
      ok "$component for $OXIDE_NIGHTLY"
    else
      warn "$component is missing for $OXIDE_NIGHTLY"
      note "run: rustup component add $component --toolchain $OXIDE_NIGHTLY"
    fi
  done
else
  warn "pinned cuda-oxide toolchain $OXIDE_NIGHTLY is not installed"
  note "run: rustup toolchain install $OXIDE_NIGHTLY --component rust-src rustc-dev llvm-tools"
fi

LLVM_CONFIG=""
for candidate in llvm-config-21 "$HOME/.local/llvm/bin/llvm-config" /usr/lib/llvm-21/bin/llvm-config; do
  if command -v "$candidate" >/dev/null 2>&1; then LLVM_CONFIG="$(command -v "$candidate")"; break; fi
  if [[ -x "$candidate" ]]; then LLVM_CONFIG="$candidate"; break; fi
done
if [[ -n "$LLVM_CONFIG" ]] && [[ "$($LLVM_CONFIG --version 2>/dev/null)" == 21.* ]]; then
  ok "LLVM $($LLVM_CONFIG --version) ($LLVM_CONFIG)"
else
  warn 'LLVM 21 is missing'
  note 'run: bash scripts/wsl-setup-oxide.sh (installs a user-local LLVM 21 toolchain)'
fi

LLC=""
for candidate in llc-21 "$HOME/.local/llvm/bin/llc" /usr/lib/llvm-21/bin/llc; do
  if command -v "$candidate" >/dev/null 2>&1; then LLC="$(command -v "$candidate")"; break; fi
  if [[ -x "$candidate" ]]; then LLC="$candidate"; break; fi
done
if [[ -n "$LLC" ]] && "$LLC" --version 2>/dev/null | grep -qi nvptx; then
  ok "LLVM NVPTX target ($LLC)"
else
  warn 'LLVM 21 llc with the NVPTX target is missing'
  note 'run: bash scripts/wsl-setup-oxide.sh'
fi

CLANG=""
for candidate in clang-21 "$HOME/.local/llvm/bin/clang" /usr/lib/llvm-21/bin/clang; do
  if command -v "$candidate" >/dev/null 2>&1; then CLANG="$(command -v "$candidate")"; break; fi
  if [[ -x "$candidate" ]]; then CLANG="$candidate"; break; fi
done
if [[ -n "$CLANG" ]]; then
  ok "Clang ($($CLANG --version | head -1))"
  CLANG_ROOT="$(cd "$(dirname "$CLANG")/.." && pwd)"
  if find "$CLANG_ROOT/lib" /usr/lib/llvm-21 -maxdepth 4 \( -name 'libclang.so' -o -name 'libclang.so.*' -o -path '*/lib/clang/21' \) -print -quit 2>/dev/null | grep -q .; then
    ok "libclang headers/runtime ($CLANG_ROOT)"
  else
    warn "libclang was not found under $CLANG_ROOT/lib"
    note 'run: bash scripts/wsl-setup-oxide.sh'
  fi
else
  warn 'Clang 21 is missing'
  note 'run: bash scripts/wsl-setup-oxide.sh'
fi

if command -v cargo-oxide >/dev/null 2>&1; then
  ok "cargo-oxide ($(command -v cargo-oxide))"
else
  warn 'cargo-oxide is missing from PATH'
  note 'run: bash scripts/wsl-setup-oxide.sh'
fi

if [[ -f /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc && -z "${CUDA_OXIDE_LIBDEVICE:-}" ]]; then
  export CUDA_OXIDE_LIBDEVICE=/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc
fi
if [[ -d "$PLUGIN" ]] && command -v cargo-oxide >/dev/null 2>&1; then
  printf '\nRunning cargo oxide doctor in %s\n' "$PLUGIN"
  if (cd "$PLUGIN" && cargo +"$OXIDE_NIGHTLY" oxide doctor); then
    ok 'cargo oxide doctor'
  else
    warn 'cargo oxide doctor failed; use its diagnostics above, then rerun this script'
  fi
fi

printf '\n'
if (( failures > 0 )); then
  printf 'FAILED: %d environment requirement(s) are not satisfied.\n' "$failures" >&2
  printf 'Recommended repair: bash scripts/wsl-setup-oxide.sh\n' >&2
  exit 1
fi
printf 'READY: FeLLM host and cuda-oxide plugin pipelines can be built in WSL2.\n'
