#!/usr/bin/env bash
# Install the cuda-oxide *plugin* toolchain in WSL2.
# This is intentionally separate from the FeLLM host (stable 1.97).
#
# Requirements (from https://github.com/NVlabs/cuda-oxide):
#   - Linux (Ubuntu 24.04 tested)
#   - Rust nightly-2026-04-03 + rust-src, rustc-dev, llvm-tools
#   - cargo-oxide (installed with that nightly)
#   - CUDA Toolkit 12.x+ (nvcc + cuda.h)
#   - Clang 21 + libclang-*-21-dev (bindgen for cuda-bindings)
#   - LLVM 21+ with NVPTX (llc-21), optional but recommended
#
# Usage:
#   wsl -e bash scripts/wsl-setup-oxide.sh
#   wsl -e bash scripts/wsl-setup-oxide.sh --apt   # also apt-install CUDA/clang/llvm (needs sudo)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OXIDE_NIGHTLY="nightly-2026-04-03"
CUDA_OXIDE_REV="1f4d813719012d384f2db12b88efc9314c8bf50c"
CARGO_OXIDE_MARKER="${HOME}/.cargo/.fellm-cuda-oxide-rev"
DO_APT=0
for arg in "$@"; do
  case "$arg" in
    --apt) DO_APT=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
  esac
done

echo "==> FeLLM oxide plugin toolchain setup"
echo "    root: $ROOT"
echo "    oxide nightly: $OXIDE_NIGHTLY"
echo "    host toolchain (untouched): $(cat "$ROOT/rust-toolchain.toml" | tr '\n' ' ')"

# ---------------------------------------------------------------------------
# 1) Rust nightly (no sudo)
# ---------------------------------------------------------------------------
echo "==> [1/5] Rust $OXIDE_NIGHTLY + components"
rustup toolchain install "$OXIDE_NIGHTLY"
rustup component add rust-src rustc-dev llvm-tools rustfmt rust-analyzer clippy \
  --toolchain "$OXIDE_NIGHTLY"
rustup run "$OXIDE_NIGHTLY" rustc --version

# ---------------------------------------------------------------------------
# 2) cargo-oxide (no sudo; built with the pinned nightly)
# ---------------------------------------------------------------------------
echo "==> [2/5] cargo-oxide"
if ! command -v cargo-oxide >/dev/null 2>&1 || \
  [[ ! -f "$CARGO_OXIDE_MARKER" ]] || \
  [[ "$(cat "$CARGO_OXIDE_MARKER")" != "$CUDA_OXIDE_REV" ]]; then
  cargo +"$OXIDE_NIGHTLY" install --git https://github.com/NVlabs/cuda-oxide.git \
    --rev "$CUDA_OXIDE_REV" cargo-oxide --locked --force
  mkdir -p "$(dirname "$CARGO_OXIDE_MARKER")"
  printf '%s\n' "$CUDA_OXIDE_REV" >"$CARGO_OXIDE_MARKER"
fi
cargo oxide --help >/dev/null

# ---------------------------------------------------------------------------
# 3) CUDA toolkit + Clang (user-local by default; --apt uses sudo)
# ---------------------------------------------------------------------------
USER_CUDA="${HOME}/.local/cuda-12.8"
USER_LLVM="${HOME}/.local/llvm"

install_user_cuda() {
  if [[ -f "$USER_CUDA/include/cuda.h" ]]; then
    echo "    user CUDA already at $USER_CUDA"
    return 0
  fi
  echo "    downloading CUDA 12.8 toolkit runfile (toolkit only, no driver)..."
  mkdir -p "${HOME}/.local/oxide-deps/src"
  local run="cuda_12.8.0_570.86.10_linux.run"
  local path="${HOME}/.local/oxide-deps/src/$run"
  if [[ ! -f "$path" ]]; then
    wget -q --show-progress \
      "https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/$run" \
      -O "$path"
  fi
  chmod +x "$path"
  echo "    extracting toolkit → $USER_CUDA"
  sh "$path" --silent --toolkit --toolkitpath="$USER_CUDA" \
    --defaultroot="$USER_CUDA" --no-opengl-libs --override --no-man-page
}

install_user_clang() {
  if [[ -x "$USER_LLVM/bin/clang" ]]; then
    echo "    user clang already at $USER_LLVM"
    return 0
  fi
  echo "    downloading LLVM 21 official binary (clang + resource headers)..."
  mkdir -p "${HOME}/.local/oxide-deps/src"
  local ver="21.1.8"
  local archive="clang+llvm-${ver}-x86_64-linux-gnu-ubuntu-24.04.tar.xz"
  local url="https://github.com/llvm/llvm-project/releases/download/llvmorg-${ver}/${archive}"
  local path="${HOME}/.local/oxide-deps/src/$archive"
  if [[ ! -f "$path" ]]; then
    # Fallback naming if ubuntu-24.04 asset missing
    if ! wget -q --show-progress "$url" -O "$path"; then
      archive="clang+llvm-${ver}-x86_64-linux-gnu-ubuntu-22.04.tar.xz"
      url="https://github.com/llvm/llvm-project/releases/download/llvmorg-${ver}/${archive}"
      path="${HOME}/.local/oxide-deps/src/$archive"
      wget -q --show-progress "$url" -O "$path"
    fi
  fi
  mkdir -p "${HOME}/.local"
  tar -xJf "$path" -C "${HOME}/.local"
  local extracted
  extracted="$(find "${HOME}/.local" -maxdepth 1 -type d -name "clang+llvm-${ver}-*" | head -1)"
  rm -rf "$USER_LLVM"
  mv "$extracted" "$USER_LLVM"
  echo "    installed $USER_LLVM"
}

if [[ "$DO_APT" -eq 1 ]]; then
  echo "==> [3/5] apt packages (sudo)"
  if ! sudo -n true 2>/dev/null; then
    echo "    sudo required for --apt (you will be prompted)"
  fi
  sudo apt-get update
  sudo apt-get install -y \
    lsb-release wget software-properties-common gnupg libffi-dev \
    clang-21 libclang-21-dev libclang-cpp21-dev libclang-common-21-dev \
    || sudo apt-get install -y clang libclang-dev libffi-dev

  if ! sudo apt-get install -y llvm-21; then
    echo "    llvm-21 not in distro; installing via apt.llvm.org"
    tmp="$(mktemp -d)"
    (
      cd "$tmp"
      wget -q https://apt.llvm.org/llvm.sh
      chmod +x llvm.sh
      sudo ./llvm.sh 21
    )
    rm -rf "$tmp"
  fi

  if command -v clang-21 >/dev/null 2>&1; then
    sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100 || true
    sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100 || true
  fi

  if ! command -v nvcc >/dev/null 2>&1 && [[ ! -f /usr/local/cuda/include/cuda.h ]]; then
    echo "    installing CUDA toolkit via NVIDIA apt repo..."
    if [[ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]]; then
      wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
      sudo dpkg -i /tmp/cuda-keyring.deb
      sudo apt-get update
    fi
    sudo apt-get install -y cuda-toolkit-12-8 \
      || sudo apt-get install -y cuda-toolkit
  fi
else
  echo "==> [3/5] user-local CUDA + Clang (no sudo; pass --apt for system packages)"
  install_user_cuda
  install_user_clang
fi

# ---------------------------------------------------------------------------
# 4) PATH / env
# ---------------------------------------------------------------------------
echo "==> [4/5] environment"
if [[ -f "$USER_CUDA/include/cuda.h" ]]; then
  export CUDA_TOOLKIT_PATH="$USER_CUDA"
  export CUDA_HOME="$USER_CUDA"
  export PATH="$USER_CUDA/bin:${PATH:-}"
elif [[ -f /usr/local/cuda/include/cuda.h ]]; then
  export CUDA_TOOLKIT_PATH="/usr/local/cuda"
  export CUDA_HOME="/usr/local/cuda"
  export PATH="/usr/local/cuda/bin:${PATH:-}"
fi
if [[ -d "$USER_LLVM/bin" ]]; then
  export PATH="$USER_LLVM/bin:${PATH:-}"
fi
if [[ -d /usr/lib/llvm-21/bin ]]; then
  export PATH="/usr/lib/llvm-21/bin:${PATH:-}"
fi

# Persist for interactive WSL shells (idempotent block).
ENV_SNIPPET="${HOME}/.fellm-oxide-env"
cat > "$ENV_SNIPPET" <<EOF
# FeLLM oxide plugin env (Pipeline B). Sourced from ~/.bashrc if present.
export CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-}"
export CUDA_HOME="\${CUDA_TOOLKIT_PATH}"
[[ -n "\${CUDA_TOOLKIT_PATH}" ]] && export PATH="\${CUDA_TOOLKIT_PATH}/bin:\${PATH}"
[[ -d "\$HOME/.local/llvm/bin" ]] && export PATH="\$HOME/.local/llvm/bin:\${PATH}"
EOF
if [[ -f "${HOME}/.bashrc" ]] && ! grep -q 'fellm-oxide-env' "${HOME}/.bashrc"; then
  echo "[ -f \"\$HOME/.fellm-oxide-env\" ] && . \"\$HOME/.fellm-oxide-env\"" >> "${HOME}/.bashrc"
  echo "    appended source of ~/.fellm-oxide-env to ~/.bashrc"
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "    nvcc: $(command -v nvcc) ($(nvcc --version | tail -1))"
  echo "    CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_PATH"
else
  echo "    error: nvcc not found" >&2
  exit 1
fi
if command -v clang >/dev/null 2>&1; then
  echo "    clang: $(command -v clang) ($(clang --version | head -1))"
else
  echo "    error: clang not found (bindgen needs it)" >&2
  exit 1
fi
if command -v llc-21 >/dev/null 2>&1; then
  echo "    llc-21: $(command -v llc-21)"
  llc-21 --version 2>/dev/null | grep -i nvptx || true
else
  echo "    note: using rustup llvm-tools llc (doctor already OK for this)"
fi

# ---------------------------------------------------------------------------
# 5) doctor
# ---------------------------------------------------------------------------
echo "==> [5/5] cargo oxide doctor"
cd "$ROOT/plugins/cuda_kernels"
cargo +"$OXIDE_NIGHTLY" oxide doctor
