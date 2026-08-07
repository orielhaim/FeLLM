# FeLLM oxide plugin env (Pipeline B)
# Source from bash: `. ~/.fellm-oxide-env` or via scripts/wsl-build-plugin.sh
if [[ -f "${HOME}/.local/cuda-12.8/include/cuda.h" ]]; then
  export CUDA_TOOLKIT_PATH="${HOME}/.local/cuda-12.8"
elif [[ -f /usr/local/cuda/include/cuda.h ]]; then
  export CUDA_TOOLKIT_PATH="/usr/local/cuda"
elif [[ -f /usr/include/cuda.h ]]; then
  # Ubuntu's nvidia-cuda-toolkit uses a split /usr layout.
  export CUDA_TOOLKIT_PATH="/usr"
fi
export CUDA_HOME="${CUDA_TOOLKIT_PATH:-}"
[[ -n "${CUDA_TOOLKIT_PATH:-}" ]] && export PATH="${CUDA_TOOLKIT_PATH}/bin:${PATH}"
[[ -d "${HOME}/.local/llvm/bin" ]] && export PATH="${HOME}/.local/llvm/bin:${PATH}"
if [[ -f /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc ]]; then
  export CUDA_OXIDE_LIBDEVICE=/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc
fi
export LIBRARY_PATH="${HOME}/.local/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${HOME}/.local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# WSL's Windows-provided CUDA driver must precede Ubuntu toolkit shims.
if [[ -f /usr/lib/wsl/lib/libcuda.so.1 ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
fi
