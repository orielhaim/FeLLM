# FeLLM oxide plugin env (Pipeline B)
# Source from bash: `. ~/.fellm-oxide-env` or via scripts/wsl-build-plugin.sh
export CUDA_TOOLKIT_PATH="${HOME}/.local/cuda-12.8"
export CUDA_HOME="${CUDA_TOOLKIT_PATH}"
export PATH="${CUDA_TOOLKIT_PATH}/bin:${HOME}/.local/llvm/bin:${PATH}"
export LIBRARY_PATH="${HOME}/.local/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${HOME}/.local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
