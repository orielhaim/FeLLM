#!/usr/bin/env bash
# Build the stable host and the isolated cuda-oxide plugin in WSL2.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "$ROOT/scripts/wsl-build-host.sh" --cuda
bash "$ROOT/scripts/wsl-build-plugin.sh"
