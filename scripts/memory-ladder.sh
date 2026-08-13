#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 MODEL [OUTPUT.jsonl]" >&2
  echo "env: FELLM_LADDER_BUDGETS='1G 2G 4G' FELLM_LADDER_REPETITIONS=3 FELLM_LADDER_TOKENS=64" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
model=$(realpath "$1")
output=${2:-memory-ladder.jsonl}
budgets=${FELLM_LADDER_BUDGETS:-"1G 2G 4G"}
repetitions=${FELLM_LADDER_REPETITIONS:-3}
tokens=${FELLM_LADDER_TOKENS:-64}
prompt=${FELLM_LADDER_PROMPT:-"Explain why bounded storage staging matters."}
fellm_bin=${FELLM_LADDER_BIN:-"$(pwd)/target/release/fellm"}

[[ -f "$model" ]] || { echo "model not found: $model" >&2; exit 2; }
[[ -x "$fellm_bin" ]] || { echo "FeLLM binary not executable: $fellm_bin" >&2; exit 2; }
command -v systemd-run >/dev/null || { echo "systemd-run is required" >&2; exit 2; }
command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 2; }
systemctl --user show-environment >/dev/null 2>&1 || {
  echo "a running user systemd manager is required for hard MemoryMax enforcement" >&2
  exit 2
}

workdir=$(mktemp -d /tmp/fellm-memory-ladder.XXXXXX)
cleanup() {
  local resolved
  resolved=$(realpath "$workdir")
  [[ "$resolved" == /tmp/fellm-memory-ladder.* ]] || return
  rm -r -- "$resolved"
}
trap cleanup EXIT

: >"$output"
expected_hash=""

bytes_from_size() {
  local value=${1^^}
  if [[ $value =~ ^([0-9]+)([KMGT]?)B?$ ]]; then
    local number=${BASH_REMATCH[1]}
    case ${BASH_REMATCH[2]} in
      "") echo "$number" ;;
      K) echo "$((number * 1024))" ;;
      M) echo "$((number * 1024 * 1024))" ;;
      G) echo "$((number * 1024 * 1024 * 1024))" ;;
      T) echo "$((number * 1024 * 1024 * 1024 * 1024))" ;;
    esac
    return
  fi
  echo "invalid memory size '$1' (use bytes or K/M/G/T suffixes)" >&2
  return 2
}

run_case() {
  local mode=$1 budget=$2 repetition=$3
  local budget_bytes
  budget_bytes=$(bytes_from_size "$budget")
  local stdout_file="$workdir/${mode}-${budget}-${repetition}.stdout"
  local stderr_file="$workdir/${mode}-${budget}-${repetition}.stderr"
  local time_file="$workdir/${mode}-${budget}-${repetition}.time"
  local -a mode_args=()
  case "$mode" in
    cpu)
      mode_args=(--backend cpu)
      ;;
    cuda-full)
      mode_args=(--backend cuda --cpu-fallback false)
      ;;
    cuda-ram)
      mode_args=(--backend cuda --cpu-fallback false --storage-provider page-cache)
      ;;
    ssd-page-cache)
      mode_args=(--backend cuda --cpu-fallback false --storage-provider page-cache --disable-cpu-partitions true)
      ;;
    ssd-buffered)
      mode_args=(--backend cuda --cpu-fallback false --storage-provider buffered --disable-cpu-partitions true)
      ;;
    *)
      echo "unknown ladder mode: $mode" >&2
      return 2
      ;;
  esac

  local unit="fellm-ladder-${mode//[^a-zA-Z0-9]/}-${budget//[^a-zA-Z0-9]/}-${repetition}-$$"
  local started_ns finished_ns status=0
  started_ns=$(date +%s%N)
  systemd-run --user --quiet --wait --collect --pipe --unit "$unit" \
    --property "MemoryMax=$budget" --property "MemorySwapMax=0" \
    /usr/bin/time -q -f 'peak_rss_kib=%M\nelapsed_seconds=%e' -o "$time_file" \
    "$fellm_bin" --config "$(pwd)/fellm.toml" run "$model" \
    --prompt "$prompt" --completion true --max-tokens "$tokens" \
    --host-memory-limit "$budget_bytes" \
    "${mode_args[@]}" >"$stdout_file" 2>"$stderr_file" || status=$?
  finished_ns=$(date +%s%N)

  local hash peak_rss_kib elapsed_seconds
  hash=$(sha256sum "$stdout_file" | cut -d' ' -f1)
  peak_rss_kib=$(sed -n 's/^peak_rss_kib=//p' "$time_file" 2>/dev/null || true)
  elapsed_seconds=$(sed -n 's/^elapsed_seconds=//p' "$time_file" 2>/dev/null || true)
  if [[ -z "$expected_hash" && $status -eq 0 ]]; then
    expected_hash=$hash
  fi
  local identical=false
  [[ -n "$expected_hash" && "$hash" == "$expected_hash" ]] && identical=true
  printf '{"schema_version":1,"mode":"%s","memory_max":"%s","swap_max":0,"repetition":%d,"exit_code":%d,"output_sha256":"%s","token_identical":%s,"peak_rss_kib":%s,"elapsed_seconds":%s,"wall_nanos":%s}\n' \
    "$mode" "$budget" "$repetition" "$status" "$hash" "$identical" \
    "${peak_rss_kib:-null}" "${elapsed_seconds:-null}" "$((finished_ns-started_ns))" >>"$output"
}

modes=${FELLM_LADDER_MODES:-"cpu cuda-full cuda-ram ssd-page-cache ssd-buffered"}
for budget in $budgets; do
  for mode in $modes; do
    for ((repetition=1; repetition<=repetitions; repetition++)); do
      run_case "$mode" "$budget" "$repetition"
    done
  done
done

echo "memory ladder written to $output"
