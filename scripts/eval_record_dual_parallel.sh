#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

if [ "$#" -gt 0 ]; then
  CHECKPOINT="$1"
  shift
else
  CHECKPOINT="${CHECKPOINT:-$(latest_checkpoint)}"
fi
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9}"
GAMES="${GAMES:-}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
BASE_SERVER_PORT="${BASE_SERVER_PORT:-5566}"
BASE_LUA_PORT="${BASE_LUA_PORT:-4451}"
BASE_DISPLAY="${BASE_DISPLAY:-102}"
PORT_STRIDE="${PORT_STRIDE:-10}"
DISPLAY_STRIDE="${DISPLAY_STRIDE:-2}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)-dual-parallel}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/evals/${RUN_STAMP}}"

if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT=/path/to/model.checkpoint or pass it as the first argument." >&2
  exit 1
fi

IFS=',' read -r -a gpus <<<"${GPU_LIST}"
if [ "${#gpus[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty." >&2
  exit 1
fi

if [ -z "${GAMES}" ]; then
  GAMES="${#gpus[@]}"
fi
if [ -z "${MAX_PARALLEL}" ]; then
  MAX_PARALLEL="${#gpus[@]}"
fi

mkdir -p "${OUT_DIR}"

echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUT_DIR}"
echo "Games: ${GAMES}"
echo "GPU_LIST: ${GPU_LIST}"
echo "MAX_PARALLEL: ${MAX_PARALLEL}"

pids=()
job_dirs=()

cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -TERM "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup INT TERM

wait_for_slot() {
  while [ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]; do
    local next_pids=()
    local pid
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        next_pids+=("${pid}")
      fi
    done
    pids=("${next_pids[@]}")
    if [ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]; then
      sleep 2
    fi
  done
}

for ((idx = 0; idx < GAMES; idx++)); do
  wait_for_slot

  gpu="${gpus[$((idx % ${#gpus[@]}))]}"
  server_port=$((BASE_SERVER_PORT + idx * PORT_STRIDE))
  lua_port=$((BASE_LUA_PORT + idx * PORT_STRIDE))
  display=":$((BASE_DISPLAY + idx * DISPLAY_STRIDE))"
  observer_display=":$((BASE_DISPLAY + idx * DISPLAY_STRIDE + 1))"
  job_name="$(printf 'game-%02d-gpu-%s' "$((idx + 1))" "${gpu}")"
  job_dir="${OUT_DIR}/${job_name}"
  mkdir -p "${job_dir}"
  job_dirs+=("${job_dir}")

  echo "Start ${job_name}: server=${server_port} lua=${lua_port} display=${display}/${observer_display}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export SERVER_PORT="${server_port}"
    export LUA_PORT="${lua_port}"
    export DISPLAY_NUM="${display}"
    export OBSERVER_DISPLAY_NUM="${observer_display}"
    export RECORD_DIR="${job_dir}"
    export EVAL_LOG="${job_dir}/eval.log"
    export RUN_STAMP="${RUN_STAMP}-${job_name}"
    export FREECIV_TAKE_RETRIES="${FREECIV_TAKE_RETRIES:-60}"
    export FREECIV_TAKE_WAIT="${FREECIV_TAKE_WAIT:-1}"
    exec "${SCRIPT_DIR}/eval_record_dual_view.sh" "${CHECKPOINT}" "$@"
  ) >"${job_dir}/runner.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "Done: ${OUT_DIR}"
for job_dir in "${job_dirs[@]}"; do
  if [ -f "${job_dir}/eval-agent.mp4" ] || [ -f "${job_dir}/eval-global.mp4" ]; then
    echo "${job_dir}"
  else
    echo "FAILED_OR_NO_VIDEO ${job_dir}"
  fi
done

exit "${status}"
