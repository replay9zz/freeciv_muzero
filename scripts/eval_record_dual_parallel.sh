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
GPU_LIST="${GPU_LIST:-0,1,2,3,4}"
GAMES="${GAMES:-20}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
GAMES_PER_BATCH="${GAMES_PER_BATCH:-5}"
BASE_SERVER_PORT="${BASE_SERVER_PORT:-5566}"
BASE_LUA_PORT="${BASE_LUA_PORT:-4451}"
BASE_DISPLAY="${BASE_DISPLAY:-102}"
PORT_STRIDE="${PORT_STRIDE:-10}"
DISPLAY_STRIDE="${DISPLAY_STRIDE:-2}"
MAX_TURNS="${MAX_TURNS:-300}"
RECORD_FPS="${RECORD_FPS:-5}"
DISPLAY_SIZE="${DISPLAY_SIZE:-1920x1080}"
CLIENT_RESOLUTION="${CLIENT_RESOLUTION:-${DISPLAY_SIZE}}"
RUN_HEIGHT="${RUN_HEIGHT:-${CLIENT_RESOLUTION#*x}}"
RUN_LABEL="${RUN_LABEL:-eval-${MAX_TURNS}t-${GAMES}g-${RUN_HEIGHT}p${RECORD_FPS}fps}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)-${RUN_LABEL}}"
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

if [ -z "${MAX_PARALLEL}" ]; then
  MAX_PARALLEL="${GAMES_PER_BATCH}"
  if [ "${MAX_PARALLEL}" -gt "${#gpus[@]}" ]; then
    MAX_PARALLEL="${#gpus[@]}"
  fi
fi

mkdir -p "${OUT_DIR}"

echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUT_DIR}"
echo "Games: ${GAMES}"
echo "GPU_LIST: ${GPU_LIST}"
echo "MAX_PARALLEL: ${MAX_PARALLEL}"
echo "Run stamp: ${RUN_STAMP}"

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
  job_name="$(printf 'game-%02d' "$((idx + 1))")"
  job_dir="${OUT_DIR}/${job_name}"
  mkdir -p "${job_dir}"
  job_dirs+=("${job_dir}")

  echo "Start ${job_name}: gpu=${gpu} server=${server_port} lua=${lua_port} display=${display}/${observer_display}"
  (
    echo "GPU: ${gpu}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export MAX_TURNS
    export RECORD_FPS
    export DISPLAY_SIZE
    export CLIENT_RESOLUTION
    export SERVER_PORT="${server_port}"
    export LUA_PORT="${lua_port}"
    export DISPLAY_NUM="${display}"
    export OBSERVER_DISPLAY_NUM="${observer_display}"
    export RECORD_DIR="${job_dir}"
    export EVAL_LOG="${job_dir}/eval.log"
    export JSON="${JSON:-1}"
    export JSON_OUT="${job_dir}/remote_play.jsonl"
    export SCORE_LOG="${job_dir}/scores.jsonl"
    export SCORE_LOG_INTERVAL="${SCORE_LOG_INTERVAL:-1}"
    export TURN_SCORE_CSV="${job_dir}/turn_scores.csv"
    export RUN_STAMP="${RUN_STAMP}-${job_name}"
    export EVAL_GAME_NAME="${job_name}"
    export EVAL_GAME_INDEX="$((idx + 1))"
    export EVAL_GPU="${gpu}"
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
