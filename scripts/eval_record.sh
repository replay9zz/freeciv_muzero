#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

DISPLAY_NUM="${DISPLAY_NUM:-:102}"
SERVER_PORT="${SERVER_PORT:-5566}"
LUA_PORT="${LUA_PORT:-4451}"
RECORD_DIR="${RECORD_DIR:-${ROOT_DIR}/results/recordings}"
RECORD_FPS="${RECORD_FPS:-30}"
DISPLAY_SIZE="${DISPLAY_SIZE:-1280x800}"
RECORD_SIZE="${RECORD_SIZE:-${DISPLAY_SIZE}}"
RECORD_FILE="${RECORD_FILE:-${RECORD_DIR}/eval-$(date +%Y%m%d-%H%M%S).mp4}"
RECORD_START_TIMEOUT="${RECORD_START_TIMEOUT:-15}"
EVAL_LOG="${EVAL_LOG:-}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to record evaluation video." >&2
  exit 1
fi

mkdir -p "${RECORD_DIR}"

export DISPLAY_NUM
export DISPLAY_SIZE

cleanup_freeciv_all "${SERVER_PORT}" "${LUA_PORT}" "${DISPLAY_NUM}"

eval_pid=""
ffmpeg_pid=""

stop_recording() {
  if [ -n "${ffmpeg_pid}" ] && kill -0 "${ffmpeg_pid}" >/dev/null 2>&1; then
    kill -INT "${ffmpeg_pid}" >/dev/null 2>&1 || true
    wait "${ffmpeg_pid}" >/dev/null 2>&1 || true
  fi
  if [ -n "${eval_pid}" ] && kill -0 "${eval_pid}" >/dev/null 2>&1; then
    kill -TERM "${eval_pid}" >/dev/null 2>&1 || true
    wait "${eval_pid}" >/dev/null 2>&1 || true
  fi
}

trap stop_recording EXIT INT TERM

if [ -n "${EVAL_LOG}" ]; then
  mkdir -p "$(dirname "${EVAL_LOG}")"
  "${SCRIPT_DIR}/eval_headless.sh" "$@" >"${EVAL_LOG}" 2>&1 &
else
  "${SCRIPT_DIR}/eval_headless.sh" "$@" &
fi
eval_pid="$!"

display_socket="/tmp/.X11-unix/X${DISPLAY_NUM#:}"
for _ in $(seq 1 "${RECORD_START_TIMEOUT}"); do
  if [ -S "${display_socket}" ]; then
    break
  fi
  if ! kill -0 "${eval_pid}" >/dev/null 2>&1; then
    wait "${eval_pid}"
    exit $?
  fi
  sleep 1
done

if [ ! -S "${display_socket}" ]; then
  echo "Display ${DISPLAY_NUM} did not become available for recording." >&2
  exit 1
fi

for _ in $(seq 1 5); do
  ffmpeg \
    -hide_banner \
    -loglevel warning \
    -y \
    -f x11grab \
    -draw_mouse 0 \
    -video_size "${RECORD_SIZE}" \
    -framerate "${RECORD_FPS}" \
    -i "${DISPLAY_NUM}" \
    -c:v libx264 \
    -preset veryfast \
    -pix_fmt yuv420p \
    "${RECORD_FILE}" &
  ffmpeg_pid="$!"
  sleep 1
  if kill -0 "${ffmpeg_pid}" >/dev/null 2>&1; then
    break
  fi
  wait "${ffmpeg_pid}" >/dev/null 2>&1 || true
  ffmpeg_pid=""
  sleep 1
done

if [ -z "${ffmpeg_pid}" ]; then
  echo "Failed to start ffmpeg recording for ${DISPLAY_NUM}." >&2
  exit 1
fi

wait "${eval_pid}"
eval_status=$?

kill -INT "${ffmpeg_pid}" >/dev/null 2>&1 || true
wait "${ffmpeg_pid}" >/dev/null 2>&1 || true
ffmpeg_pid=""

echo "Recorded evaluation to ${RECORD_FILE}"
exit "${eval_status}"
