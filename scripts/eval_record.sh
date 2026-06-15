#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

DISPLAY_NUM="${DISPLAY_NUM:-:102}"
SERVER_PORT="${SERVER_PORT:-5566}"
LUA_PORT="${LUA_PORT:-4451}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RECORD_DIR="${RECORD_DIR:-${ROOT_DIR}/results/evals/${RUN_STAMP}}"
RECORD_FPS="${RECORD_FPS:-30}"
DISPLAY_SIZE="${DISPLAY_SIZE:-1280x800}"
RECORD_SIZE="${RECORD_SIZE:-${DISPLAY_SIZE}}"
RECORD_FILE="${RECORD_FILE:-${RECORD_DIR}/eval.mp4}"
RECORD_START_TIMEOUT="${RECORD_START_TIMEOUT:-15}"
EVAL_LOG="${EVAL_LOG:-}"
RUN_START_EPOCH="$(date +%s)"
SAVE_SCAN_MARKER="${SAVE_SCAN_MARKER:-${RECORD_DIR}/.save-scan-start}"
BUILD_DIR="${BUILD_DIR:-$(default_build_dir)}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to record evaluation video." >&2
  exit 1
fi

mkdir -p "${RECORD_DIR}"
touch "${SAVE_SCAN_MARKER}"

export DISPLAY_NUM
export DISPLAY_SIZE
export FREECIV_SAVE_PATH="${FREECIV_SAVE_PATH:-${RECORD_DIR}:${HOME}/.freeciv/saves}"
export FREECIV_SAVE_ON_EXIT="${FREECIV_SAVE_ON_EXIT:-1}"

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

copy_saved_games_to_record_dir() {
  if [ -z "${EVAL_LOG}" ] || [ ! -f "${EVAL_LOG}" ]; then
    return 0
  fi
  local save_name candidate copied=0 dir
  local scan_dirs=()
  while IFS= read -r save_name; do
    [ -n "${save_name}" ] || continue
    save_name="$(basename "${save_name}")"
    for candidate in \
      "${RECORD_DIR}/${save_name}" \
      "${BUILD_DIR}/${save_name}" \
      "${ROOT_DIR}/${save_name}" \
      "${HOME}/.freeciv/saves/${save_name}"; do
      if [ -f "${candidate}" ]; then
        if [ "${candidate}" != "${RECORD_DIR}/${save_name}" ]; then
          cp -f "${candidate}" "${RECORD_DIR}/${save_name}"
        fi
        copied=1
        break
      fi
    done
  done < <(sed -n "s/^Game saved as //p" "${EVAL_LOG}" | awk '{print $1}' | sort -u)
  if [ "${copied}" != "1" ]; then
    for dir in "${RECORD_DIR}" "${BUILD_DIR}" "${ROOT_DIR}" "${HOME}/.freeciv/saves"; do
      [ -d "${dir}" ] && scan_dirs+=("${dir}")
    done
    if [ "${#scan_dirs[@]}" -gt 0 ]; then
      while IFS= read -r candidate; do
        [ -f "${candidate}" ] || continue
        save_name="$(basename "${candidate}")"
        if [ "${candidate}" != "${RECORD_DIR}/${save_name}" ]; then
          cp -f "${candidate}" "${RECORD_DIR}/${save_name}"
        fi
        copied=1
      done < <(
        find "${scan_dirs[@]}" -maxdepth 1 -type f -newer "${SAVE_SCAN_MARKER}" \
          \( -name 'freeciv-*.sav*' -o -name '*.sav' -o -name '*.sav.*' \) \
          -printf '%T@ %p\n' 2>/dev/null \
          | sort -nr \
          | awk 'NR <= 20 { $1 = ""; sub(/^ /, ""); print }'
      )
    fi
  fi
  if [ "${copied}" = "1" ]; then
    echo "Copied saved game(s) to ${RECORD_DIR}"
  fi
}

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

copy_saved_games_to_record_dir

run_end_epoch="$(date +%s)"
run_elapsed=$((run_end_epoch - RUN_START_EPOCH))
echo "Recorded evaluation to ${RECORD_FILE}"
echo "Elapsed: $(format_elapsed "${run_elapsed}") (${run_elapsed}s)"
exit "${eval_status}"
