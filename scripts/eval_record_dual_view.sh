#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

DISPLAY_NUM="${DISPLAY_NUM:-:102}"
OBSERVER_DISPLAY_NUM="${OBSERVER_DISPLAY_NUM:-:$(( ${DISPLAY_NUM#:} + 1 ))}"
SERVER_PORT="${SERVER_PORT:-5566}"
LUA_PORT="${LUA_PORT:-4451}"
OBSERVER_LUA_PORT="${OBSERVER_LUA_PORT:-$((LUA_PORT + 1))}"
HOST="${HOST:-127.0.0.1}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)-eval-dual}"
RECORD_DIR="${RECORD_DIR:-${ROOT_DIR}/results/evals/${RUN_STAMP}}"
RECORD_FPS="${RECORD_FPS:-30}"
DISPLAY_SIZE="${DISPLAY_SIZE:-1920x1080}"
OBSERVER_DISPLAY_SIZE="${OBSERVER_DISPLAY_SIZE:-${DISPLAY_SIZE}}"
DISPLAY_DEPTH="${DISPLAY_DEPTH:-24}"
CLIENT_RESOLUTION="${CLIENT_RESOLUTION:-1920x1080}"
RECORD_SIZE="${RECORD_SIZE:-${CLIENT_RESOLUTION}}"
RECORD_AGENT_FILE="${RECORD_AGENT_FILE:-${RECORD_DIR}/eval-agent.mp4}"
RECORD_GLOBAL_FILE="${RECORD_GLOBAL_FILE:-${RECORD_DIR}/eval-global.mp4}"
RECORD_MAP_ONLY="${RECORD_MAP_ONLY:-1}"
RECORD_AGENT_MAP_FILE="${RECORD_AGENT_MAP_FILE:-${RECORD_DIR}/eval-agent-map.mp4}"
RECORD_GLOBAL_MAP_FILE="${RECORD_GLOBAL_MAP_FILE:-${RECORD_DIR}/eval-global-map.mp4}"
RECORD_MAP_CROP_X="${RECORD_MAP_CROP_X:-293}"
RECORD_MAP_CROP_Y="${RECORD_MAP_CROP_Y:-34}"
RECORD_AGENT_MAP_CROP_BOTTOM="${RECORD_AGENT_MAP_CROP_BOTTOM:-0}"
RECORD_GLOBAL_MAP_CROP_BOTTOM="${RECORD_GLOBAL_MAP_CROP_BOTTOM:-180}"
HEATMAP_VIDEO_FILE="${HEATMAP_VIDEO_FILE:-${RECORD_DIR}/eval-heatmaps.mp4}"
HEATMAP_VIDEO_DIR="${HEATMAP_VIDEO_DIR:-${RECORD_DIR}/heatmaps/videos}"
HEATMAP_TB_DIR="${HEATMAP_TB_DIR:-${RECORD_DIR}/heatmaps/tb}"
HEATMAP_FRAME_DIR="${HEATMAP_FRAME_DIR:-${RECORD_DIR}/heatmaps/frames}"
HEATMAP_METADATA="${HEATMAP_METADATA:-${RECORD_DIR}/heatmaps/heatmap.json}"
HEATMAPS="${HEATMAPS:-0}"
HEATMAP_TAGS="${HEATMAP_TAGS:-belief_units,threat,visible_units,territory}"
HEATMAP_SEPARATE="${HEATMAP_SEPARATE:-1}"
HEATMAP_PANEL_WIDTH="${HEATMAP_PANEL_WIDTH:-640}"
HEATMAP_PANEL_HEIGHT="${HEATMAP_PANEL_HEIGHT:-${CLIENT_RESOLUTION#*x}}"
HEATMAP_TILE_SHAPE="${HEATMAP_TILE_SHAPE:-hex}"
HEATMAP_START_DELAY="${HEATMAP_START_DELAY:-0}"
RECORD_START_TIMEOUT="${RECORD_START_TIMEOUT:-30}"
OBSERVER_NAME="${OBSERVER_NAME:-global0}"
OBSERVER_START_TIMEOUT="${OBSERVER_START_TIMEOUT:-30}"
OBSERVER_OBSERVE_DELAY="${OBSERVER_OBSERVE_DELAY:-2}"
OBSERVER_OBSERVE_RETRIES="${OBSERVER_OBSERVE_RETRIES:-5}"
OBSERVER_CLIENT_HOME="${OBSERVER_CLIENT_HOME:-${RECORD_DIR}/observer-home}"
OBSERVER_ZOOM_SET="${OBSERVER_ZOOM_SET:-TRUE}"
OBSERVER_ZOOM_DEFAULT_LEVEL="${OBSERVER_ZOOM_DEFAULT_LEVEL:-0.40}"
EVAL_LOG="${EVAL_LOG:-${RECORD_DIR}/eval.log}"
RUN_INFO_FILE="${RUN_INFO_FILE:-${RECORD_DIR}/run_info.txt}"
RUN_START_EPOCH="$(date +%s)"
SAVE_SCAN_MARKER="${SAVE_SCAN_MARKER:-${RECORD_DIR}/.save-scan-start}"
MAX_TURNS="${MAX_TURNS:-300}"
FREECIV_GENERATED_MAP="${FREECIV_GENERATED_MAP:-1}"
if [ "${FREECIV_GENERATED_MAP}" = "1" ]; then
  HEATMAP_MAP_WIDTH="${HEATMAP_MAP_WIDTH:-${MAP_WIDTH:-32}}"
  HEATMAP_MAP_HEIGHT="${HEATMAP_MAP_HEIGHT:-${MAP_HEIGHT:-32}}"
else
  HEATMAP_MAP_WIDTH="${HEATMAP_MAP_WIDTH:-${MAP_WIDTH:-16}}"
  HEATMAP_MAP_HEIGHT="${HEATMAP_MAP_HEIGHT:-${MAP_HEIGHT:-16}}"
fi
FREECIV_TAKE_RETRIES="${FREECIV_TAKE_RETRIES:-60}"
FREECIV_TAKE_WAIT="${FREECIV_TAKE_WAIT:-1}"
START_AFTER_TAKE="${START_AFTER_TAKE:-}"
RECORD_START_COMMAND="${RECORD_START_COMMAND:-${START_COMMAND:-start}}"
RECORD_EXTERNAL_START="${RECORD_EXTERNAL_START:-0}"
if [ "${RECORD_EXTERNAL_START}" = "1" ]; then
  START_AFTER_TAKE=0
fi

BUILD_DIR="${BUILD_DIR:-$(default_build_dir)}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
CHECKPOINT="${1:-${CHECKPOINT:-$(latest_checkpoint)}}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to record evaluation video." >&2
  exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe is required to align the heatmap video duration." >&2
  exit 1
fi

mkdir -p "${RECORD_DIR}"
if [ "${HEATMAPS}" = "1" ]; then
  mkdir -p "${HEATMAP_TB_DIR}" "${HEATMAP_FRAME_DIR}" "${HEATMAP_VIDEO_DIR}"
fi
touch "${SAVE_SCAN_MARKER}"

if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT=/path/to/model.checkpoint or pass it as the first argument." >&2
  exit 1
fi

echo "Checkpoint: ${CHECKPOINT}"

write_run_info() {
  local git_rev git_status
  git_rev="$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || true)"
  if [ -z "$(git -C "${ROOT_DIR}" status --porcelain 2>/dev/null)" ]; then
    git_status="clean"
  else
    git_status="dirty"
  fi
  {
    printf 'checkpoint: %s\n' "${CHECKPOINT}"
    printf 'created_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'script: %s\n' "${SCRIPT_DIR}/eval_record_dual_view.sh"
    printf 'args: %s\n' "$*"
    printf 'record_dir: %s\n' "${RECORD_DIR}"
    printf 'eval_log: %s\n' "${EVAL_LOG}"
    printf 'game_name: %s\n' "${EVAL_GAME_NAME:-}"
    printf 'game_index: %s\n' "${EVAL_GAME_INDEX:-}"
    printf 'gpu: %s\n' "${EVAL_GPU:-${CUDA_VISIBLE_DEVICES:-}}"
    printf 'git_rev: %s\n' "${git_rev}"
    printf 'git_status: %s\n' "${git_status}"
    printf 'max_turns: %s\n' "${MAX_TURNS:-}"
    printf 'num_simulations: %s\n' "${NUM_SIMULATIONS:-}"
    printf 'temperature: %s\n' "${TEMPERATURE:-}"
    printf 'freeciv_generated_map: %s\n' "${FREECIV_GENERATED_MAP}"
    printf 'map_width: %s\n' "${MAP_WIDTH:-}"
    printf 'map_height: %s\n' "${MAP_HEIGHT:-}"
    printf 'server_port: %s\n' "${SERVER_PORT}"
    printf 'lua_port: %s\n' "${LUA_PORT}"
    printf 'observer_lua_port: %s\n' "${OBSERVER_LUA_PORT}"
    printf 'display_num: %s\n' "${DISPLAY_NUM}"
    printf 'observer_display_num: %s\n' "${OBSERVER_DISPLAY_NUM}"
    printf 'display_size: %s\n' "${DISPLAY_SIZE}"
    printf 'client_resolution: %s\n' "${CLIENT_RESOLUTION}"
    printf 'record_fps: %s\n' "${RECORD_FPS}"
    printf 'record_size: %s\n' "${RECORD_SIZE}"
    printf 'heatmap_tags: %s\n' "${HEATMAP_TAGS}"
    printf 'heatmap_tile_shape: %s\n' "${HEATMAP_TILE_SHAPE}"
    printf 'heatmap_map_width: %s\n' "${HEATMAP_MAP_WIDTH}"
    printf 'heatmap_map_height: %s\n' "${HEATMAP_MAP_HEIGHT}"
  } >"${RUN_INFO_FILE}"
}

write_run_info "$@"

export DISPLAY_NUM
export DISPLAY_SIZE
export DISPLAY_DEPTH
export CLIENT_RESOLUTION
export SERVER_PORT
export LUA_PORT
export FREECIV_GENERATED_MAP
export FREECIV_TAKE_RETRIES
export FREECIV_TAKE_WAIT
export START_AFTER_TAKE
if [ "${HEATMAPS}" = "1" ]; then
  export FREECIV_BELIEF_TENSORBOARD=1
  export FREECIV_BELIEF_TENSORBOARD_HEX="${FREECIV_BELIEF_TENSORBOARD_HEX:-1}"
  export FREECIV_BELIEF_TENSORBOARD_DIR="${HEATMAP_TB_DIR}"
  export FREECIV_BELIEF_TENSORBOARD_INTERVAL="${FREECIV_BELIEF_TENSORBOARD_INTERVAL:-1}"
else
  export FREECIV_BELIEF_TENSORBOARD=0
fi
export FREECIV_SAVE_PATH="${FREECIV_SAVE_PATH:-${RECORD_DIR}:${HOME}/.freeciv/saves}"
export FREECIV_SAVE_ON_EXIT="${FREECIV_SAVE_ON_EXIT:-1}"
export FREECIV_PROCESS_LOG="${FREECIV_PROCESS_LOG:-${RECORD_DIR}/freeciv-processes.log}"

cleanup_freeciv_all "${SERVER_PORT}" "${LUA_PORT}" "${DISPLAY_NUM}"
cleanup_freeciv_ports "${SERVER_PORT}" "${OBSERVER_LUA_PORT}"
pkill -f "Xvfb ${OBSERVER_DISPLAY_NUM}" >/dev/null 2>&1 || true
rm -f "/tmp/.X11-unix/X${OBSERVER_DISPLAY_NUM#:}"

eval_pid=""
observer_pid=""
observer_xvfb_pid=""
ffmpeg_agent_pid=""
ffmpeg_global_pid=""
observer_display_ready=0
progress_fd_open=0
progress_rows=0

eval_progress_allowed() {
  local value="${EVAL_TERMINAL_PROGRESS_BAR:-1}"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  [ "${value}" != "0" ] && [ "${value}" != "false" ] && [ "${value}" != "no" ] && [ "${value}" != "off" ]
}

init_eval_progress_bar() {
  if [ "${progress_fd_open}" = "1" ]; then
    return 0
  fi
  if ! eval_progress_allowed; then
    return 1
  fi
  if [ ! -t 1 ] || [ ! -e /dev/tty ]; then
    return 1
  fi
  if ! exec 9>/dev/tty 2>/dev/null; then
    return 1
  fi
  progress_fd_open=1
  resize_eval_progress_bar >/dev/null
  return 0
}

resize_eval_progress_bar() {
  if [ "${progress_fd_open}" != "1" ]; then
    return 1
  fi
  local rows columns
  read -r rows columns < <(stty size </dev/tty 2>/dev/null || printf '24 80\n')
  rows="${rows:-24}"
  columns="${columns:-80}"
  if [ "${rows}" -lt 3 ]; then
    rows=3
  fi
  if [ "${columns}" -lt 40 ]; then
    columns=40
  fi
  if [ "${rows}" != "${progress_rows}" ]; then
    printf '\0337\033[1;%dr\033[%d;1H\033[2K\0338' "$((rows - 1))" "${rows}" >&9
    progress_rows="${rows}"
  fi
  printf '%s %s\n' "${rows}" "${columns}"
}

clear_eval_progress_bar() {
  if [ "${progress_fd_open}" != "1" ]; then
    return 0
  fi
  local rows columns
  read -r rows columns < <(stty size </dev/tty 2>/dev/null || printf '%s 80\n' "${progress_rows:-24}")
  rows="${rows:-${progress_rows:-24}}"
  printf '\0337\033[r\033[%d;1H\033[2K\0338' "${rows}" >&9
  exec 9>&-
  progress_fd_open=0
  progress_rows=0
}

eval_turn_progress() {
  local turn=0
  if [ -f "${EVAL_LOG}" ]; then
    turn="$(
      tail -n 300 "${EVAL_LOG}" \
        | sed -n 's/.*\[turn \([0-9][0-9]*\) step.*/\1/p' \
        | tail -n 1
    )"
    turn="${turn:-0}"
  fi
  if [ "${turn}" -lt 0 ]; then
    turn=0
  elif [ "${turn}" -gt "${MAX_TURNS}" ]; then
    turn="${MAX_TURNS}"
  fi
  printf '%s\n' "${turn}"
}

write_eval_progress_bar() {
  init_eval_progress_bar || return 0
  local size rows columns now elapsed turn percent filled text filled_text empty_text
  size="$(resize_eval_progress_bar)" || return 0
  rows="${size%% *}"
  columns="${size##* }"
  now="$(date +%s)"
  elapsed=$((now - RUN_START_EPOCH))
  turn="$(eval_turn_progress)"
  percent=0
  if [ "${MAX_TURNS}" -gt 0 ]; then
    percent=$((turn * 100 / MAX_TURNS))
  fi
  filled=$((percent * columns / 100))
  if [ "${percent}" -gt 0 ] && [ "${filled}" -eq 0 ]; then
    filled=1
  fi
  text=" Eval Turn ${turn}/${MAX_TURNS} ${percent}% | elapsed $(format_elapsed "${elapsed}") | out ${RECORD_DIR}"
  if [ "${#text}" -gt "${columns}" ]; then
    text="${text:0:columns}"
  fi
  text="$(printf '%-*s' "${columns}" "${text}")"
  filled_text="${text:0:filled}"
  empty_text="${text:filled}"
  printf '\0337\033[%d;1H\033[2K\033[7m%s\033[0m%s\0338' "${rows}" "${filled_text}" "${empty_text}" >&9
}

stop_recording() {
  if [ -n "${ffmpeg_agent_pid}" ] && kill -0 "${ffmpeg_agent_pid}" >/dev/null 2>&1; then
    kill -INT "${ffmpeg_agent_pid}" >/dev/null 2>&1 || true
    wait "${ffmpeg_agent_pid}" >/dev/null 2>&1 || true
  fi
  if [ -n "${ffmpeg_global_pid}" ] && kill -0 "${ffmpeg_global_pid}" >/dev/null 2>&1; then
    kill -INT "${ffmpeg_global_pid}" >/dev/null 2>&1 || true
    wait "${ffmpeg_global_pid}" >/dev/null 2>&1 || true
  fi
  if [ -n "${observer_pid}" ] && kill -0 "${observer_pid}" >/dev/null 2>&1; then
    kill -TERM "-${observer_pid}" >/dev/null 2>&1 || kill -TERM "${observer_pid}" >/dev/null 2>&1 || true
    wait "${observer_pid}" >/dev/null 2>&1 || true
  fi
  if [ -n "${observer_xvfb_pid}" ] && kill -0 "${observer_xvfb_pid}" >/dev/null 2>&1; then
    kill -TERM "${observer_xvfb_pid}" >/dev/null 2>&1 || true
    wait "${observer_xvfb_pid}" >/dev/null 2>&1 || true
  fi
  pkill -f "Xvfb ${OBSERVER_DISPLAY_NUM}" >/dev/null 2>&1 || true
  if [ -n "${eval_pid}" ] && kill -0 "${eval_pid}" >/dev/null 2>&1; then
    kill -TERM "${eval_pid}" >/dev/null 2>&1 || true
    wait "${eval_pid}" >/dev/null 2>&1 || true
  fi
}

trap 'stop_recording; clear_eval_progress_bar' EXIT INT TERM

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

wait_tcp() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  "${PYTHON}" - "$host" "$port" "$timeout" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.monotonic() + float(sys.argv[3])
while time.monotonic() < deadline:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.3)
    try:
        sock.connect((host, port))
        sock.close()
        raise SystemExit(0)
    except OSError:
        time.sleep(0.25)
    finally:
        try:
            sock.close()
        except OSError:
            pass
raise SystemExit(1)
PY
}

send_chat_command() {
  local port="$1"
  local command="$2"
  local label="$3"
  "${PYTHON}" - "${HOST}" "${port}" "${command}" "${label}" <<'PY'
import sys
import time
from freeciv_sim.remote.lua_client import LuaRemoteClient

host = sys.argv[1]
port = int(sys.argv[2])
cmd = sys.argv[3]
label = sys.argv[4]
client = LuaRemoteClient(host, port, timeout=2.5)
deadline = time.monotonic() + 15.0
last_exc = None
while time.monotonic() < deadline:
    try:
        client.connect()
        break
    except Exception as exc:
        last_exc = exc
        time.sleep(0.5)
else:
    raise SystemExit(f"{label} LuaRemote connect failed: {last_exc}")

if not cmd.startswith("/"):
    cmd = "/" + cmd
safe_cmd = LuaRemoteClient._quote_lua_string(cmd)
safe_label = LuaRemoteClient._quote_lua_string(label)
lua = (
    "return (function() "
    f"local cmd={safe_cmd}; local label={safe_label}; local ok=false; "
    "if type(send_chat)=='function' then ok=pcall(send_chat, cmd) end; "
    "if (not ok) and chat and type(chat.send)=='function' then ok=pcall(chat.send, cmd) end; "
    "if (not ok) and client and type(client.send_chat)=='function' then ok=pcall(client.send_chat, cmd) end; "
    "if chat and chat.base then if ok then chat.base('__OK__ '..label) else chat.base('__ERR__ '..label) end end; "
    "return ok and '__OK__' or '__ERR__' "
    "end)()"
)
try:
    result = client.eval(lua)
    payload = result.last_return() if result else None
    client.close()
    if not (isinstance(payload, str) and payload.startswith("__OK__")):
        raise SystemExit(f"{label} {cmd} failed: {payload!r}")
except Exception as exc:
    try:
        client.close()
    except Exception:
        pass
    raise SystemExit(str(exc))
PY
}

send_observe_command() {
  send_chat_command "${OBSERVER_LUA_PORT}" observe observe_cmd
}

send_start_command() {
  send_chat_command "${LUA_PORT}" "${RECORD_START_COMMAND}" start_cmd
}

send_start_command_with_retry() {
  local retries="${START_COMMAND_RETRIES:-10}"
  local wait_seconds="${START_COMMAND_RETRY_WAIT:-1}"
  local attempt
  for ((attempt = 1; attempt <= retries; attempt++)); do
    if send_start_command; then
      return 0
    fi
    sleep "${wait_seconds}"
  done
  return 1
}

prepare_observer_client_home() {
  local freeciv_dir="${OBSERVER_CLIENT_HOME}/.freeciv"
  local rc_file="${freeciv_dir}/freeciv-client-rc-3.2"
  local source_rc="${HOME}/.freeciv/freeciv-client-rc-3.2"
  local tmp_file="${rc_file}.tmp"

  mkdir -p "${freeciv_dir}"
  if [ ! -f "${rc_file}" ]; then
    if [ -f "${source_rc}" ]; then
      cp "${source_rc}" "${rc_file}"
    else
      printf '[client]\nversion="3.2.1"\n' >"${rc_file}"
    fi
  fi

  awk \
    -v zoom_set="${OBSERVER_ZOOM_SET}" \
    -v zoom_default_level="${OBSERVER_ZOOM_DEFAULT_LEVEL}" '
      BEGIN {
        seen_zoom_set = 0
        seen_zoom_default_level = 0
      }
      /^zoom_set=/ {
        print "zoom_set=" zoom_set
        seen_zoom_set = 1
        next
      }
      /^zoom_default_level=/ {
        print "zoom_default_level=" zoom_default_level
        seen_zoom_default_level = 1
        next
      }
      { print }
      END {
        if (!seen_zoom_set) {
          print "zoom_set=" zoom_set
        }
        if (!seen_zoom_default_level) {
          print "zoom_default_level=" zoom_default_level
        }
      }
    ' "${rc_file}" >"${tmp_file}"
  mv "${tmp_file}" "${rc_file}"
}

wait_display() {
  local display_num="$1"
  local timeout="$2"
  local display_socket="/tmp/.X11-unix/X${display_num#:}"
  for _ in $(seq 1 "${timeout}"); do
    if [ -S "${display_socket}" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_x_ready() {
  local display_num="$1"
  local timeout="$2"
  for _ in $(seq 1 "${timeout}"); do
    if DISPLAY="${display_num}" xdpyinfo >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_ffmpeg() {
  local display_num="$1"
  local output_file="$2"
  ffmpeg \
    -hide_banner \
    -loglevel warning \
    -y \
    -f x11grab \
    -draw_mouse 0 \
    -video_size "${RECORD_SIZE}" \
    -framerate "${RECORD_FPS}" \
    -i "${display_num}" \
    -c:v libx264 \
    -preset veryfast \
    -pix_fmt yuv420p \
    "${output_file}" &
}

start_ffmpeg_with_retry() {
  local display_num="$1"
  local output_file="$2"
  local pid_var="$3"
  local pid=""
  for _ in $(seq 1 10); do
    start_ffmpeg "${display_num}" "${output_file}"
    pid="$!"
    sleep 0.5
    if kill -0 "${pid}" >/dev/null 2>&1; then
      printf -v "${pid_var}" '%s' "${pid}"
      return 0
    fi
    wait "${pid}" >/dev/null 2>&1 || true
    sleep 0.5
  done
  echo "Failed to start ffmpeg recording for ${display_num}." >&2
  return 1
}

render_map_only_video() {
  local input_file="$1"
  local output_file="$2"
  local crop_bottom="$3"
  local record_width="${RECORD_SIZE%x*}"
  local record_height="${RECORD_SIZE#*x}"
  local crop_filter

  if [ "${RECORD_MAP_ONLY}" != "1" ] || [ ! -f "${input_file}" ]; then
    return 0
  fi

  crop_filter="crop=iw-${RECORD_MAP_CROP_X}:ih-${RECORD_MAP_CROP_Y}-${crop_bottom}:${RECORD_MAP_CROP_X}:${RECORD_MAP_CROP_Y},scale=${record_width}:${record_height}:flags=lanczos:force_original_aspect_ratio=decrease,pad=${record_width}:${record_height}:(ow-iw)/2:(oh-ih)/2:black"
  if ffmpeg \
    -hide_banner \
    -loglevel warning \
    -y \
    -i "${input_file}" \
    -vf "${crop_filter}" \
    -an \
    -c:v libx264 \
    -preset veryfast \
    -crf 20 \
    -pix_fmt yuv420p \
    "${output_file}"; then
    echo "Rendered map-only view to ${output_file}"
  else
    echo "Warning: failed to render map-only view from ${input_file}" >&2
  fi
}

if [ -n "${EVAL_LOG}" ]; then
  mkdir -p "$(dirname "${EVAL_LOG}")"
  "${SCRIPT_DIR}/eval_headless.sh" "${CHECKPOINT}" "${@:2}" >"${EVAL_LOG}" 2>&1 &
else
  "${SCRIPT_DIR}/eval_headless.sh" "${CHECKPOINT}" "${@:2}" &
fi
eval_pid="$!"

while ! wait_display "${DISPLAY_NUM}" 1; do
  if ! kill -0 "${eval_pid}" >/dev/null 2>&1; then
    wait "${eval_pid}"
    exit $?
  fi
  RECORD_START_TIMEOUT=$((RECORD_START_TIMEOUT - 1))
  if [ "${RECORD_START_TIMEOUT}" -le 0 ]; then
    echo "Display ${DISPLAY_NUM} did not become available for recording." >&2
    exit 1
  fi
done

if ! wait_x_ready "${DISPLAY_NUM}" "${RECORD_START_TIMEOUT}"; then
  echo "Display ${DISPLAY_NUM} did not become ready for recording." >&2
  exit 1
fi
start_ffmpeg_with_retry "${DISPLAY_NUM}" "${RECORD_AGENT_FILE}" ffmpeg_agent_pid

if wait_tcp "${HOST}" "${SERVER_PORT}" "${RECORD_START_TIMEOUT}"; then
  Xvfb "${OBSERVER_DISPLAY_NUM}" -screen 0 "${OBSERVER_DISPLAY_SIZE}x${DISPLAY_DEPTH}" -nolisten tcp >/tmp/freeciv-muzero-observer-xvfb.log 2>&1 &
  observer_xvfb_pid="$!"
  if ! wait_display "${OBSERVER_DISPLAY_NUM}" "${OBSERVER_START_TIMEOUT}"; then
    echo "Warning: observer display ${OBSERVER_DISPLAY_NUM} did not become available." >&2
  else
    observer_display_ready=1
    if ! wait_x_ready "${OBSERVER_DISPLAY_NUM}" "${OBSERVER_START_TIMEOUT}"; then
      echo "Warning: observer display ${OBSERVER_DISPLAY_NUM} did not become ready for recording." >&2
      observer_display_ready=0
    fi
  fi
fi

if [ "${observer_display_ready}" = "1" ]; then
  prepare_observer_client_home
  (
    set -euo pipefail
    export HOME="${OBSERVER_CLIENT_HOME}"
    export DISPLAY="${OBSERVER_DISPLAY_NUM}"
    export ENABLE_LUAREMOTE=1
    export FREECIV_LUAREMOTE_PORT="${OBSERVER_LUA_PORT}"
    export FREECIV_PORT="${OBSERVER_LUA_PORT}"
    export FREECIV_BUILD_DIR="${BUILD_DIR}"
    cd "${BUILD_DIR}"
    printf '\n--- process: observer freeciv-gtk3.22 server_port=%s luaremote_port=%s ---\n' \
      "${SERVER_PORT}" "${OBSERVER_LUA_PORT}"
    exec ./run.sh freeciv-gtk3.22 \
      -a -s "${HOST}" -p "${SERVER_PORT}" -n "${OBSERVER_NAME}" -P none \
      -- --resolution "${CLIENT_RESOLUTION}"
  ) >>"${FREECIV_PROCESS_LOG}" 2>&1 &
  observer_pid="$!"
  if wait_tcp "${HOST}" "${OBSERVER_LUA_PORT}" "${OBSERVER_START_TIMEOUT}"; then
    observe_sent=0
    for _ in $(seq 1 "${OBSERVER_OBSERVE_RETRIES}"); do
      sleep "${OBSERVER_OBSERVE_DELAY}"
      if send_observe_command; then
        observe_sent=1
        break
      fi
    done
	    if [ "${observe_sent}" != "1" ]; then
	      echo "Warning: observer client started but /observe command failed." >&2
	    fi
	  else
	    echo "Warning: observer LuaRemote ${OBSERVER_LUA_PORT} did not become available." >&2
	  fi
	  if [ "${observe_sent:-0}" = "1" ]; then
	    start_ffmpeg_with_retry "${OBSERVER_DISPLAY_NUM}" "${RECORD_GLOBAL_FILE}" ffmpeg_global_pid
	  fi
	  if [ "${observe_sent:-0}" = "1" ] && [ "${RECORD_EXTERNAL_START}" = "1" ]; then
	    if send_start_command_with_retry; then
	      start_sent=1
	    else
	      echo "Warning: /${RECORD_START_COMMAND} command failed." >&2
	    fi
	  fi
else
  echo "Warning: Freeciv server/display unavailable; skipping global observer." >&2
fi

if [ "${RECORD_EXTERNAL_START}" = "1" ] && [ "${start_sent:-0}" != "1" ]; then
  if send_start_command_with_retry; then
    start_sent=1
  else
    echo "Warning: /${RECORD_START_COMMAND} command failed without observer." >&2
  fi
fi

if [ "${observer_display_ready}" != "1" ]; then
  echo "Warning: skipping global recording; display ${OBSERVER_DISPLAY_NUM} is unavailable." >&2
elif [ "${observe_sent:-0}" != "1" ]; then
  echo "Warning: skipping global recording; observer did not enter observe mode." >&2
fi

set +e
while kill -0 "${eval_pid}" >/dev/null 2>&1; do
  write_eval_progress_bar
  sleep 1
done
wait "${eval_pid}"
eval_status=$?
set -e
write_eval_progress_bar
clear_eval_progress_bar

kill -INT "${ffmpeg_agent_pid}" >/dev/null 2>&1 || true
wait "${ffmpeg_agent_pid}" >/dev/null 2>&1 || true
ffmpeg_agent_pid=""
if [ -n "${ffmpeg_global_pid}" ]; then
  kill -INT "${ffmpeg_global_pid}" >/dev/null 2>&1 || true
  wait "${ffmpeg_global_pid}" >/dev/null 2>&1 || true
  ffmpeg_global_pid=""
fi

copy_saved_games_to_record_dir

echo "Recorded agent view to ${RECORD_AGENT_FILE}"
if [ -f "${RECORD_GLOBAL_FILE}" ]; then
  echo "Recorded global view to ${RECORD_GLOBAL_FILE}"
fi
render_map_only_video "${RECORD_AGENT_FILE}" "${RECORD_AGENT_MAP_FILE}" "${RECORD_AGENT_MAP_CROP_BOTTOM}"
render_map_only_video "${RECORD_GLOBAL_FILE}" "${RECORD_GLOBAL_MAP_FILE}" "${RECORD_GLOBAL_MAP_CROP_BOTTOM}"

if [ "${HEATMAPS}" = "1" ]; then
  render_args=(
    "${SCRIPT_DIR}/render_tb_heatmap_panel.py"
    --logdir "${HEATMAP_TB_DIR}"
    --out-dir "${HEATMAP_FRAME_DIR}"
    --tags "${HEATMAP_TAGS}"
    --width "${HEATMAP_PANEL_WIDTH}"
    --height "${HEATMAP_PANEL_HEIGHT}"
    --tile-shape "${HEATMAP_TILE_SHAPE}"
    --map-width "${HEATMAP_MAP_WIDTH}"
    --map-height "${HEATMAP_MAP_HEIGHT}"
    --metadata-out "${HEATMAP_METADATA}"
  )
  if [ "${HEATMAP_SEPARATE}" = "1" ]; then
    render_args+=(--separate)
  fi
  "${PYTHON}" "${render_args[@]}"

  if [ "${HEATMAP_SEPARATE}" = "1" ]; then
    first_tag="${HEATMAP_TAGS%%,*}"
    frame_count="$(find "${HEATMAP_FRAME_DIR}/${first_tag}" -maxdepth 1 -type f -name 'frame_*.png' | wc -l)"
  else
    frame_count="$(find "${HEATMAP_FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.png' | wc -l)"
  fi
  if [ "${frame_count}" -lt 1 ]; then
    echo "No heatmap frames were produced." >&2
    echo "Checkpoint: ${CHECKPOINT}"
    exit "${eval_status}"
  fi

  duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "${RECORD_AGENT_FILE}")"
  heatmap_fps="$(awk -v frames="${frame_count}" -v duration="${duration}" -v delay="${HEATMAP_START_DELAY}" 'BEGIN { active = duration - delay; intervals = frames > 1 ? frames - 1 : 1; if (active > 0) print intervals / active; else print 1 }')"

  render_heatmap_video() {
    local input_pattern="$1"
    local output_file="$2"
    ffmpeg \
      -hide_banner \
      -loglevel warning \
      -y \
      -framerate "${heatmap_fps}" \
      -i "${input_pattern}" \
      -vf "scale=${HEATMAP_PANEL_WIDTH}:${HEATMAP_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,pad=${HEATMAP_PANEL_WIDTH}:${HEATMAP_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,setpts=PTS-STARTPTS,tpad=start_duration=${HEATMAP_START_DELAY}:start_mode=clone" \
      -an \
      -r "${RECORD_FPS}" \
      -c:v libx264 \
      -preset veryfast \
      -pix_fmt yuv420p \
      "${output_file}"
  }

  if [ "${HEATMAP_SEPARATE}" = "1" ]; then
    IFS=',' read -r -a heatmap_tags <<<"${HEATMAP_TAGS}"
    rendered_any=0
    for tag in "${heatmap_tags[@]}"; do
      tag="${tag// /}"
      tag_dir="${HEATMAP_FRAME_DIR}/${tag}"
      if [ ! -d "${tag_dir}" ]; then
        echo "Warning: missing heatmap frame dir ${tag_dir}" >&2
        continue
      fi
      out_file="${HEATMAP_VIDEO_DIR}/eval-heatmap-${tag}.mp4"
      render_heatmap_video "${tag_dir}/frame_%06d.png" "${out_file}"
      echo "Rendered heatmap video to ${out_file}"
      rendered_any=1
    done
    if [ "${rendered_any}" != "1" ]; then
      echo "No heatmap videos were rendered." >&2
    fi
  else
    render_heatmap_video "${HEATMAP_FRAME_DIR}/frame_%06d.png" "${HEATMAP_VIDEO_FILE}"
    echo "Rendered heatmap video to ${HEATMAP_VIDEO_FILE}"
  fi

  echo "Rendered heatmap frames to ${HEATMAP_FRAME_DIR}"
else
  echo "Heatmap output disabled. Set HEATMAPS=1 to render heatmap TensorBoard, frames, and videos."
fi
echo "Checkpoint: ${CHECKPOINT}"
run_end_epoch="$(date +%s)"
run_elapsed=$((run_end_epoch - RUN_START_EPOCH))
{
  printf 'finished_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'elapsed: %s\n' "$(format_elapsed "${run_elapsed}")"
  printf 'elapsed_seconds: %s\n' "${run_elapsed}"
  printf 'exit_status: %s\n' "${eval_status}"
} >>"${RUN_INFO_FILE}"
echo "Elapsed: $(format_elapsed "${run_elapsed}") (${run_elapsed}s)"
sync_results_to_drive "${RECORD_DIR}"
exit "${eval_status}"
