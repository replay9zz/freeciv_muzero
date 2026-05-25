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
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RECORD_DIR="${RECORD_DIR:-${ROOT_DIR}/results/evals/${RUN_STAMP}}"
RECORD_FPS="${RECORD_FPS:-30}"
DISPLAY_SIZE="${DISPLAY_SIZE:-1920x1080}"
OBSERVER_DISPLAY_SIZE="${OBSERVER_DISPLAY_SIZE:-${DISPLAY_SIZE}}"
DISPLAY_DEPTH="${DISPLAY_DEPTH:-24}"
CLIENT_RESOLUTION="${CLIENT_RESOLUTION:-1920x1080}"
RECORD_SIZE="${RECORD_SIZE:-${CLIENT_RESOLUTION}}"
RECORD_AGENT_FILE="${RECORD_AGENT_FILE:-${RECORD_DIR}/eval-agent.mp4}"
RECORD_GLOBAL_FILE="${RECORD_GLOBAL_FILE:-${RECORD_DIR}/eval-global.mp4}"
HEATMAP_VIDEO_FILE="${HEATMAP_VIDEO_FILE:-${RECORD_DIR}/eval-heatmaps.mp4}"
HEATMAP_TB_DIR="${HEATMAP_TB_DIR:-${RECORD_DIR}/heatmaps/tb}"
HEATMAP_FRAME_DIR="${HEATMAP_FRAME_DIR:-${RECORD_DIR}/heatmaps/frames}"
HEATMAP_METADATA="${HEATMAP_METADATA:-${RECORD_DIR}/heatmaps/heatmap.json}"
HEATMAP_TAGS="${HEATMAP_TAGS:-belief_units,threat,visible_units,territory}"
HEATMAP_PANEL_WIDTH="${HEATMAP_PANEL_WIDTH:-640}"
HEATMAP_PANEL_HEIGHT="${HEATMAP_PANEL_HEIGHT:-${CLIENT_RESOLUTION#*x}}"
HEATMAP_TILE_SHAPE="${HEATMAP_TILE_SHAPE:-hex}"
HEATMAP_START_DELAY="${HEATMAP_START_DELAY:-0}"
RECORD_START_TIMEOUT="${RECORD_START_TIMEOUT:-30}"
OBSERVER_NAME="${OBSERVER_NAME:-global0}"
OBSERVER_START_TIMEOUT="${OBSERVER_START_TIMEOUT:-30}"
EVAL_LOG="${EVAL_LOG:-}"
FREECIV_GENERATED_MAP="${FREECIV_GENERATED_MAP:-1}"
if [ "${FREECIV_GENERATED_MAP}" = "1" ]; then
  HEATMAP_MAP_WIDTH="${HEATMAP_MAP_WIDTH:-${MAP_WIDTH:-16}}"
  HEATMAP_MAP_HEIGHT="${HEATMAP_MAP_HEIGHT:-${MAP_HEIGHT:-16}}"
else
  HEATMAP_MAP_WIDTH="${HEATMAP_MAP_WIDTH:-${MAP_WIDTH:-4}}"
  HEATMAP_MAP_HEIGHT="${HEATMAP_MAP_HEIGHT:-${MAP_HEIGHT:-16}}"
fi
FREECIV_TAKE_RETRIES="${FREECIV_TAKE_RETRIES:-60}"
FREECIV_TAKE_WAIT="${FREECIV_TAKE_WAIT:-1}"

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

mkdir -p "${RECORD_DIR}" "${HEATMAP_TB_DIR}" "${HEATMAP_FRAME_DIR}"

if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT=/path/to/model.checkpoint or pass it as the first argument." >&2
  exit 1
fi

echo "Checkpoint: ${CHECKPOINT}"

export DISPLAY_NUM
export DISPLAY_SIZE
export DISPLAY_DEPTH
export CLIENT_RESOLUTION
export SERVER_PORT
export LUA_PORT
export FREECIV_GENERATED_MAP
export FREECIV_TAKE_RETRIES
export FREECIV_TAKE_WAIT
export FREECIV_BELIEF_TENSORBOARD=1
export FREECIV_BELIEF_TENSORBOARD_HEX="${FREECIV_BELIEF_TENSORBOARD_HEX:-1}"
export FREECIV_BELIEF_TENSORBOARD_DIR="${HEATMAP_TB_DIR}"
export FREECIV_BELIEF_TENSORBOARD_INTERVAL="${FREECIV_BELIEF_TENSORBOARD_INTERVAL:-1}"

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

trap stop_recording EXIT INT TERM

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

send_observe_command() {
  "${PYTHON}" - "${HOST}" "${OBSERVER_LUA_PORT}" <<'PY'
import sys
import time
from freeciv_sim.remote.lua_client import LuaRemoteClient

host = sys.argv[1]
port = int(sys.argv[2])
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
    raise SystemExit(f"observer LuaRemote connect failed: {last_exc}")

cmd = "/observe"
safe_cmd = LuaRemoteClient._quote_lua_string(cmd)
lua = (
    "return (function() "
    f"local cmd={safe_cmd}; local ok=false; "
    "if type(send_chat)=='function' then ok=pcall(send_chat, cmd) end; "
    "if (not ok) and chat and type(chat.send)=='function' then ok=pcall(chat.send, cmd) end; "
    "if (not ok) and client and type(client.send_chat)=='function' then ok=pcall(client.send_chat, cmd) end; "
    "if chat and chat.base then if ok then chat.base('__OK__ observe_cmd') else chat.base('__ERR__ observe_cmd') end end; "
    "return ok and '__OK__' or '__ERR__' "
    "end)()"
)
try:
    result = client.eval(lua)
    payload = result.last_return() if result else None
    client.close()
    if not (isinstance(payload, str) and payload.startswith("__OK__")):
        raise SystemExit(f"observer /observe failed: {payload!r}")
except Exception as exc:
    try:
        client.close()
    except Exception:
        pass
    raise SystemExit(str(exc))
PY
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
  if [ "${observer_display_ready}" = "1" ]; then
    start_ffmpeg_with_retry "${OBSERVER_DISPLAY_NUM}" "${RECORD_GLOBAL_FILE}" ffmpeg_global_pid
  fi
fi

if [ "${observer_display_ready}" = "1" ]; then
  (
    set -euo pipefail
    export DISPLAY="${OBSERVER_DISPLAY_NUM}"
    export ENABLE_LUAREMOTE=1
    export FREECIV_LUAREMOTE_PORT="${OBSERVER_LUA_PORT}"
    export FREECIV_PORT="${OBSERVER_LUA_PORT}"
    export FREECIV_BUILD_DIR="${BUILD_DIR}"
    cd "${BUILD_DIR}"
    exec ./run.sh freeciv-gtk3.22 \
      -a -s "${HOST}" -p "${SERVER_PORT}" -n "${OBSERVER_NAME}" -P none \
      -- --resolution "${CLIENT_RESOLUTION}"
  ) &
  observer_pid="$!"
  if wait_tcp "${HOST}" "${OBSERVER_LUA_PORT}" "${OBSERVER_START_TIMEOUT}"; then
    if ! send_observe_command; then
      echo "Warning: observer client started but /observe command failed." >&2
    fi
  else
    echo "Warning: observer LuaRemote ${OBSERVER_LUA_PORT} did not become available." >&2
  fi
else
  echo "Warning: Freeciv server/display unavailable; skipping global observer." >&2
fi

if [ "${observer_display_ready}" != "1" ]; then
  echo "Warning: skipping global recording; display ${OBSERVER_DISPLAY_NUM} is unavailable." >&2
fi

set +e
wait "${eval_pid}"
eval_status=$?
set -e

kill -INT "${ffmpeg_agent_pid}" >/dev/null 2>&1 || true
wait "${ffmpeg_agent_pid}" >/dev/null 2>&1 || true
ffmpeg_agent_pid=""
if [ -n "${ffmpeg_global_pid}" ]; then
  kill -INT "${ffmpeg_global_pid}" >/dev/null 2>&1 || true
  wait "${ffmpeg_global_pid}" >/dev/null 2>&1 || true
  ffmpeg_global_pid=""
fi

echo "Recorded agent view to ${RECORD_AGENT_FILE}"
if [ -f "${RECORD_GLOBAL_FILE}" ]; then
  echo "Recorded global view to ${RECORD_GLOBAL_FILE}"
fi

"${PYTHON}" "${SCRIPT_DIR}/render_tb_heatmap_panel.py" \
  --logdir "${HEATMAP_TB_DIR}" \
  --out-dir "${HEATMAP_FRAME_DIR}" \
  --tags "${HEATMAP_TAGS}" \
  --width "${HEATMAP_PANEL_WIDTH}" \
  --height "${HEATMAP_PANEL_HEIGHT}" \
  --tile-shape "${HEATMAP_TILE_SHAPE}" \
  --map-width "${HEATMAP_MAP_WIDTH}" \
  --map-height "${HEATMAP_MAP_HEIGHT}" \
  --metadata-out "${HEATMAP_METADATA}"

frame_count="$(find "${HEATMAP_FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.png' | wc -l)"
if [ "${frame_count}" -lt 1 ]; then
  echo "No heatmap frames were produced." >&2
  echo "Checkpoint: ${CHECKPOINT}"
  exit "${eval_status}"
fi

duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "${RECORD_AGENT_FILE}")"
heatmap_fps="$(awk -v frames="${frame_count}" -v duration="${duration}" -v delay="${HEATMAP_START_DELAY}" 'BEGIN { active = duration - delay; intervals = frames > 1 ? frames - 1 : 1; if (active > 0) print intervals / active; else print 1 }')"

ffmpeg \
  -hide_banner \
  -loglevel warning \
  -y \
  -framerate "${heatmap_fps}" \
  -i "${HEATMAP_FRAME_DIR}/frame_%06d.png" \
  -vf "scale=${HEATMAP_PANEL_WIDTH}:${HEATMAP_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,pad=${HEATMAP_PANEL_WIDTH}:${HEATMAP_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,setpts=PTS-STARTPTS,tpad=start_duration=${HEATMAP_START_DELAY}:start_mode=clone" \
  -an \
  -r "${RECORD_FPS}" \
  -c:v libx264 \
  -preset veryfast \
  -pix_fmt yuv420p \
  "${HEATMAP_VIDEO_FILE}"

echo "Rendered heatmap frames to ${HEATMAP_FRAME_DIR}"
echo "Rendered heatmap video to ${HEATMAP_VIDEO_FILE}"
echo "Checkpoint: ${CHECKPOINT}"
exit "${eval_status}"
