#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

FREECIV_GENERATED_MAP="${FREECIV_GENERATED_MAP:-1}"
if [ "${FREECIV_GENERATED_MAP}" = "1" ]; then
  SERVER_RC="${SERVER_RC:-${ROOT_DIR}/start_generated_32x32.serv}"
  MAP_WIDTH="${MAP_WIDTH:-32}"
  MAP_HEIGHT="${MAP_HEIGHT:-32}"
else
  SERVER_RC="${SERVER_RC:-${ROOT_DIR}/start_single.serv}"
  MAP_WIDTH="${MAP_WIDTH:-16}"
  MAP_HEIGHT="${MAP_HEIGHT:-16}"
fi

HOST="${HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-5566}"
LUA_PORT_A="${LUA_PORT_A:-4451}"
LUA_PORT_B="${LUA_PORT_B:-4452}"
DISPLAY_NUM="${DISPLAY_NUM:-:103}"
DISPLAY_SIZE="${DISPLAY_SIZE:-1280x800}"
DISPLAY_DEPTH="${DISPLAY_DEPTH:-24}"
CLIENT_RESOLUTION="${CLIENT_RESOLUTION:-${DISPLAY_SIZE}}"
MAX_TURNS="${MAX_TURNS:-300}"
MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-100}"
FREECIV_MAX_UNITS="${FREECIV_MAX_UNITS:-}"
FREECIV_MAX_CITIES="${FREECIV_MAX_CITIES:-}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-50}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SLEEP="${SLEEP:-0.1}"
DEVICE="${DEVICE:-}"
NO_SEA_UNITS="${NO_SEA_UNITS:-1}"
FREECIV_BELIEF_TENSORBOARD="${FREECIV_BELIEF_TENSORBOARD:-1}"
FREECIV_BELIEF_TENSORBOARD_INTERVAL="${FREECIV_BELIEF_TENSORBOARD_INTERVAL:-5}"
FREECIV_REWARD_TENSORBOARD="${FREECIV_REWARD_TENSORBOARD:-1}"
PLAYER_ID_A="${PLAYER_ID_A:-0}"
PLAYER_ID_B="${PLAYER_ID_B:-1}"
CLIENT_NAME_A="${CLIENT_NAME_A:-agent0}"
CLIENT_NAME_B="${CLIENT_NAME_B:-agent1}"
START_AFTER_TAKE="${START_AFTER_TAKE:-1}"
REMOTE_START_AFTER_TAKE="${REMOTE_START_AFTER_TAKE:-0}"
DUAL_START_DELAY="${DUAL_START_DELAY:-3}"
FREECIV_START_WAIT="${FREECIV_START_WAIT:-5}"
FREECIV_TAKE_RETRIES="${FREECIV_TAKE_RETRIES:-120}"
FREECIV_TAKE_WAIT="${FREECIV_TAKE_WAIT:-1}"
XVFB_PATTERN="Xvfb ${DISPLAY_NUM}"

BUILD_DIR="${BUILD_DIR:-$(default_build_dir)}"
SCENARIO_PATH="${SCENARIO_PATH:-$(default_scenario_path)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-dual-$$}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/remote_play/${RUN_ID}}"
SERVER_RC_DIR="${SERVER_RC_DIR:-/tmp/freeciv-muzero-rc-${RUN_ID}}"
FREECIV_AIFILL="${FREECIV_AIFILL:-2}"
export FREECIV_AIFILL
export FREECIV_START_WAIT
export FREECIV_TAKE_RETRIES
export FREECIV_TAKE_WAIT

SERVER_RC_TEMPLATE="$(prepare_server_rc_template \
  "${SERVER_RC}" \
  "${SERVER_RC_DIR}" \
  "${SERVER_PORT}" \
  "1" \
  "1" \
  "${RUN_ID}")"
SERVER_RC_FILE="${SERVER_RC_TEMPLATE//\{server_port\}/${SERVER_PORT}}"

CHECKPOINT_A="${1:-${CHECKPOINT_A:-$(latest_checkpoint)}}"
CHECKPOINT_B="${2:-${CHECKPOINT_B:-${CHECKPOINT_A}}}"

if [ -z "${CHECKPOINT_A}" ] || [ ! -f "${CHECKPOINT_A}" ]; then
  echo "Checkpoint A not found: ${CHECKPOINT_A}" >&2
  exit 1
fi
if [ -z "${CHECKPOINT_B}" ] || [ ! -f "${CHECKPOINT_B}" ]; then
  echo "Checkpoint B not found: ${CHECKPOINT_B}" >&2
  exit 1
fi

cleanup() {
  local pids
  pids="$(jobs -pr || true)"
  if [ -n "${pids}" ]; then
    kill ${pids} >/dev/null 2>&1 || true
    wait ${pids} >/dev/null 2>&1 || true
  fi
  cleanup_freeciv_all "${SERVER_PORT}" "${LUA_PORT_A}" "${DISPLAY_NUM}"
  cleanup_freeciv_all "${SERVER_PORT}" "${LUA_PORT_B}" "${DISPLAY_NUM}"
  pkill -f "${XVFB_PATTERN}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

cleanup
mkdir -p "${OUTPUT_DIR}"

Xvfb "${DISPLAY_NUM}" -screen 0 "${DISPLAY_SIZE}x${DISPLAY_DEPTH}" -nolisten tcp >"${OUTPUT_DIR}/xvfb.log" 2>&1 &
sleep 1
export DISPLAY="${DISPLAY_NUM}"

cd "${ROOT_DIR}"
init_python_env

if [ "${FREECIV_GENERATED_MAP}" = "1" ]; then
  (cd "${BUILD_DIR}" && ./run.sh freeciv-server -p "${SERVER_PORT}" -r "${SERVER_RC_FILE}") >"${OUTPUT_DIR}/server.log" 2>&1 &
else
  (cd "${BUILD_DIR}" && ./run.sh freeciv-server -p "${SERVER_PORT}" -f "${SCENARIO_PATH}" -r "${SERVER_RC_FILE}") >"${OUTPUT_DIR}/server.log" 2>&1 &
fi
sleep 2

common_args=(
  --host "${HOST}"
  --map-width "${MAP_WIDTH}"
  --map-height "${MAP_HEIGHT}"
  --max-turns "${MAX_TURNS}"
  --max-actions-per-turn "${MAX_ACTIONS_PER_TURN}"
  --num-simulations "${NUM_SIMULATIONS}"
  --temperature "${TEMPERATURE}"
  --sleep "${SLEEP}"
  --episodes 1
  --json
)

if [ -n "${FREECIV_MAX_UNITS}" ]; then
  common_args+=(--max-units "${FREECIV_MAX_UNITS}")
fi
if [ -n "${FREECIV_MAX_CITIES}" ]; then
  common_args+=(--max-cities "${FREECIV_MAX_CITIES}")
fi
if [ "${NO_SEA_UNITS}" = "1" ]; then
  common_args+=(--no-sea-units)
fi
if [ "${FREECIV_BELIEF_TENSORBOARD}" = "1" ]; then
  common_args+=(--belief-tensorboard --belief-tensorboard-interval "${FREECIV_BELIEF_TENSORBOARD_INTERVAL}")
fi
if [ "${REMOTE_START_AFTER_TAKE}" = "1" ]; then
  common_args+=(--start-after-take)
fi
if [ -n "${DEVICE}" ]; then
  common_args+=(--device "${DEVICE}")
fi

send_start_after_agents_ready() {
  "${PYTHON:-${ROOT_DIR}/.venv/bin/python}" - "${HOST}" "${LUA_PORT_A}" "${LUA_PORT_B}" "${DUAL_START_DELAY}" <<'PY'
import socket
import sys
import time

from freeciv_sim.remote.lua_client import LuaRemoteClient

host = sys.argv[1]
ports = [int(sys.argv[2]), int(sys.argv[3])]
delay = float(sys.argv[4])

def wait_port(port: int) -> None:
    deadline = time.monotonic() + 30.0
    last_exc = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            sock.close()
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(0.5)
        finally:
            try:
                sock.close()
            except OSError:
                pass
    raise SystemExit(f"LuaRemote port {port} not ready: {last_exc}")

for port in ports:
    wait_port(port)
time.sleep(max(0.0, delay))

client = LuaRemoteClient(host, ports[0], timeout=2.5)
client.connect()
safe_cmd = LuaRemoteClient._quote_lua_string("/start")
lua = (
    "return (function() "
    f"local cmd={safe_cmd}; local ok=false; "
    "if type(send_chat)=='function' then ok=pcall(send_chat, cmd) end; "
    "if (not ok) and chat and type(chat.send)=='function' then ok=pcall(chat.send, cmd) end; "
    "if (not ok) and client and type(client.send_chat)=='function' then ok=pcall(client.send_chat, cmd) end; "
    "if chat and chat.base then if ok then chat.base('__OK__ dual_start') else chat.base('__ERR__ dual_start') end end; "
    "return ok and '__OK__' or '__ERR__' "
    "end)()"
)
res = client.eval(lua)
payload = res.last_return() if res else None
client.close()
if not (isinstance(payload, str) and payload.startswith("__OK__")):
    raise SystemExit(f"/start failed: {payload!r}")
PY
}

FREECIV_CLIENT_CMD="${BUILD_DIR}/run.sh freeciv-gtk3.22 -a -s ${HOST} -p ${SERVER_PORT} -n ${CLIENT_NAME_A} -P none -- --resolution ${CLIENT_RESOLUTION}" \
FREECIV_TAKE_PLAYER_ID="${PLAYER_ID_A}" \
FREECIV_REWARD_TENSORBOARD="${FREECIV_REWARD_TENSORBOARD}" \
python remote_play.py \
  --checkpoint "${CHECKPOINT_A}" \
  --port "${LUA_PORT_A}" \
  --player-id "${PLAYER_ID_A}" \
  --belief-tensorboard-dir "${OUTPUT_DIR}/agent${PLAYER_ID_A}_tb" \
  --json-out "${OUTPUT_DIR}/agent${PLAYER_ID_A}.jsonl" \
  "${common_args[@]}" \
  >"${OUTPUT_DIR}/agent${PLAYER_ID_A}.log" 2>&1 &
pid_a=$!

FREECIV_CLIENT_CMD="${BUILD_DIR}/run.sh freeciv-gtk3.22 -a -s ${HOST} -p ${SERVER_PORT} -n ${CLIENT_NAME_B} -P none -- --resolution ${CLIENT_RESOLUTION}" \
FREECIV_TAKE_PLAYER_ID="${PLAYER_ID_B}" \
FREECIV_REWARD_TENSORBOARD="${FREECIV_REWARD_TENSORBOARD}" \
python remote_play.py \
  --checkpoint "${CHECKPOINT_B}" \
  --port "${LUA_PORT_B}" \
  --player-id "${PLAYER_ID_B}" \
  --belief-tensorboard-dir "${OUTPUT_DIR}/agent${PLAYER_ID_B}_tb" \
  --json-out "${OUTPUT_DIR}/agent${PLAYER_ID_B}.jsonl" \
  "${common_args[@]}" \
  >"${OUTPUT_DIR}/agent${PLAYER_ID_B}.log" 2>&1 &
pid_b=$!

if [ "${START_AFTER_TAKE}" = "1" ]; then
  send_start_after_agents_ready >>"${OUTPUT_DIR}/dual-start.log" 2>&1 || true
fi

status_a=0
status_b=0
wait "${pid_a}" || status_a=$?
wait "${pid_b}" || status_b=$?

echo "Dual-agent output: ${OUTPUT_DIR}"
exit $((status_a != 0 || status_b != 0))
