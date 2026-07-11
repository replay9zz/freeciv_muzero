#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)-easy-ai-trajectories}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/imitation/${RUN_STAMP}}"
SNAPSHOTS_FILE="${SNAPSHOTS_FILE:-${OUT_DIR}/snapshots.jsonl}"
OUT_FILE="${OUT_FILE:-${SNAPSHOTS_FILE}}"
ACTIONLOG_FILE="${ACTIONLOG_FILE:-${OUT_DIR}/actionlog.jsonl}"
ACTIONLOG_ACTIONS_FILE="${ACTIONLOG_ACTIONS_FILE:-${OUT_DIR}/actionlog_actions.jsonl}"
SAMPLES_FILE="${SAMPLES_FILE:-${OUT_DIR}/imitation_samples.jsonl}"
SERVER_RC="${SERVER_RC:-${ROOT_DIR}/start_generated_32x32.serv}"
RULESET_DIR="${RULESET_DIR:-civ2civ3_actionlog}"
SERVER_PORT="${SERVER_PORT:-5566}"
LUA_PORT="${LUA_PORT:-4451}"
HOST="${HOST:-127.0.0.1}"
DISPLAY_NUM="${DISPLAY_NUM:-:104}"
DISPLAY_SIZE="${DISPLAY_SIZE:-1280x800}"
DISPLAY_DEPTH="${DISPLAY_DEPTH:-24}"
MAX_TURNS="${MAX_TURNS:-100}"
POLL_INTERVAL="${POLL_INTERVAL:-0.5}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"
TURN_TIMEOUT="${TURN_TIMEOUT:-1}"
CLIENT_NAME="${CLIENT_NAME:-observer0}"
BUILD_DIR="${BUILD_DIR:-$(default_build_dir)}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
SERVER_LOG="${SERVER_LOG:-${OUT_DIR}/server.log}"
CLIENT_LOG="${CLIENT_LOG:-${OUT_DIR}/client.log}"
XVFB_LOG="${XVFB_LOG:-${OUT_DIR}/xvfb.log}"
FREECIV_SOURCE_DATA="${FREECIV_SOURCE_DATA:-${ROOT_DIR}/../freeciv/data}"
BUILD_SAMPLES="${BUILD_SAMPLES:-1}"

mkdir -p "${OUT_DIR}"
TMP_RC="${OUT_DIR}/start-ai.serv"
cp "${SERVER_RC}" "${TMP_RC}"
if grep -Eq '^[[:space:]]*rulesetdir[[:space:]]+' "${TMP_RC}"; then
  sed -i -E "s|^[[:space:]]*rulesetdir[[:space:]].*|rulesetdir ${RULESET_DIR}|" "${TMP_RC}"
else
  printf 'rulesetdir %s\n' "${RULESET_DIR}" | cat - "${TMP_RC}" >"${TMP_RC}.tmp"
  mv "${TMP_RC}.tmp" "${TMP_RC}"
fi
if ! grep -Eq '^[[:space:]]*set[[:space:]]+timeout[[:space:]]+' "${TMP_RC}"; then
  printf '\nset timeout %s\n' "${TURN_TIMEOUT}" >>"${TMP_RC}"
fi
SERVER_FIFO="${OUT_DIR}/server.stdin"
rm -f "${SERVER_FIFO}"
mkfifo "${SERVER_FIFO}"
exec 8<>"${SERVER_FIFO}"

server_pid=""
client_pid=""
xvfb_pid=""

cleanup() {
  if [ -n "${client_pid}" ] && kill -0 "${client_pid}" >/dev/null 2>&1; then
    kill -TERM "-${client_pid}" >/dev/null 2>&1 || kill -TERM "${client_pid}" >/dev/null 2>&1 || true
    wait "${client_pid}" >/dev/null 2>&1 || true
  fi
  if [ -n "${server_pid}" ] && kill -0 "${server_pid}" >/dev/null 2>&1; then
    kill -TERM "-${server_pid}" >/dev/null 2>&1 || kill -TERM "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
  if [ -n "${xvfb_pid}" ] && kill -0 "${xvfb_pid}" >/dev/null 2>&1; then
    kill -TERM "${xvfb_pid}" >/dev/null 2>&1 || true
    wait "${xvfb_pid}" >/dev/null 2>&1 || true
  fi
  exec 8>&- 8<&- || true
  rm -f "${SERVER_FIFO}"
  cleanup_freeciv_all "${SERVER_PORT}" "${LUA_PORT}" "${DISPLAY_NUM}"
}
trap cleanup EXIT INT TERM

cleanup_freeciv_all "${SERVER_PORT}" "${LUA_PORT}" "${DISPLAY_NUM}"

Xvfb "${DISPLAY_NUM}" -screen 0 "${DISPLAY_SIZE}x${DISPLAY_DEPTH}" -nolisten tcp >"${XVFB_LOG}" 2>&1 &
xvfb_pid="$!"
sleep 1

(
  cd "${BUILD_DIR}"
  if [ -d "${FREECIV_SOURCE_DATA}" ]; then
    export FREECIV_DATA_PATH="${FREECIV_DATA_PATH:-${HOME}/.freeciv/3.2:${FREECIV_SOURCE_DATA}}"
  fi
  exec ./run.sh freeciv-server -p "${SERVER_PORT}" -r "${TMP_RC}"
) <"${SERVER_FIFO}" >"${SERVER_LOG}" 2>&1 &
server_pid="$!"

sleep 3

(
  cd "${BUILD_DIR}"
  export DISPLAY="${DISPLAY_NUM}"
  export FREECIV_LUAREMOTE_PORT="${LUA_PORT}"
  if [ -d "${FREECIV_SOURCE_DATA}" ]; then
    export FREECIV_DATA_PATH="${FREECIV_DATA_PATH:-${HOME}/.freeciv/3.2:${FREECIV_SOURCE_DATA}}"
  fi
  exec ./run.sh freeciv-gtk3.22 -a -s "${HOST}" -p "${SERVER_PORT}" -n "${CLIENT_NAME}" -P none -- --resolution "${DISPLAY_SIZE}"
) >"${CLIENT_LOG}" 2>&1 &
client_pid="$!"

"${PYTHON}" - "${HOST}" "${LUA_PORT}" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.connect((host, port))
        sock.close()
        raise SystemExit(0)
    except OSError:
        time.sleep(0.5)
    finally:
        try:
            sock.close()
        except OSError:
            pass
raise SystemExit(1)
PY

"${PYTHON}" - "${HOST}" "${LUA_PORT}" <<'PY' || true
import sys
from freeciv_sim.remote.lua_client import LuaRemoteClient

host = sys.argv[1]
port = int(sys.argv[2])
client = LuaRemoteClient(host, port, timeout=2.5)
client.connect()

def send(command: str) -> None:
    safe = LuaRemoteClient._quote_lua_string(command)
    lua = (
        "return (function() "
        f"local cmd={safe}; local ok=false; "
        "if type(send_chat)=='function' then ok=pcall(send_chat, cmd) end; "
        "if (not ok) and chat and type(chat.send)=='function' then ok=pcall(chat.send, cmd) end; "
        "if (not ok) and client and type(client.send_chat)=='function' then ok=pcall(client.send_chat, cmd) end; "
        "return ok and '__OK__' or '__ERR__' "
        "end)()"
    )
    result = client.eval(lua)
    payload = result.last_return() if result else None
    if not (isinstance(payload, str) and payload.startswith("__OK__")):
        raise SystemExit(f"{command} failed: {payload!r}")

try:
    send("/observe")
finally:
    client.close()
PY

printf 'start\n' >&8

"${PYTHON}" -m freeciv_sim.imitation.collect_ai_trajectories \
  --host "${HOST}" \
  --port "${LUA_PORT}" \
  --out "${OUT_FILE}" \
  --max-turns "${MAX_TURNS}" \
  --poll-interval "${POLL_INTERVAL}" \
  --progress-interval "${PROGRESS_INTERVAL}"

if [ "${OUT_FILE}" != "${OUT_DIR}/trajectories.jsonl" ]; then
  ln -sf "$(basename "${OUT_FILE}")" "${OUT_DIR}/trajectories.jsonl"
fi

"${PYTHON}" "${ROOT_DIR}/scripts/export_actionlog_jsonl.py" \
  --log-file "${SERVER_LOG}" \
  --out-jsonl "${ACTIONLOG_FILE}" \
  --out-actions "${ACTIONLOG_ACTIONS_FILE}"

if [ "${BUILD_SAMPLES}" = "1" ]; then
  "${PYTHON}" -m freeciv_sim.imitation.build_imitation_dataset \
    --snapshots "${SNAPSHOTS_FILE}" \
    --actionlog "${ACTIONLOG_FILE}" \
    --actions "${ACTIONLOG_ACTIONS_FILE}" \
    --out "${SAMPLES_FILE}"
fi

cat >"${OUT_DIR}/metadata.json" <<EOF
{
  "run_stamp": "${RUN_STAMP}",
  "ruleset": "${RULESET_DIR}",
  "server_rc": "${TMP_RC}",
  "snapshots": "${SNAPSHOTS_FILE}",
  "actionlog": "${ACTIONLOG_FILE}",
  "actionlog_actions": "${ACTIONLOG_ACTIONS_FILE}",
  "samples": "${SAMPLES_FILE}"
}
EOF

echo "Snapshots: ${SNAPSHOTS_FILE}"
echo "Actionlog: ${ACTIONLOG_FILE}"
echo "Actionlog actions: ${ACTIONLOG_ACTIONS_FILE}"
echo "Imitation samples: ${SAMPLES_FILE}"
