#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

SERVER_RC="${SERVER_RC:-${ROOT_DIR}/start_single.serv}"
HOST="${HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-5566}"
LUA_PORT="${LUA_PORT:-4451}"
PLAYER_ID="${PLAYER_ID:-0}"
TAKE_PLAYER_ID="${TAKE_PLAYER_ID:-${PLAYER_ID}}"
MAP_WIDTH="${MAP_WIDTH:-4}"
MAP_HEIGHT="${MAP_HEIGHT:-16}"
MAX_TURNS="${MAX_TURNS:-300}"
MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-100}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-50}"
TEMPERATURE="${TEMPERATURE:-0.0}"
EPISODES="${EPISODES:-1}"
CLIENT_NAME="${CLIENT_NAME:-agent0}"
NO_SEA_UNITS="${NO_SEA_UNITS:-1}"
JSON="${JSON:-0}"

BUILD_DIR="${BUILD_DIR:-$(default_build_dir)}"
SCENARIO_PATH="${SCENARIO_PATH:-$(default_scenario_path)}"
CHECKPOINT="${1:-${CHECKPOINT:-$(latest_checkpoint)}}"

if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT=/path/to/model.checkpoint or pass it as the first argument." >&2
  exit 1
fi

cleanup() {
  cleanup_freeciv_all "${SERVER_PORT}" "${LUA_PORT}"
}

trap cleanup EXIT INT TERM

cleanup

cd "${ROOT_DIR}"
init_python_env

export FREECIV_BUILD_DIR="${BUILD_DIR}"
export FREECIV_SERVER_PORT="${SERVER_PORT}"
export FREECIV_SERVER_CMD="${BUILD_DIR}/run.sh freeciv-server -p ${SERVER_PORT} -f ${SCENARIO_PATH} -r ${SERVER_RC}"
export FREECIV_CLIENT_ARGS="-a -s ${HOST} -p ${SERVER_PORT} -n ${CLIENT_NAME} -P none"
export FREECIV_TAKE_PLAYER_ID="${TAKE_PLAYER_ID}"

args=(
  remote_play.py
  --checkpoint "${CHECKPOINT}"
  --client-cmd "bash ${ROOT_DIR}/scripts/freeciv_gui.sh"
  --host "${HOST}"
  --port "${LUA_PORT}"
  --player-id "${PLAYER_ID}"
  --map-width "${MAP_WIDTH}"
  --map-height "${MAP_HEIGHT}"
  --max-turns "${MAX_TURNS}"
  --max-actions-per-turn "${MAX_ACTIONS_PER_TURN}"
  --num-simulations "${NUM_SIMULATIONS}"
  --temperature "${TEMPERATURE}"
  --episodes "${EPISODES}"
)

if [ "${NO_SEA_UNITS}" = "1" ]; then
  args+=(--no-sea-units)
fi

if [ "${JSON}" = "1" ]; then
  args+=(--json)
fi

if [ -n "${JSON_OUT:-}" ]; then
  args+=(--json-out "${JSON_OUT}")
fi

if [ -n "${TURN_SCORE_CSV:-}" ]; then
  args+=(--turn-score-csv "${TURN_SCORE_CSV}")
fi

if [ -n "${SCORE_LOG:-}" ]; then
  args+=(--score-log "${SCORE_LOG}")
fi

exec python "${args[@]}"
