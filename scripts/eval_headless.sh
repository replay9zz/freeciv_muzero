#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

FREECIV_GENERATED_MAP="${FREECIV_GENERATED_MAP:-0}"
if [ "${FREECIV_GENERATED_MAP}" = "1" ]; then
  SERVER_RC="${SERVER_RC:-${ROOT_DIR}/start_generated_16x16.serv}"
  MAP_WIDTH="${MAP_WIDTH:-16}"
  MAP_HEIGHT="${MAP_HEIGHT:-16}"
else
  SERVER_RC="${SERVER_RC:-${ROOT_DIR}/start_single.serv}"
  MAP_WIDTH="${MAP_WIDTH:-4}"
  MAP_HEIGHT="${MAP_HEIGHT:-16}"
fi
if server_rc_has_start "${SERVER_RC}"; then
  TAKE_PLAYER="${TAKE_PLAYER:-}"
  START_AFTER_TAKE="${START_AFTER_TAKE:-0}"
else
  TAKE_PLAYER="${TAKE_PLAYER:--}"
  START_AFTER_TAKE="${START_AFTER_TAKE:-1}"
fi
START_COMMAND="${START_COMMAND:-}"
HOST="${HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-5566}"
LUA_PORT="${LUA_PORT:-4451}"
DISPLAY_NUM="${DISPLAY_NUM:-:102}"
DISPLAY_SIZE="${DISPLAY_SIZE:-1280x800}"
DISPLAY_DEPTH="${DISPLAY_DEPTH:-24}"
CLIENT_RESOLUTION="${CLIENT_RESOLUTION:-${DISPLAY_SIZE}}"
PLAYER_ID="${PLAYER_ID:-0}"
TAKE_PLAYER_ID="${TAKE_PLAYER_ID:-${PLAYER_ID}}"
MAX_TURNS="${MAX_TURNS:-300}"
MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-100}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-50}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SLEEP="${SLEEP:-0.1}"
EPISODES="${EPISODES:-1}"
CLIENT_NAME="${CLIENT_NAME:-agent0}"
NO_SEA_UNITS="${NO_SEA_UNITS:-1}"
JSON="${JSON:-0}"
FREECIV_BELIEF_TENSORBOARD="${FREECIV_BELIEF_TENSORBOARD:-1}"
FREECIV_BELIEF_TENSORBOARD_INTERVAL="${FREECIV_BELIEF_TENSORBOARD_INTERVAL:-5}"
FREECIV_REWARD_TENSORBOARD="${FREECIV_REWARD_TENSORBOARD:-1}"
PREFER_UNIT_MOVE="${PREFER_UNIT_MOVE:-0}"
DIRECT_UNIT_MOVE_DEMO="${DIRECT_UNIT_MOVE_DEMO:-0}"
XVFB_PATTERN="Xvfb ${DISPLAY_NUM}"

BUILD_DIR="${BUILD_DIR:-$(default_build_dir)}"
SCENARIO_PATH="${SCENARIO_PATH:-$(default_scenario_path)}"
FREECIV_SCORE_RUN_ID="${FREECIV_SCORE_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
SERVER_RC_DIR="${SERVER_RC_DIR:-/tmp/freeciv-muzero-rc-${FREECIV_SCORE_RUN_ID}}"
SERVER_RC_TEMPLATE="$(prepare_server_rc_template \
  "${SERVER_RC}" \
  "${SERVER_RC_DIR}" \
  "${SERVER_PORT}" \
  "1" \
  "1" \
  "${FREECIV_SCORE_RUN_ID}")"
if [ "${FREECIV_GENERATED_MAP}" = "1" ]; then
  FREECIV_SERVER_CMD_VALUE="${BUILD_DIR}/run.sh freeciv-server -p ${SERVER_PORT} -r ${SERVER_RC_TEMPLATE}"
else
  FREECIV_SERVER_CMD_VALUE="${BUILD_DIR}/run.sh freeciv-server -p ${SERVER_PORT} -f ${SCENARIO_PATH} -r ${SERVER_RC_TEMPLATE}"
fi
CHECKPOINT="${1:-${CHECKPOINT:-$(latest_checkpoint)}}"

if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT=/path/to/model.checkpoint or pass it as the first argument." >&2
  exit 1
fi

cleanup() {
  cleanup_freeciv_all "${SERVER_PORT}" "${LUA_PORT}" "${DISPLAY_NUM}"
  pkill -f "${XVFB_PATTERN}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

cleanup

Xvfb "${DISPLAY_NUM}" -screen 0 "${DISPLAY_SIZE}x${DISPLAY_DEPTH}" -nolisten tcp >/tmp/freeciv-muzero-eval-xvfb.log 2>&1 &
sleep 1
export DISPLAY="${DISPLAY_NUM}"

cd "${ROOT_DIR}"
init_python_env

export FREECIV_BUILD_DIR="${BUILD_DIR}"
export FREECIV_SERVER_PORT="${SERVER_PORT}"
export FREECIV_SERVER_CMD="${FREECIV_SERVER_CMD_VALUE}"
export FREECIV_CLIENT_CMD="${BUILD_DIR}/run.sh freeciv-gtk3.22 -a -s ${HOST} -p ${SERVER_PORT} -n ${CLIENT_NAME} -P none -- --resolution ${CLIENT_RESOLUTION}"
export FREECIV_TAKE_PLAYER_ID="${TAKE_PLAYER_ID}"
export FREECIV_BELIEF_TENSORBOARD="${FREECIV_BELIEF_TENSORBOARD}"
export FREECIV_BELIEF_TENSORBOARD_INTERVAL="${FREECIV_BELIEF_TENSORBOARD_INTERVAL}"
export FREECIV_REWARD_TENSORBOARD="${FREECIV_REWARD_TENSORBOARD}"

args=(
  remote_play.py
  --checkpoint "${CHECKPOINT}"
  --host "${HOST}"
  --port "${LUA_PORT}"
  --map-width "${MAP_WIDTH}"
  --map-height "${MAP_HEIGHT}"
  --max-turns "${MAX_TURNS}"
  --max-actions-per-turn "${MAX_ACTIONS_PER_TURN}"
  --num-simulations "${NUM_SIMULATIONS}"
  --temperature "${TEMPERATURE}"
  --sleep "${SLEEP}"
  --episodes "${EPISODES}"
)

if [ "${TAKE_PLAYER}" != "-" ]; then
  args+=(--player-id "${PLAYER_ID}")
fi

if [ "${NO_SEA_UNITS}" = "1" ]; then
  args+=(--no-sea-units)
fi

if [ -n "${TAKE_PLAYER}" ]; then
  args+=(--take-player "${TAKE_PLAYER}")
fi

if [ "${START_AFTER_TAKE}" = "1" ]; then
  args+=(--start-after-take)
fi

if [ -n "${START_COMMAND}" ]; then
  args+=(--start-command "${START_COMMAND}")
fi

if [ "${FREECIV_BELIEF_TENSORBOARD}" = "1" ]; then
  args+=(
    --belief-tensorboard
    --belief-tensorboard-interval "${FREECIV_BELIEF_TENSORBOARD_INTERVAL}"
  )
fi

if [ "${PREFER_UNIT_MOVE}" = "1" ]; then
  args+=(--prefer-unit-move)
fi

if [ "${DIRECT_UNIT_MOVE_DEMO}" = "1" ]; then
  args+=(--direct-unit-move-demo)
fi

if [ -n "${FREECIV_BELIEF_TENSORBOARD_DIR:-}" ]; then
  args+=(--belief-tensorboard-dir "${FREECIV_BELIEF_TENSORBOARD_DIR}")
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
