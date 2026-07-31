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
if server_rc_has_start "${SERVER_RC}"; then
  TAKE_PLAYER="${TAKE_PLAYER:-}"
  START_AFTER_TAKE="${START_AFTER_TAKE:-0}"
else
  TAKE_PLAYER="${TAKE_PLAYER:-}"
  START_AFTER_TAKE="${START_AFTER_TAKE:-1}"
fi
FREECIV_MUZERO_RULESET_DIR="${FREECIV_MUZERO_RULESET_DIR:-$(server_rc_rulesetdir "${SERVER_RC}")}"
# Shaping overrides disabled by default. Set these env vars explicitly to re-enable.
# FREECIV_CITY_MIN_DISTANCE="${FREECIV_CITY_MIN_DISTANCE:-$(server_rc_set_value "${SERVER_RC}" citymindist)}"
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
FREECIV_MAX_UNITS="${FREECIV_MAX_UNITS:-}"
FREECIV_MAX_CITIES="${FREECIV_MAX_CITIES:-}"
# FREECIV_EXPANSION_MIN_CITIES="${FREECIV_EXPANSION_MIN_CITIES:-6}"
# FREECIV_EXPANSION_TILES_PER_CITY="${FREECIV_EXPANSION_TILES_PER_CITY:-48}"
# FREECIV_SETTLER_FORCE_MOVES_PER_TURN="${FREECIV_SETTLER_FORCE_MOVES_PER_TURN:-2}"
# FREECIV_AUTO_SETTLERS="${FREECIV_AUTO_SETTLERS:-0}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-50}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SLEEP="${SLEEP:-0.1}"
EPISODES="${EPISODES:-1}"
CLIENT_NAME="${CLIENT_NAME:-agent0}"
NO_SEA_UNITS="${NO_SEA_UNITS:-1}"
JSON="${JSON:-0}"
FREECIV_BELIEF_TENSORBOARD="${FREECIV_BELIEF_TENSORBOARD:-1}"
FREECIV_BELIEF_TENSORBOARD_INTERVAL="${FREECIV_BELIEF_TENSORBOARD_INTERVAL:-5}"
PREFER_UNIT_MOVE="${PREFER_UNIT_MOVE:-0}"
DIRECT_UNIT_MOVE_DEMO="${DIRECT_UNIT_MOVE_DEMO:-0}"
MUZERO_MCTS_BACKUP_OPERATOR="${MUZERO_MCTS_BACKUP_OPERATOR:-wasserstein}"
MUZERO_MCTS_WASSERSTEIN_POWER="${MUZERO_MCTS_WASSERSTEIN_POWER:-1.0}"
MUZERO_MCTS_WASSERSTEIN_SELECTION="${MUZERO_MCTS_WASSERSTEIN_SELECTION:-optimistic}"
MUZERO_MCTS_WASSERSTEIN_UNCERTAINTY_COEF="${MUZERO_MCTS_WASSERSTEIN_UNCERTAINTY_COEF:-0.25}"
MUZERO_MCTS_WASSERSTEIN_MIN_STD="${MUZERO_MCTS_WASSERSTEIN_MIN_STD:-1e-6}"
MUZERO_MCTS_WASSERSTEIN_SHIFT_EPSILON="${MUZERO_MCTS_WASSERSTEIN_SHIFT_EPSILON:-1e-6}"
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
export FREECIV_MUZERO_RULESET_DIR
# export FREECIV_CITY_MIN_DISTANCE
# export FREECIV_EXPANSION_MIN_CITIES
# export FREECIV_EXPANSION_TILES_PER_CITY
# export FREECIV_SETTLER_FORCE_MOVES_PER_TURN
# export FREECIV_AUTO_SETTLERS
export FREECIV_BELIEF_TENSORBOARD="${FREECIV_BELIEF_TENSORBOARD}"
export FREECIV_BELIEF_TENSORBOARD_INTERVAL="${FREECIV_BELIEF_TENSORBOARD_INTERVAL}"

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
  --mcts-backup-operator "${MUZERO_MCTS_BACKUP_OPERATOR}"
  --mcts-wasserstein-power "${MUZERO_MCTS_WASSERSTEIN_POWER}"
  --mcts-wasserstein-selection "${MUZERO_MCTS_WASSERSTEIN_SELECTION}"
  --mcts-wasserstein-uncertainty-coef "${MUZERO_MCTS_WASSERSTEIN_UNCERTAINTY_COEF}"
  --mcts-wasserstein-min-std "${MUZERO_MCTS_WASSERSTEIN_MIN_STD}"
  --mcts-wasserstein-shift-epsilon "${MUZERO_MCTS_WASSERSTEIN_SHIFT_EPSILON}"
)

if [ -n "${FREECIV_MAX_UNITS}" ]; then
  args+=(--max-units "${FREECIV_MAX_UNITS}")
fi

if [ -n "${FREECIV_MAX_CITIES}" ]; then
  args+=(--max-cities "${FREECIV_MAX_CITIES}")
fi

if [ -n "${PLAYER_ID}" ]; then
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

if [ -n "${SCORE_LOG_INTERVAL:-}" ]; then
  args+=(--score-log-interval "${SCORE_LOG_INTERVAL}")
fi

if [ -n "${CITY_SCORE_LOG:-}" ]; then
  args+=(--city-score-log "${CITY_SCORE_LOG}")
fi

if [ -n "${CITY_SCORE_LOG_INTERVAL:-}" ]; then
  args+=(--city-score-log-interval "${CITY_SCORE_LOG_INTERVAL}")
fi

FILTER_FREECIV_DISCONNECTS="${FILTER_FREECIV_DISCONNECTS:-1}"
if [ "${FILTER_FREECIV_DISCONNECTS}" = "1" ]; then
  FREECIV_EVAL_LOG_FILTER="${SCRIPT_DIR}/filter_freeciv_eval_log.awk"
  export FREECIV_EVAL_LOG_FILTER
  run_with_timing_and_log eval_headless bash -o pipefail -c '
    python "$@" 2>&1 | awk -f "${FREECIV_EVAL_LOG_FILTER}"
    exit "${PIPESTATUS[0]}"
  ' bash "${args[@]}"
else
  run_with_timing_and_log eval_headless python "${args[@]}"
fi
