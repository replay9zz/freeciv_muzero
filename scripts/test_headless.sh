#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

FREECIV_GENERATED_MAP="${FREECIV_GENERATED_MAP:-1}"
if [ "${FREECIV_GENERATED_MAP}" = "1" ]; then
  SERVER_RC="${SERVER_RC:-${ROOT_DIR}/start_generated_32x32.serv}"
  MAP_WIDTH="${MAP_WIDTH:-${FREECIV_MAP_W:-32}}"
  MAP_HEIGHT="${MAP_HEIGHT:-${FREECIV_MAP_H:-32}}"
else
  SERVER_RC="${SERVER_RC:-${ROOT_DIR}/start_single.serv}"
  MAP_WIDTH="${MAP_WIDTH:-${FREECIV_MAP_W:-16}}"
  MAP_HEIGHT="${MAP_HEIGHT:-${FREECIV_MAP_H:-16}}"
fi
if server_rc_has_start "${SERVER_RC}"; then
  FREECIV_TAKE_PLAYER="${FREECIV_TAKE_PLAYER:-}"
  FREECIV_START_AFTER_TAKE="${FREECIV_START_AFTER_TAKE:-0}"
else
  FREECIV_TAKE_PLAYER="${FREECIV_TAKE_PLAYER:-}"
  FREECIV_START_AFTER_TAKE="${FREECIV_START_AFTER_TAKE:-1}"
fi
FREECIV_MUZERO_RULESET_DIR="${FREECIV_MUZERO_RULESET_DIR:-$(server_rc_rulesetdir "${SERVER_RC}")}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH=model.checkpoint}"
NUM_TESTS="${NUM_TESTS:-1}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-4}"
MAX_TURNS="${MAX_TURNS:-30}"
MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-${FREECIV_MAX_ACTIONS_PER_TURN:-}}"
FREECIV_SLEEP="${FREECIV_SLEEP:-0.2}"
FREECIV_MAX_UNITS="${FREECIV_MAX_UNITS:-24}"
FREECIV_MAX_CITIES="${FREECIV_MAX_CITIES:-16}"
FREECIV_OBSERVE_BELIEF="${FREECIV_OBSERVE_BELIEF:-1}"
FREECIV_ACTION_CURRICULUM_STAGE="${FREECIV_ACTION_CURRICULUM_STAGE:-}"
FREECIV_ACTION_CURRICULUM_GROUPS="${FREECIV_ACTION_CURRICULUM_GROUPS:-}"
FREECIV_ROTATE_PORTS_ON_RESTART="${FREECIV_ROTATE_PORTS_ON_RESTART:-0}"
SERVER_PORT="${SERVER_PORT:-5566}"
LUA_PORT="${LUA_PORT:-4451}"
DISPLAY_NUM="${DISPLAY_NUM:-:101}"
DISPLAY_SIZE="${DISPLAY_SIZE:-1280x800}"
FREECIV_CLIENT_START_WAIT="${FREECIV_CLIENT_START_WAIT:-3}"
FREECIV_CLIENT_START_TIMEOUT="${FREECIV_CLIENT_START_TIMEOUT:-60}"
FREECIV_SERVER_START_TIMEOUT="${FREECIV_SERVER_START_TIMEOUT:-60}"
USE_GPU="${USE_GPU:-1}"
MUZERO_MAX_NUM_GPUS="${MUZERO_MAX_NUM_GPUS:-}"
MUZERO_SELFPLAY_ON_GPU="${MUZERO_SELFPLAY_ON_GPU:-true}"
MUZERO_REANALYSE_ON_GPU="${MUZERO_REANALYSE_ON_GPU:-false}"
MUZERO_CHANNELS="${MUZERO_CHANNELS:-32}"
MUZERO_BLOCKS="${MUZERO_BLOCKS:-2}"
MUZERO_NUM_UNROLL_STEPS="${MUZERO_NUM_UNROLL_STEPS:-5}"
MUZERO_SEED="${MUZERO_SEED:-0}"
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
  FREECIV_SERVER_CMD_TEMPLATE="${BUILD_DIR}/run.sh freeciv-server -p {server_port} -r ${SERVER_RC_TEMPLATE}"
else
  FREECIV_SERVER_CMD_TEMPLATE="${BUILD_DIR}/run.sh freeciv-server -p {server_port} -f ${SCENARIO_PATH} -r ${SERVER_RC_TEMPLATE}"
fi

if [ "${USE_GPU}" = "0" ]; then
  export CUDA_VISIBLE_DEVICES=""
  MUZERO_MAX_NUM_GPUS=0
  MUZERO_SELFPLAY_ON_GPU=false
else
  init_train_runtime test_headless
  MUZERO_MAX_NUM_GPUS="${MUZERO_MAX_NUM_GPUS:-1}"
fi

cleanup() {
  cleanup_freeciv_all "${SERVER_PORT}" "${LUA_PORT}" "${DISPLAY_NUM}"
  pkill -f "${XVFB_PATTERN}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cleanup

Xvfb "${DISPLAY_NUM}" -screen 0 "${DISPLAY_SIZE}x24" -nolisten tcp >/tmp/freeciv-muzero-xvfb.log 2>&1 &
sleep 1
export DISPLAY="${DISPLAY_NUM}"

cd "${ROOT_DIR}"
init_python_env
FREECIV_PROCESS_LOG="${FREECIV_PROCESS_LOG:-${ROOT_DIR}/results/logs/freeciv-process-${FREECIV_SCORE_RUN_ID}.log}"
export FREECIV_PROCESS_LOG

export CHECKPOINT_PATH NUM_TESTS NUM_SIMULATIONS MAX_TURNS MAX_ACTIONS_PER_TURN
export MUZERO_MAX_NUM_GPUS MUZERO_SELFPLAY_ON_GPU MUZERO_REANALYSE_ON_GPU
export MUZERO_CHANNELS MUZERO_BLOCKS MUZERO_NUM_UNROLL_STEPS
export MUZERO_SEED
export MUZERO_MCTS_BACKUP_OPERATOR MUZERO_MCTS_WASSERSTEIN_POWER
export MUZERO_MCTS_WASSERSTEIN_SELECTION MUZERO_MCTS_WASSERSTEIN_UNCERTAINTY_COEF
export MUZERO_MCTS_WASSERSTEIN_MIN_STD MUZERO_MCTS_WASSERSTEIN_SHIFT_EPSILON
export FREECIV_SERVER_PORT="${SERVER_PORT}"
export FREECIV_SERVER_PORT_STRIDE=1
export FREECIV_SERVER_CMD="${FREECIV_SERVER_CMD_TEMPLATE}"
export FREECIV_HOST=127.0.0.1
export FREECIV_LUAREMOTE_PORT="${LUA_PORT}"
export FREECIV_LUAREMOTE_PORT_STRIDE=1
export FREECIV_CLIENT_CMD="${BUILD_DIR}/run.sh freeciv-gtk3.22 -a -s 127.0.0.1 -p {server_port} -n agent0 -P none -- --resolution ${DISPLAY_SIZE}"
export FREECIV_CLIENT_START_WAIT FREECIV_CLIENT_START_TIMEOUT FREECIV_SERVER_START_TIMEOUT
export FREECIV_PLAYER_ID=0
export FREECIV_TAKE_PLAYER_ID=0
export FREECIV_TAKE_PLAYER
export FREECIV_START_AFTER_TAKE FREECIV_MUZERO_RULESET_DIR
export FREECIV_MAP_W="${MAP_WIDTH}"
export FREECIV_MAP_H="${MAP_HEIGHT}"
export FREECIV_MAX_TURNS="${MAX_TURNS}"
export FREECIV_MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN}"
export FREECIV_MAX_UNITS FREECIV_MAX_CITIES FREECIV_SLEEP
export FREECIV_OBSERVE_BELIEF FREECIV_ACTION_CURRICULUM_STAGE FREECIV_ACTION_CURRICULUM_GROUPS
export FREECIV_ROTATE_PORTS_ON_RESTART

LOG_DIR="${ROOT_DIR}/results/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${TEST_LOG_PATH:-${LOG_DIR}/test_headless-${FREECIV_SCORE_RUN_ID}.log}"
mkdir -p "$(dirname "${LOG_PATH}")"

set +e
{
  echo "Run: test_headless"
  echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  echo "Log: ${LOG_PATH}"
  python - <<'PY'
import os
import ray

from muzero import MuZero


def env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


config = {
    "seed": int(os.environ["MUZERO_SEED"]),
    "max_num_gpus": int(os.environ["MUZERO_MAX_NUM_GPUS"]),
    "selfplay_on_gpu": env_bool("MUZERO_SELFPLAY_ON_GPU", True),
    "train_on_gpu": False,
    "reanalyse_on_gpu": env_bool("MUZERO_REANALYSE_ON_GPU", False),
    "num_workers": 1,
    "num_simulations": int(os.environ["NUM_SIMULATIONS"]),
    "max_turns": int(os.environ["MAX_TURNS"]),
    "channels": int(os.environ["MUZERO_CHANNELS"]),
    "blocks": int(os.environ["MUZERO_BLOCKS"]),
    "num_unroll_steps": int(os.environ["MUZERO_NUM_UNROLL_STEPS"]),
    "mcts_backup_operator": os.environ["MUZERO_MCTS_BACKUP_OPERATOR"],
    "mcts_wasserstein_power": float(os.environ["MUZERO_MCTS_WASSERSTEIN_POWER"]),
    "mcts_wasserstein_selection": os.environ["MUZERO_MCTS_WASSERSTEIN_SELECTION"],
    "mcts_wasserstein_uncertainty_coef": float(os.environ["MUZERO_MCTS_WASSERSTEIN_UNCERTAINTY_COEF"]),
    "mcts_wasserstein_min_std": float(os.environ["MUZERO_MCTS_WASSERSTEIN_MIN_STD"]),
    "mcts_wasserstein_shift_epsilon": float(os.environ["MUZERO_MCTS_WASSERSTEIN_SHIFT_EPSILON"]),
}
if os.getenv("MAX_ACTIONS_PER_TURN"):
    config["max_actions_per_turn"] = int(os.environ["MAX_ACTIONS_PER_TURN"])

muzero = MuZero("freeciv_remote", config)
muzero.load_model(checkpoint_path=os.environ["CHECKPOINT_PATH"])
num_gpus = 1 if config["selfplay_on_gpu"] and config["max_num_gpus"] > 0 else 0
result = muzero.test(
    render=False,
    opponent="self",
    muzero_player=0,
    num_tests=int(os.environ["NUM_TESTS"]),
    num_gpus=num_gpus,
)
print(f"[test] average_reward={result}")
ray.shutdown()
PY
  status=$?
  echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Exit status: ${status}"
  exit "${status}"
} 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
