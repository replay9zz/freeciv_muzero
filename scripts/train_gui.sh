#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

SERVER_RC="${SERVER_RC:-${ROOT_DIR}/start_single.serv}"
SERVER_PORT="${SERVER_PORT:-5566}"
LUA_PORT="${LUA_PORT:-4451}"
TRAINING_STEPS="${TRAINING_STEPS:-200}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-5}"
MAX_TURNS="${MAX_TURNS:-30}"
TRAINING_DELAY="${TRAINING_DELAY:-0.0}"
FREECIV_SLEEP="${FREECIV_SLEEP:-0.5}"
FREECIV_REWARD_EXPLORE="${FREECIV_REWARD_EXPLORE:-1.0}"
FREECIV_REWARD_CIV_SCORE="${FREECIV_REWARD_CIV_SCORE:-0.25}"
FREECIV_REWARD_CITY="${FREECIV_REWARD_CITY:-12.0}"
FREECIV_REWARD_POPULATION="${FREECIV_REWARD_POPULATION:-2.0}"
FREECIV_REWARD_SETTLER="${FREECIV_REWARD_SETTLER:-3.0}"
USE_GPU="${USE_GPU:-1}"
MUZERO_MAX_NUM_GPUS="${MUZERO_MAX_NUM_GPUS:-1}"
MUZERO_SELFPLAY_ON_GPU="${MUZERO_SELFPLAY_ON_GPU:-true}"
MUZERO_TRAIN_ON_GPU="${MUZERO_TRAIN_ON_GPU:-true}"
MUZERO_REANALYSE_ON_GPU="${MUZERO_REANALYSE_ON_GPU:-true}"
CLIENT_PATTERN="freeciv-gtk3.22 -a -s 127.0.0.1 -p ${SERVER_PORT} -n agent0 -P none"

BUILD_DIR="${BUILD_DIR:-$(default_build_dir)}"
SCENARIO_PATH="${SCENARIO_PATH:-$(default_scenario_path)}"
SERVER_PATTERN="freeciv-server -p ${SERVER_PORT} -f ${SCENARIO_PATH} -r ${SERVER_RC}"

if [ "${USE_GPU}" = "0" ]; then
  MUZERO_MAX_NUM_GPUS=0
  MUZERO_SELFPLAY_ON_GPU=false
  MUZERO_TRAIN_ON_GPU=false
  MUZERO_REANALYSE_ON_GPU=false
fi

cleanup() {
  cleanup_freeciv_all "${SERVER_PORT}" "${LUA_PORT}"
  pkill -f "${CLIENT_PATTERN}" >/dev/null 2>&1 || true
  pkill -f "${SERVER_PATTERN}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

cleanup

cd "${ROOT_DIR}"
init_python_env

python muzero.py freeciv_remote "{
  \"training_steps\": ${TRAINING_STEPS},
  \"num_workers\": 1,
  \"num_simulations\": ${NUM_SIMULATIONS},
  \"max_turns\": ${MAX_TURNS},
  \"training_delay\": ${TRAINING_DELAY},
  \"max_num_gpus\": ${MUZERO_MAX_NUM_GPUS},
  \"selfplay_on_gpu\": ${MUZERO_SELFPLAY_ON_GPU},
  \"train_on_gpu\": ${MUZERO_TRAIN_ON_GPU},
  \"reanalyse_on_gpu\": ${MUZERO_REANALYSE_ON_GPU},
  \"env\": {
    \"FREECIV_SERVER_PORT\": \"${SERVER_PORT}\",
    \"FREECIV_SERVER_CMD\": \"${BUILD_DIR}/run.sh freeciv-server -p ${SERVER_PORT} -f ${SCENARIO_PATH} -r ${SERVER_RC}\",
    \"FREECIV_HOST\": \"127.0.0.1\",
    \"FREECIV_LUAREMOTE_PORT\": \"${LUA_PORT}\",
    \"FREECIV_CLIENT_CMD\": \"${BUILD_DIR}/run.sh freeciv-gtk3.22 -a -s 127.0.0.1 -p ${SERVER_PORT} -n agent0 -P none\",
    \"FREECIV_PLAYER_ID\": \"0\",
    \"FREECIV_TAKE_PLAYER_ID\": \"0\",
    \"FREECIV_MAP_W\": \"4\",
    \"FREECIV_MAP_H\": \"16\",
    \"FREECIV_MAX_TURNS\": \"${MAX_TURNS}\",
    \"FREECIV_SLEEP\": \"${FREECIV_SLEEP}\",
    \"FREECIV_REWARD_EXPLORE\": \"${FREECIV_REWARD_EXPLORE}\",
    \"FREECIV_REWARD_CIV_SCORE\": \"${FREECIV_REWARD_CIV_SCORE}\",
    \"FREECIV_REWARD_CITY\": \"${FREECIV_REWARD_CITY}\",
    \"FREECIV_REWARD_POPULATION\": \"${FREECIV_REWARD_POPULATION}\",
    \"FREECIV_REWARD_SETTLER\": \"${FREECIV_REWARD_SETTLER}\"
  }
}"
