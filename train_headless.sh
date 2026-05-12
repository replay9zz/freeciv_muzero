#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/freeciv_build_v3_2}"
SCENARIO_PATH="${SCENARIO_PATH:-${HOME}/.freeciv/scenarios/minimal_v4.sav}"
SERVER_RC="${SERVER_RC:-${REPO_ROOT}/freeciv_rl/start_single.serv}"
SERVER_PORT="5566"
LUA_PORT="4451"
DISPLAY_NUM="${DISPLAY_NUM:-:101}"
TRAINING_STEPS="${TRAINING_STEPS:-5000}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-2}"
MAX_TURNS="${MAX_TURNS:-300}"
TRAINING_DELAY="${TRAINING_DELAY:-0.1}"
FREECIV_SLEEP="${FREECIV_SLEEP:-0.2}"
FREECIV_REWARD_EXPLORE="${FREECIV_REWARD_EXPLORE:-0.5}"
FREECIV_REWARD_CIV_SCORE="${FREECIV_REWARD_CIV_SCORE:-0.5}"
FREECIV_REWARD_CITY="${FREECIV_REWARD_CITY:-30.0}"
FREECIV_REWARD_POPULATION="${FREECIV_REWARD_POPULATION:-3.0}"
FREECIV_REWARD_SETTLER="${FREECIV_REWARD_SETTLER:-10.0}"
XVFB_PATTERN="Xvfb ${DISPLAY_NUM}"
CLIENT_PATTERN="freeciv-gtk3.22 -a -s 127.0.0.1 -p 5566 -n agent0 -P none"
SERVER_PATTERN="freeciv-server -p 5566 -f ${SCENARIO_PATH} -r ${SERVER_RC}"

cleanup() {
  fuser -k -TERM "${SERVER_PORT}/tcp" >/dev/null 2>&1 || true
  fuser -k -TERM "${LUA_PORT}/tcp" >/dev/null 2>&1 || true
  pkill -f "${CLIENT_PATTERN}" >/dev/null 2>&1 || true
  pkill -f "${SERVER_PATTERN}" >/dev/null 2>&1 || true
  pkill -f "${XVFB_PATTERN}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

cleanup

Xvfb "${DISPLAY_NUM}" -screen 0 1280x800x24 -nolisten tcp >/tmp/freeciv-muzero-xvfb.log 2>&1 &
sleep 1
export DISPLAY="${DISPLAY_NUM}"

cd "${ROOT_DIR}"
source .venv/bin/activate

python muzero.py freeciv_remote "{
  \"training_steps\": ${TRAINING_STEPS},
  \"num_workers\": 1,
  \"num_simulations\": ${NUM_SIMULATIONS},
  \"max_turns\": ${MAX_TURNS},
  \"training_delay\": ${TRAINING_DELAY},
  \"max_num_gpus\": 0,
  \"selfplay_on_gpu\": false,
  \"train_on_gpu\": false,
  \"reanalyse_on_gpu\": false,
  \"env\": {
    \"FREECIV_SERVER_PORT\": \"5566\",
    \"FREECIV_SERVER_CMD\": \"${BUILD_DIR}/run.sh freeciv-server -p 5566 -f ${SCENARIO_PATH} -r ${SERVER_RC}\",
    \"FREECIV_HOST\": \"127.0.0.1\",
    \"FREECIV_LUAREMOTE_PORT\": \"4451\",
    \"FREECIV_CLIENT_CMD\": \"${BUILD_DIR}/run.sh freeciv-gtk3.22 -a -s 127.0.0.1 -p 5566 -n agent0 -P none\",
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
