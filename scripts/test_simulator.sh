#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
NUM_TESTS="${NUM_TESTS:-5}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-4}"
MAX_TURNS="${MAX_TURNS:-100}"
MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-8}"
USE_GPU="${USE_GPU:-1}"
MUZERO_MAX_NUM_GPUS="${MUZERO_MAX_NUM_GPUS:-}"
MUZERO_CHANNELS="${MUZERO_CHANNELS:-32}"
MUZERO_BLOCKS="${MUZERO_BLOCKS:-2}"
MUZERO_SEED="${MUZERO_SEED:-0}"
FREECIV_MAP_W="${FREECIV_MAP_W:-32}"
FREECIV_MAP_H="${FREECIV_MAP_H:-32}"
FREECIV_MAX_UNITS="${FREECIV_MAX_UNITS:-24}"
FREECIV_MAX_CITIES="${FREECIV_MAX_CITIES:-16}"
FREECIV_OBSERVE_BELIEF="${FREECIV_OBSERVE_BELIEF:-1}"
FREECIV_SIM_SINGLE_PLAYER="${FREECIV_SIM_SINGLE_PLAYER:-1}"
FREECIV_MUZERO_RULESET_DIR="${FREECIV_MUZERO_RULESET_DIR:-civ2civ3}"

if [ "${USE_GPU}" = "0" ]; then
  export CUDA_VISIBLE_DEVICES=""
  MUZERO_MAX_NUM_GPUS=0
else
  init_train_runtime test_simulator
  MUZERO_MAX_NUM_GPUS="${MUZERO_MAX_NUM_GPUS:-1}"
fi

cd "${ROOT_DIR}"
init_python_env

export CHECKPOINT_PATH NUM_TESTS NUM_SIMULATIONS MAX_TURNS MAX_ACTIONS_PER_TURN
export MUZERO_MAX_NUM_GPUS MUZERO_CHANNELS MUZERO_BLOCKS MUZERO_SEED USE_GPU
export FREECIV_MAP_W FREECIV_MAP_H FREECIV_MAX_UNITS FREECIV_MAX_CITIES
export FREECIV_OBSERVE_BELIEF FREECIV_SIM_SINGLE_PLAYER FREECIV_MUZERO_RULESET_DIR

python - <<'PY'
import os
import ray

from muzero import MuZero


use_gpu = os.environ["USE_GPU"] != "0"
config = {
    "seed": int(os.environ["MUZERO_SEED"]),
    "max_num_gpus": int(os.environ["MUZERO_MAX_NUM_GPUS"]),
    "selfplay_on_gpu": use_gpu,
    "train_on_gpu": False,
    "reanalyse_on_gpu": False,
    "use_last_model_value": False,
    "num_workers": 1,
    "num_simulations": int(os.environ["NUM_SIMULATIONS"]),
    "max_turns": int(os.environ["MAX_TURNS"]),
    "max_actions_per_turn": int(os.environ["MAX_ACTIONS_PER_TURN"]),
    "channels": int(os.environ["MUZERO_CHANNELS"]),
    "blocks": int(os.environ["MUZERO_BLOCKS"]),
}
muzero = MuZero("freeciv", config)
checkpoint = os.environ.get("CHECKPOINT_PATH")
if checkpoint:
    muzero.load_model(checkpoint_path=checkpoint)
result = muzero.test(
    render=False,
    opponent="self",
    muzero_player=0,
    num_tests=int(os.environ["NUM_TESTS"]),
    num_gpus=1 if use_gpu else 0,
)
print(f"[sim-test] average_reward={result}")
ray.shutdown()
PY
