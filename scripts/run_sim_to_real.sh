#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${ROOT_DIR}/results/sim_to_real/${RUN_STAMP}}"
PHASE1_DIR="${EXPERIMENT_DIR}/phase1-simulator"
PHASE2_DIR="${EXPERIMENT_DIR}/phase2-real"

PHASE1_STEPS="${PHASE1_STEPS:-1000}"
PHASE2_STEPS="${PHASE2_STEPS:-200}"
PHASE1_WORKERS="${PHASE1_WORKERS:-4}"
PHASE2_WORKERS="${PHASE2_WORKERS:-1}"
PHASE1_GPUS="${PHASE1_GPUS:-1}"
PHASE2_GPUS="${PHASE2_GPUS:-1}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-4}"
MAX_TURNS="${MAX_TURNS:-100}"
MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-8}"
SIM_EVAL_GAMES="${SIM_EVAL_GAMES:-10}"
REAL_EVAL_GAMES="${REAL_EVAL_GAMES:-3}"
MUZERO_BATCH_SIZE="${MUZERO_BATCH_SIZE:-16}"
MUZERO_NUM_UNROLL_STEPS="${MUZERO_NUM_UNROLL_STEPS:-5}"
MUZERO_CHECKPOINT_INTERVAL="${MUZERO_CHECKPOINT_INTERVAL:-10}"
MUZERO_CHANNELS="${MUZERO_CHANNELS:-32}"
MUZERO_BLOCKS="${MUZERO_BLOCKS:-2}"
MUZERO_SEED="${MUZERO_SEED:-0}"
FREECIV_SEED="${FREECIV_SEED:-4242}"
FREECIV_MAP_W="${FREECIV_MAP_W:-32}"
FREECIV_MAP_H="${FREECIV_MAP_H:-32}"
FREECIV_MAX_UNITS="${FREECIV_MAX_UNITS:-24}"
FREECIV_MAX_CITIES="${FREECIV_MAX_CITIES:-16}"
FREECIV_AIFILL="${FREECIV_AIFILL:-3}"
FREECIV_OBSERVE_BELIEF="${FREECIV_OBSERVE_BELIEF:-1}"
FREECIV_MUZERO_RULESET_DIR="${FREECIV_MUZERO_RULESET_DIR:-civ2civ3}"

mkdir -p "${PHASE1_DIR}" "${PHASE2_DIR}"

common_env=(
  FREECIV_MUZERO_RULESET_DIR="${FREECIV_MUZERO_RULESET_DIR}"
  FREECIV_MAP_W="${FREECIV_MAP_W}"
  FREECIV_MAP_H="${FREECIV_MAP_H}"
  FREECIV_MAX_UNITS="${FREECIV_MAX_UNITS}"
  FREECIV_MAX_CITIES="${FREECIV_MAX_CITIES}"
  FREECIV_AIFILL="${FREECIV_AIFILL}"
  FREECIV_OBSERVE_BELIEF="${FREECIV_OBSERVE_BELIEF}"
  MUZERO_CHANNELS="${MUZERO_CHANNELS}"
  MUZERO_BLOCKS="${MUZERO_BLOCKS}"
  MUZERO_SEED="${MUZERO_SEED}"
  NUM_SIMULATIONS="${NUM_SIMULATIONS}"
  MAX_TURNS="${MAX_TURNS}"
  MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN}"
  USE_GPU=1
)

echo "[sim-to-real] experiment=${EXPERIMENT_DIR}"
echo "[sim-to-real] simulator random baseline"
env "${common_env[@]}" NUM_TESTS="${SIM_EVAL_GAMES}" \
  "${SCRIPT_DIR}/test_simulator.sh" 2>&1 | tee "${EXPERIMENT_DIR}/sim-baseline.log"

echo "[sim-to-real] phase 1 simulator training"
env "${common_env[@]}" \
  MUZERO_RESULTS_PATH="${PHASE1_DIR}" \
  MUZERO_MAX_NUM_GPUS="${PHASE1_GPUS}" \
  NUM_WORKERS="${PHASE1_WORKERS}" \
  TRAINING_STEPS="${PHASE1_STEPS}" \
  MUZERO_BATCH_SIZE="${MUZERO_BATCH_SIZE}" \
  MUZERO_NUM_UNROLL_STEPS="${MUZERO_NUM_UNROLL_STEPS}" \
  MUZERO_CHECKPOINT_INTERVAL="${MUZERO_CHECKPOINT_INTERVAL}" \
  RUN_LOG="${PHASE1_DIR}/train.log" \
  "${SCRIPT_DIR}/train_simulator.sh"

PHASE1_CHECKPOINT="${PHASE1_DIR}/model.checkpoint"
test -f "${PHASE1_CHECKPOINT}"

echo "[sim-to-real] phase 1 simulator evaluation"
env "${common_env[@]}" CHECKPOINT_PATH="${PHASE1_CHECKPOINT}" \
  NUM_TESTS="${SIM_EVAL_GAMES}" \
  "${SCRIPT_DIR}/test_simulator.sh" 2>&1 | tee "${EXPERIMENT_DIR}/sim-phase1.log"

env "${common_env[@]}" FREECIV_SIM_SINGLE_PLAYER=1 \
  "${ROOT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/audit_sim_transfer.py" \
  --phase1 "${PHASE1_CHECKPOINT}" --output "${EXPERIMENT_DIR}/phase1-audit.json"

echo "[sim-to-real] phase 1 zero-shot real evaluation"
env "${common_env[@]}" FREECIV_SEED="${FREECIV_SEED}" \
  CHECKPOINT_PATH="${PHASE1_CHECKPOINT}" NUM_TESTS="${REAL_EVAL_GAMES}" \
  FREECIV_SLEEP=0 TEST_LOG_PATH="${EXPERIMENT_DIR}/real-phase1.log" \
  "${SCRIPT_DIR}/test_headless.sh"

PHASE2_TARGET_STEPS=$((PHASE1_STEPS + PHASE2_STEPS))
echo "[sim-to-real] phase 2 real training target_step=${PHASE2_TARGET_STEPS}"
env "${common_env[@]}" \
  MUZERO_RESULTS_PATH="${PHASE2_DIR}" \
  MUZERO_MAX_NUM_GPUS="${PHASE2_GPUS}" \
  CHECKPOINT_PATH="${PHASE1_CHECKPOINT}" REPLAY_BUFFER_PATH= \
  NUM_WORKERS="${PHASE2_WORKERS}" TRAINING_STEPS="${PHASE2_TARGET_STEPS}" \
  TRAINING_DELAY=0 FREECIV_SLEEP=0 \
  MUZERO_BATCH_SIZE="${MUZERO_BATCH_SIZE}" \
  MUZERO_NUM_UNROLL_STEPS="${MUZERO_NUM_UNROLL_STEPS}" \
  MUZERO_CHECKPOINT_INTERVAL="${MUZERO_CHECKPOINT_INTERVAL}" \
  RUN_LOG="${PHASE2_DIR}/train.log" \
  "${SCRIPT_DIR}/train_headless.sh"

PHASE2_CHECKPOINT="${PHASE2_DIR}/model.checkpoint"
test -f "${PHASE2_CHECKPOINT}"

echo "[sim-to-real] phase 2 real evaluation"
env "${common_env[@]}" FREECIV_SEED="${FREECIV_SEED}" \
  CHECKPOINT_PATH="${PHASE2_CHECKPOINT}" NUM_TESTS="${REAL_EVAL_GAMES}" \
  FREECIV_SLEEP=0 TEST_LOG_PATH="${EXPERIMENT_DIR}/real-phase2.log" \
  "${SCRIPT_DIR}/test_headless.sh"

env "${common_env[@]}" FREECIV_SIM_SINGLE_PLAYER=1 \
  "${ROOT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/audit_sim_transfer.py" \
  --phase1 "${PHASE1_CHECKPOINT}" --phase2 "${PHASE2_CHECKPOINT}" \
  --output "${EXPERIMENT_DIR}/transfer-audit.json"

sim_baseline="$(awk -F= '/\[sim-test\] average_reward=/{v=$2} END{print v}' "${EXPERIMENT_DIR}/sim-baseline.log")"
sim_phase1="$(awk -F= '/\[sim-test\] average_reward=/{v=$2} END{print v}' "${EXPERIMENT_DIR}/sim-phase1.log")"
real_phase1="$(awk -F= '/\[test\] average_reward=/{v=$2} END{print v}' "${EXPERIMENT_DIR}/real-phase1.log")"
real_phase2="$(awk -F= '/\[test\] average_reward=/{v=$2} END{print v}' "${EXPERIMENT_DIR}/real-phase2.log")"

"${ROOT_DIR}/.venv/bin/python" - "${EXPERIMENT_DIR}/summary.json" \
  "${sim_baseline}" "${sim_phase1}" "${real_phase1}" "${real_phase2}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
sim_baseline, sim_phase1, real_phase1, real_phase2 = map(float, sys.argv[2:])
summary = {
    "simulator_random_average_reward": sim_baseline,
    "simulator_phase1_average_reward": sim_phase1,
    "simulator_phase1_gain": sim_phase1 - sim_baseline,
    "real_phase1_zero_shot_average_reward": real_phase1,
    "real_phase2_average_reward": real_phase2,
    "real_phase2_gain": real_phase2 - real_phase1,
}
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[sim-to-real] complete summary=${EXPERIMENT_DIR}/summary.json"
