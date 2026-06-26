#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CURRICULUM_STAGES="${CURRICULUM_STAGES:-0,1,2,3,4,full}"
CURRICULUM_TARGET_STEPS="${CURRICULUM_TARGET_STEPS:-5000,10000,20000,40000,80000,120000}"
MODEL_REGISTRY_RUN_ID="${MODEL_REGISTRY_RUN_ID:-$(date +%Y%m%d-%H%M%S)-curriculum}"
export MODEL_REGISTRY_RUN_ID

IFS=',' read -r -a stages <<<"${CURRICULUM_STAGES}"
IFS=',' read -r -a target_steps <<<"${CURRICULUM_TARGET_STEPS}"

if [ "${#stages[@]}" -ne "${#target_steps[@]}" ]; then
  echo "CURRICULUM_STAGES and CURRICULUM_TARGET_STEPS must have the same length." >&2
  exit 1
fi

latest_file() {
  local name="$1"
  find "${ROOT_DIR}/results/freeciv_remote" -type f -name "${name}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 { $1=""; sub(/^ /, ""); print }'
}

checkpoint_path="${CHECKPOINT_PATH:-}"
replay_buffer_path="${REPLAY_BUFFER_PATH:-}"

for idx in "${!stages[@]}"; do
  stage="${stages[$idx]}"
  target="${target_steps[$idx]}"
  echo "[curriculum] registry_run=${MODEL_REGISTRY_RUN_ID} stage=${stage} target_training_steps=${target}"

  CHECKPOINT_PATH="${checkpoint_path}" \
  REPLAY_BUFFER_PATH="${replay_buffer_path}" \
  TRAINING_STEPS="${target}" \
  FREECIV_ACTION_CURRICULUM_STAGE="${stage}" \
  "${SCRIPT_DIR}/train_headless.sh"

  checkpoint_path="$(latest_file model.checkpoint)"
  replay_buffer_path="$(latest_file replay_buffer.pkl)"
  if [ -z "${checkpoint_path}" ]; then
    echo "No model.checkpoint found after stage ${stage}." >&2
    exit 1
  fi
  "${SCRIPT_DIR}/register_model.sh" "${checkpoint_path}" latest --stage "${stage}" --tag curriculum
  safe_stage="${stage//[^A-Za-z0-9_.-]/_}"
  "${SCRIPT_DIR}/register_model.sh" "${checkpoint_path}" "stage/${safe_stage}" --stage "${stage}" --tag curriculum --dated
  "${SCRIPT_DIR}/register_model.sh" "${checkpoint_path}" "ladder/step-${target}" --stage "${stage}" --tag curriculum --dated
  echo "[curriculum] next_checkpoint=${checkpoint_path}"
  if [ -n "${replay_buffer_path}" ]; then
    echo "[curriculum] next_replay_buffer=${replay_buffer_path}"
  fi
done
