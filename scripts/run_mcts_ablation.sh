#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
trap 'exit 130' INT TERM

RUN_ID="${ABLATION_RUN_ID:-$(date +%Y%m%d-%H%M%S)-mcts-ablation}"
OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-${ROOT_DIR}/results/mcts_ablation/${RUN_ID}}"
SEEDS="${ABLATION_SEEDS:-1}"
TRAINING_STEPS="${TRAINING_STEPS:-10000}"
NUM_TESTS="${NUM_TESTS:-10}"
SUMMARY_PATH="${OUTPUT_ROOT}/summary.tsv"
GOOGLE_DRIVE_RESULTS="${GOOGLE_DRIVE_RESULTS:-}"
GOOGLE_DRIVE_RESULTS_INCLUDE_TENSORBOARD="${GOOGLE_DRIVE_RESULTS_INCLUDE_TENSORBOARD:-1}"
GOOGLE_DRIVE_RESULTS_BACKGROUND="${GOOGLE_DRIVE_RESULTS_BACKGROUND:-0}"
export GOOGLE_DRIVE_RESULTS GOOGLE_DRIVE_RESULTS_INCLUDE_TENSORBOARD
export GOOGLE_DRIVE_RESULTS_BACKGROUND

conditions=(baseline wasserstein stochastic wasserstein_stochastic)
mkdir -p "${OUTPUT_ROOT}"
printf 'condition\tseed\twasserstein\tstochastic\ttrain_status\ttest_status\taverage_reward\tcheckpoint\n' >"${SUMMARY_PATH}"
echo "[ablation] note: current stochastic search uses one fixed chance outcome (0)."
echo "[ablation] drive=${GOOGLE_DRIVE_RESULTS:-disabled} tensorboard=${GOOGLE_DRIVE_RESULTS_INCLUDE_TENSORBOARD}"

IFS=',' read -r -a seed_values <<<"${SEEDS}"
overall_status=0

for seed in "${seed_values[@]}"; do
  if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: ${seed}" >&2
    exit 2
  fi

  for condition in "${conditions[@]}"; do
    wasserstein=0
    stochastic=0
    backup_operator=mean
    case "${condition}" in
      wasserstein)
        wasserstein=1
        backup_operator=wasserstein
        ;;
      stochastic)
        stochastic=1
        ;;
      wasserstein_stochastic)
        wasserstein=1
        stochastic=1
        backup_operator=wasserstein
        ;;
    esac

    case_dir="${OUTPUT_ROOT}/${condition}/seed-${seed}"
    train_dir="${case_dir}/train"
    test_log="${case_dir}/test.log"
    mkdir -p "${train_dir}"
    run_key="${RUN_ID}-${condition}-seed${seed}"

    {
      echo "condition=${condition}"
      echo "seed=${seed}"
      echo "training_steps=${TRAINING_STEPS}"
      echo "num_tests=${NUM_TESTS}"
      echo "mcts_backup_operator=${backup_operator}"
      echo "muzero_stochastic=${stochastic}"
      echo "google_drive_results=${GOOGLE_DRIVE_RESULTS}"
      echo "google_drive_results_include_tensorboard=${GOOGLE_DRIVE_RESULTS_INCLUDE_TENSORBOARD}"
      echo "git_rev=$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    } >"${case_dir}/run_info.txt"

    echo "[ablation] train condition=${condition} seed=${seed}"
    MUZERO_RESULTS_PATH="${train_dir}" \
    MUZERO_SEED="${seed}" \
    FREECIV_SEED="${seed}" \
    MUZERO_STOCHASTIC="${stochastic}" \
    MUZERO_MCTS_BACKUP_OPERATOR="${backup_operator}" \
    FREECIV_SCORE_RUN_ID="${run_key}-train" \
    TRAINING_STEPS="${TRAINING_STEPS}" \
      "${SCRIPT_DIR}/train_headless.sh"
    train_status=$?

    checkpoint="${train_dir}/model.checkpoint"
    test_status=125
    average_reward=""
    if [ "${train_status}" -eq 0 ] && [ -f "${checkpoint}" ]; then
      echo "[ablation] test condition=${condition} seed=${seed}"
      CHECKPOINT_PATH="${checkpoint}" \
      TEST_LOG_PATH="${test_log}" \
      MUZERO_SEED="${seed}" \
      FREECIV_SEED="${seed}" \
      MUZERO_STOCHASTIC="${stochastic}" \
      MUZERO_MCTS_BACKUP_OPERATOR="${backup_operator}" \
      FREECIV_SCORE_RUN_ID="${run_key}-test" \
      NUM_TESTS="${NUM_TESTS}" \
        "${SCRIPT_DIR}/test_headless.sh"
      test_status=$?
      average_reward="$(sed -n 's/^\[test\] average_reward=//p' "${test_log}" | tail -n 1)"
    else
      echo "[ablation] skip test: training failed or checkpoint missing" >&2
    fi

    if [ "${train_status}" -ne 0 ] || [ "${test_status}" -ne 0 ]; then
      overall_status=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${condition}" "${seed}" "${wasserstein}" "${stochastic}" \
      "${train_status}" "${test_status}" "${average_reward}" "${checkpoint}" \
      >>"${SUMMARY_PATH}"
  done
done

"${ROOT_DIR}/.venv/bin/python" - "${OUTPUT_ROOT}" <<'PY'
import pathlib
import sys

import drive_sync

drive_sync.sync_path(pathlib.Path(sys.argv[1]), force=True)
PY
echo "[ablation] summary=${SUMMARY_PATH}"
exit "${overall_status}"
