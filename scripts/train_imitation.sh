#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

TRAJECTORIES="${TRAJECTORIES:-${ROOT_DIR}/results/imitation/20260707-024315-easy-ai-trajectories/trajectories.jsonl}"
if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then
  TRAJECTORIES="$1"
  shift
elif [ "${1:-}" != "" ]; then
  TRAJECTORIES=""
fi

USE_GPU="${USE_GPU:-1}"
if [ "${USE_GPU}" = "0" ]; then
  export CUDA_VISIBLE_DEVICES=""
  DEVICE="${DEVICE:-cpu}"
else
  init_train_runtime train_imitation
  DEVICE="${DEVICE:-auto}"
fi

cd "${ROOT_DIR}"
init_python_env
export FREECIV_OBSERVE_BELIEF="${FREECIV_OBSERVE_BELIEF:-1}"

exec python -m freeciv_sim.imitation.train_imitation \
  ${TRAJECTORIES:+"${TRAJECTORIES}"} \
  --device "${DEVICE}" \
  "$@"
