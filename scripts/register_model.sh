#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/register_model.sh CHECKPOINT NAME [--stage STAGE] [--score SCORE] [--tag TAG] [--dated] [--registry-run RUN_ID]

Creates symlinks under results/model_registry without moving checkpoint files.
With --dated, NAME is stored under RUN_ID/NAME. RUN_ID defaults to
MODEL_REGISTRY_RUN_ID or YYYYmmdd-HHMMSS.
Examples:
  scripts/register_model.sh results/freeciv_remote/.../model.checkpoint latest --stage 3
  scripts/register_model.sh results/freeciv_remote/.../model.checkpoint ladder/step-050000 --dated --registry-run 20260626-scorefixed
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 2 ]; then
  usage
  exit 0
fi

checkpoint="$1"
name="$2"
shift 2

stage=""
score=""
tag=""
dated=0
registry_run_id="${MODEL_REGISTRY_RUN_ID:-}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage)
      stage="${2:-}"
      shift 2
      ;;
    --score)
      score="${2:-}"
      shift 2
      ;;
    --tag)
      tag="${2:-}"
      shift 2
      ;;
    --dated)
      dated=1
      shift
      ;;
    --registry-run)
      registry_run_id="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ ! -f "${checkpoint}" ]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 1
fi

registry="${ROOT_DIR}/results/model_registry"
manifest="${registry}/manifest.jsonl"
registered_name="${name}"
if [ "${dated}" = "1" ]; then
  if [ -z "${registry_run_id}" ]; then
    registry_run_id="$(date +%Y%m%d-%H%M%S)"
  fi
  registered_name="${registry_run_id}/${name}"
fi
mkdir -p "${registry}/$(dirname "${registered_name}")"

abs_checkpoint="$(cd "$(dirname "${checkpoint}")" && pwd)/$(basename "${checkpoint}")"
link="${registry}/${registered_name}.checkpoint"
ln -sfn "${abs_checkpoint}" "${link}"

python_bin="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
if [ ! -x "${python_bin}" ]; then
  python_bin="$(command -v python3)"
fi

"${python_bin}" - "${abs_checkpoint}" "${name}" "${registered_name}" "${stage}" "${score}" "${tag}" "${manifest}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

checkpoint_path, name, registered_name, stage, score, tag, manifest = sys.argv[1:]
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
record = {
    "registered_at": datetime.now(timezone.utc).isoformat(),
    "name": name,
    "registered_name": registered_name,
    "checkpoint_path": checkpoint_path,
    "stage": stage,
    "score": score,
    "tag": tag,
    "training_step": checkpoint.get("training_step"),
    "num_played_games": checkpoint.get("num_played_games"),
    "num_played_steps": checkpoint.get("num_played_steps"),
}
manifest_path = Path(manifest)
manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
print(json.dumps(record, ensure_ascii=False, sort_keys=True))
PY
