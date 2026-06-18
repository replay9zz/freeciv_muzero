#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/inspect_checkpoint.sh CHECKPOINT [GAME]

Examples:
  scripts/inspect_checkpoint.sh results/freeciv_remote/2026-06-13--07-45-29/model.checkpoint
  scripts/inspect_checkpoint.sh results/freeciv_remote/2026-06-13--07-45-29/model.checkpoint freeciv_remote
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
  usage
  exit 0
fi

CHECKPOINT="$1"
GAME="${2:-freeciv_remote}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

if [ ! -x "${PYTHON}" ]; then
  echo "Python not executable: ${PYTHON}" >&2
  echo "Set PYTHON=/path/to/python or create ${ROOT_DIR}/.venv" >&2
  exit 1
fi

cd "${ROOT_DIR}"
exec "${PYTHON}" - "${CHECKPOINT}" "${GAME}" <<'PY'
import importlib
import os
import pathlib
import sys

import torch


def shape_of(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def first_conv(weights):
    preferred = [
        "representation_network.module.conv.weight",
        "representation_network.conv.weight",
    ]
    for key in preferred:
        value = weights.get(key)
        shape = shape_of(value)
        if shape is not None and len(shape) == 4:
            return key, shape
    for key, value in weights.items():
        shape = shape_of(value)
        if shape is not None and len(shape) == 4 and "representation" in key:
            return key, shape
    for key, value in weights.items():
        shape = shape_of(value)
        if shape is not None and len(shape) == 4:
            return key, shape
    return None, None


def print_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint root is {type(checkpoint)!r}, expected dict")

    print(f"checkpoint: {path}")
    print(f"keys: {', '.join(sorted(checkpoint.keys()))}")
    for key in (
        "training_step",
        "num_played_games",
        "num_played_steps",
        "num_reanalysed_games",
        "episode_length",
        "total_reward",
        "muzero_reward",
        "opponent_reward",
        "mean_value",
        "lr",
        "total_loss",
        "value_loss",
        "reward_loss",
        "policy_loss",
    ):
        if key in checkpoint:
            print(f"{key}: {checkpoint[key]}")

    weights = checkpoint.get("weights") or {}
    print(f"weight_keys: {len(weights)}")
    conv_key, conv_shape = first_conv(weights)
    if conv_shape is not None:
        print(f"first_conv_key: {conv_key}")
        print(f"first_conv_shape: {conv_shape}")
        print(f"input_channels_from_weights: {conv_shape[1]}")
    else:
        print("first_conv_key: n/a")
    return conv_shape


def config_shapes(game_name):
    module = importlib.import_module(f"games.{game_name}")
    shapes = []
    if game_name == "freeciv_remote":
        old = os.environ.get("FREECIV_OBSERVE_BELIEF")
        try:
            for value in ("0", "1"):
                os.environ["FREECIV_OBSERVE_BELIEF"] = value
                config = module.MuZeroConfig()
                shapes.append((f"FREECIV_OBSERVE_BELIEF={value}", config.observation_shape))
        finally:
            if old is None:
                os.environ.pop("FREECIV_OBSERVE_BELIEF", None)
            else:
                os.environ["FREECIV_OBSERVE_BELIEF"] = old
    else:
        config = module.MuZeroConfig()
        shapes.append(("default", config.observation_shape))
    return shapes


def main():
    checkpoint_path = pathlib.Path(sys.argv[1]).expanduser()
    game_name = sys.argv[2]
    conv_shape = print_checkpoint(checkpoint_path)
    print(f"game: {game_name}")
    try:
        shapes = config_shapes(game_name)
    except Exception as exc:
        print(f"config_shape_error: {exc}")
        return 0

    matched = []
    for label, shape in shapes:
        print(f"config_observation_shape[{label}]: {shape}")
        if conv_shape is not None and len(shape) >= 1 and int(shape[0]) == int(conv_shape[1]):
            matched.append(label)
    if matched:
        print(f"input_channel_match: {', '.join(matched)}")
    elif conv_shape is not None:
        print("input_channel_match: none")
    return 0


raise SystemExit(main())
PY
