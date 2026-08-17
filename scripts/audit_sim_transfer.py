#!/usr/bin/env python3
"""Verify simulator/remote network compatibility and checkpoint lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

import torch

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import models
from games.freeciv import Game as SimulatorGame
from games.freeciv import MuZeroConfig as SimulatorConfig
from games.freeciv_remote import MuZeroConfig as RemoteConfig


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: pathlib.Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def strict_model_load(config, checkpoint: dict) -> int:
    model = models.MuZeroNetwork(config)
    model.set_weights(checkpoint["weights"])
    return sum(parameter.numel() for parameter in model.parameters())


def weight_delta(before: dict, after: dict) -> dict:
    before_weights = before["weights"]
    after_weights = after["weights"]
    if set(before_weights) != set(after_weights):
        missing = sorted(set(before_weights) - set(after_weights))
        extra = sorted(set(after_weights) - set(before_weights))
        raise RuntimeError(f"Checkpoint weight keys differ: missing={missing} extra={extra}")

    squared_delta = 0.0
    squared_before = 0.0
    changed_tensors = 0
    total_tensors = 0
    for key in sorted(before_weights):
        left = before_weights[key].detach().cpu()
        right = after_weights[key].detach().cpu()
        if left.shape != right.shape:
            raise RuntimeError(
                f"Checkpoint tensor shape differs for {key}: {left.shape} != {right.shape}"
            )
        total_tensors += 1
        if not torch.equal(left, right):
            changed_tensors += 1
        if left.is_floating_point() or left.is_complex():
            delta = right.to(torch.float64) - left.to(torch.float64)
            squared_delta += float(torch.sum(delta * delta).item())
            base = left.to(torch.float64)
            squared_before += float(torch.sum(base * base).item())
    l2 = math.sqrt(squared_delta)
    base_l2 = math.sqrt(squared_before)
    return {
        "changed_tensors": changed_tensors,
        "total_tensors": total_tensors,
        "weight_delta_l2": l2,
        "weight_delta_relative": l2 / base_l2 if base_l2 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True, type=pathlib.Path)
    parser.add_argument("--phase2", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    simulator_config = SimulatorConfig()
    remote_config = RemoteConfig()
    signature = {
        "observation_shape": list(simulator_config.observation_shape),
        "action_space_size": len(simulator_config.action_space),
        "channels": simulator_config.channels,
        "blocks": simulator_config.blocks,
        "support_size": simulator_config.support_size,
    }
    remote_signature = {
        "observation_shape": list(remote_config.observation_shape),
        "action_space_size": len(remote_config.action_space),
        "channels": remote_config.channels,
        "blocks": remote_config.blocks,
        "support_size": remote_config.support_size,
    }
    if signature != remote_signature:
        raise RuntimeError(
            f"Simulator/remote network signatures differ: {signature} != {remote_signature}"
        )

    phase1_path = args.phase1.resolve()
    phase1 = load_checkpoint(phase1_path)
    simulator_parameters = strict_model_load(simulator_config, phase1)
    remote_parameters = strict_model_load(remote_config, phase1)
    simulator_game = SimulatorGame(seed=simulator_config.seed, config=simulator_config)
    observation = simulator_game.reset()
    legal_actions = simulator_game.legal_actions()

    report = {
        "compatible": True,
        "network_signature": signature,
        "phase1": {
            "checkpoint": str(phase1_path),
            "sha256": file_sha256(phase1_path),
            "training_step": int(phase1.get("training_step", 0)),
            "optimizer_state_present": phase1.get("optimizer_state") is not None,
            "strict_load_simulator": True,
            "strict_load_remote": True,
            "simulator_parameters": simulator_parameters,
            "remote_parameters": remote_parameters,
            "simulator_reset_shape": list(observation.shape),
            "simulator_initial_legal_actions": len(legal_actions),
        },
    }

    if args.phase2:
        phase2_path = args.phase2.resolve()
        phase2 = load_checkpoint(phase2_path)
        strict_model_load(remote_config, phase2)
        expected_parent_hash = report["phase1"]["sha256"]
        actual_parent_hash = phase2.get("parent_checkpoint_sha256")
        parent_matches = actual_parent_hash == expected_parent_hash
        if not parent_matches:
            raise RuntimeError(
                "Phase 2 checkpoint lineage does not match Phase 1: "
                f"{actual_parent_hash} != {expected_parent_hash}"
            )
        report["phase2"] = {
            "checkpoint": str(phase2_path),
            "sha256": file_sha256(phase2_path),
            "training_step": int(phase2.get("training_step", 0)),
            "parent_checkpoint": phase2.get("parent_checkpoint"),
            "parent_checkpoint_sha256": actual_parent_hash,
            "parent_training_step": int(phase2.get("parent_training_step", -1)),
            "parent_matches_phase1": parent_matches,
            "strict_load_remote": True,
            **weight_delta(phase1, phase2),
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
