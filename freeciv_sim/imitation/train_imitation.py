from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import models
from freeciv_sim.state.config import MapConfig
from freeciv_sim.state.multihead_state import (
    City,
    MHUnit,
    MultiheadState,
    PRODUCTION_ITEM_INDEX,
    UNIT_SPECS,
)
from freeciv_sim.state.providers import GroundTruth, RandomMapProvider
from games.freeciv_remote import BELIEF_OBSERVATION_PLANES, MuZeroConfig


@dataclass(frozen=True)
class Sample:
    observation: np.ndarray
    action: int
    event_type: str


def _player_key(pid: int) -> str:
    return str(int(pid))


def _sorted_owned(items: Iterable[dict[str, Any]], owner: int) -> list[dict[str, Any]]:
    return sorted(
        [item for item in items if int(item.get("owner", -9999)) == int(owner)],
        key=lambda item: int(item.get("id", 0)),
    )


def _unit_name(raw: str | None) -> str:
    key = (raw or "").strip()
    lower = key.lower()
    names = {name.lower(): name for name in UNIT_SPECS}
    if lower in names:
        return names[lower]
    if lower.endswith("s") and lower[:-1] in names:
        return names[lower[:-1]]
    if "settler" in lower or "migrant" in lower:
        return "Settlers"
    if "worker" in lower:
        return "Workers"
    return key if key in UNIT_SPECS else "Warriors"


def _unit_from_json(unit: dict[str, Any]) -> MHUnit:
    name = _unit_name(unit.get("type"))
    spec = UNIT_SPECS.get(name) or UNIT_SPECS["Warriors"]
    hp = int(unit.get("hp") or 0)
    moves = int(unit.get("moves_left") or 0)
    return MHUnit(
        x=int(unit.get("x", 0)),
        y=int(unit.get("y", 0)),
        hp=hp if hp > 0 else int(spec.hp),
        atk=int(spec.atk),
        df=int(spec.df),
        firepower=int(spec.firepower),
        unit_type=spec.name,
        alive=True,
        can_build_city=bool(spec.can_build_city or "settler" in name.lower()),
        home_city=None,
        moves_left=moves if moves > 0 else max(1, int(spec.moves)),
    )


def _city_from_json(city: dict[str, Any]) -> City:
    production = city.get("production") or {}
    kind = (production.get("kind") or "").strip().lower() or None
    name = (production.get("name") or "").strip() or None
    if kind not in ("unit", "building"):
        kind = None
        name = None
    return City(
        x=int(city.get("x", 0)),
        y=int(city.get("y", 0)),
        size=max(1, int(city.get("size") or 1)),
        production_kind=kind,
        production_target=name,
    )


def _make_state(config: MuZeroConfig, snapshot: dict[str, Any], player_id: int) -> MultiheadState:
    cfg: MapConfig = config.map_config
    state = MultiheadState(
        cfg,
        RandomMapProvider(cfg.map_w, cfg.map_h, p_open=1.0),
        max_units=config.max_units,
        max_cities=config.max_cities,
    )
    au_map = np.full((cfg.map_h, cfg.map_w), "A", dtype="<U1")
    enemy_map = np.zeros((cfg.map_h, cfg.map_w), dtype=bool)
    state.gt = GroundTruth(au_map, enemy_map)
    state.units = {1: [], -1: []}
    state.cities = {1: [], -1: []}

    units = snapshot.get("units") or []
    cities = snapshot.get("cities") or []
    for raw in _sorted_owned(units, player_id)[: config.max_units]:
        state.units[1].append(_unit_from_json(raw))
    for raw in sorted(
        [unit for unit in units if int(unit.get("owner", -9999)) != int(player_id)],
        key=lambda item: int(item.get("id", 0)),
    )[: config.max_units]:
        state.units[-1].append(_unit_from_json(raw))
        x, y = int(raw.get("x", 0)), int(raw.get("y", 0))
        if 0 <= y < cfg.map_h and 0 <= x < cfg.map_w:
            enemy_map[y, x] = True

    while len(state.units[1]) < config.max_units:
        state.units[1].append(MHUnit(0, 0, 0, 0, 0, 1, "None", False, False, None, 0))
    while len(state.units[-1]) < config.max_units:
        state.units[-1].append(MHUnit(0, 0, 0, 0, 0, 1, "None", False, False, None, 0))

    for raw in _sorted_owned(cities, player_id)[: config.max_cities]:
        state.cities[1].append(_city_from_json(raw))
    for raw in sorted(
        [city for city in cities if int(city.get("owner", -9999)) != int(player_id)],
        key=lambda item: int(item.get("id", 0)),
    )[: config.max_cities]:
        state.cities[-1].append(_city_from_json(raw))
        x, y = int(raw.get("x", 0)), int(raw.get("y", 0))
        if 0 <= y < cfg.map_h and 0 <= x < cfg.map_w:
            enemy_map[y, x] = True

    state.visited = {
        1: np.ones((cfg.map_h, cfg.map_w), dtype=bool),
        -1: np.ones((cfg.map_h, cfg.map_w), dtype=bool),
    }
    state.turn = int(snapshot.get("turn") or 0)
    state.num_actions = int(snapshot.get("turn") or 0) * max(1, state.max_actions_per_turn)
    state.actions_this_turn = 0
    state.acted_unit_slots = {1: set(), -1: set()}
    state.acted_production_cities = {1: set(), -1: set()}
    state.research_done = {
        1: {tech: False for tech in state.RESEARCH_TECHS},
        -1: {tech: False for tech in state.RESEARCH_TECHS},
    }
    research = snapshot.get("research") or {}
    own_target = research.get(_player_key(player_id))
    if own_target in state.RESEARCH_TECHS:
        state.research_target[1] = own_target
    return state


def _observation(config: MuZeroConfig, snapshot: dict[str, Any], player_id: int) -> np.ndarray:
    state = _make_state(config, snapshot, player_id)
    obs = state.encode(1).astype(np.float32)
    if getattr(config, "observe_belief", False):
        zeros = np.zeros((len(BELIEF_OBSERVATION_PLANES), config.map_config.map_h, config.map_config.map_w), dtype=np.float32)
        obs = np.concatenate((obs, zeros), axis=0)
    return obs


def _unit_slot(snapshot: dict[str, Any], player_id: int, unit_id: int, max_units: int) -> int | None:
    for idx, unit in enumerate(_sorted_owned(snapshot.get("units") or [], player_id)[:max_units]):
        if int(unit.get("id", -1)) == int(unit_id):
            return idx
    return None


def _city_slot(snapshot: dict[str, Any], player_id: int, city_id: int, max_cities: int) -> int | None:
    for idx, city in enumerate(_sorted_owned(snapshot.get("cities") or [], player_id)[:max_cities]):
        if int(city.get("id", -1)) == int(city_id):
            return idx
    return None


def _direction(state: MultiheadState, src: list[int], dst: list[int]) -> int | None:
    sx, sy = int(src[0]), int(src[1])
    tx, ty = int(dst[0]), int(dst[1])
    for idx, coord in enumerate(state.movement.get_native_neighbors(sx, sy)):
        if coord == (tx, ty):
            return idx
    return None


def _infer_action(config: MuZeroConfig, prev: dict[str, Any], event: dict[str, Any]) -> tuple[int, int] | None:
    event_type = event.get("type")
    move_size = config.max_units * MultiheadState.MOVE_PER_UNIT
    attack_size = config.max_units * MultiheadState.ATTACK_PER_UNIT
    activity_size = config.max_units * MultiheadState.UNIT_ACTIVITY_PER_UNIT
    econ_offset = move_size + attack_size + activity_size

    if event_type == "unit_moved":
        player_id = int(event.get("owner"))
        slot = _unit_slot(prev, player_id, int(event.get("unit_id")), config.max_units)
        if slot is None:
            return None
        state = _make_state(config, prev, player_id)
        dir_idx = _direction(state, event.get("from") or [], event.get("to") or [])
        if dir_idx is None:
            return None
        return player_id, slot * MultiheadState.MOVE_PER_UNIT + dir_idx

    if event_type == "city_created" and event.get("inferred_action") == "build_city":
        city = event.get("city") or {}
        source = event.get("source_unit") or {}
        player_id = int(city.get("owner", source.get("owner")))
        slot = _unit_slot(prev, player_id, int(source.get("id")), config.max_units)
        if slot is None:
            return None
        return player_id, econ_offset + len(MultiheadState.RESEARCH_TECHS) + slot

    if event_type == "research_changed":
        player_id = int(event.get("player"))
        tech = event.get("to")
        if tech not in MultiheadState.RESEARCH_TECHS:
            return None
        return player_id, econ_offset + MultiheadState.RESEARCH_TECHS.index(tech)

    if event_type == "city_production_changed":
        player_id = int(event.get("owner"))
        city_id = int(event.get("city_id"))
        city_slot = _city_slot(prev, player_id, city_id, config.max_cities)
        if city_slot is None:
            return None
        production = event.get("to") or {}
        kind = (production.get("kind") or "").strip().lower()
        name = (production.get("name") or "").strip()
        item_idx = PRODUCTION_ITEM_INDEX.get((kind, name))
        if item_idx is None:
            return None
        action = (
            econ_offset
            + len(MultiheadState.RESEARCH_TECHS)
            + config.max_units
            + city_slot * len(MultiheadState.PRODUCTION_ITEM_NAMES)
            + item_idx
        )
        return player_id, action

    return None


def load_samples(config: MuZeroConfig, paths: list[pathlib.Path], validate_legal: bool) -> tuple[list[Sample], dict[str, int]]:
    samples: list[Sample] = []
    stats: dict[str, int] = {"records": 0, "events": 0, "labeled": 0, "illegal": 0}
    prev_snapshot: dict[str, Any] | None = None

    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                record = json.loads(line)
                stats["records"] += 1
                snapshot = record.get("snapshot") or {}
                events = record.get("events") or []
                if prev_snapshot is not None:
                    for event in events:
                        stats["events"] += 1
                        labeled = _infer_action(config, prev_snapshot, event)
                        if labeled is None:
                            continue
                        player_id, action = labeled
                        if not (0 <= action < len(config.action_space)):
                            continue
                        state = _make_state(config, prev_snapshot, player_id)
                        if validate_legal and not state.valid_moves(1)[action]:
                            stats["illegal"] += 1
                            continue
                        obs = state.encode(1).astype(np.float32)
                        if getattr(config, "observe_belief", False):
                            zeros = np.zeros(
                                (
                                    len(BELIEF_OBSERVATION_PLANES),
                                    config.map_config.map_h,
                                    config.map_config.map_w,
                                ),
                                dtype=np.float32,
                            )
                            obs = np.concatenate((obs, zeros), axis=0)
                        samples.append(Sample(obs, int(action), str(event.get("type"))))
                        stats["labeled"] += 1
                prev_snapshot = snapshot

    return samples, stats


def load_indexed_samples(
    config: MuZeroConfig,
    snapshots_path: pathlib.Path,
    samples_path: pathlib.Path,
    validate_legal: bool,
) -> tuple[list[Sample], dict[str, int]]:
    snapshot_rows: list[dict[str, Any]] = []
    with snapshots_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                snapshot_rows.append(json.loads(line))
    samples: list[Sample] = []
    stats = {"records": 0, "labeled": 0, "illegal": 0, "missing_snapshot": 0}
    with samples_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            record = json.loads(line)
            stats["records"] += 1
            ref = int(record.get("snapshot_ref", -1))
            if ref < 0 or ref >= len(snapshot_rows):
                stats["missing_snapshot"] += 1
                continue
            action = int(record["action_index"])
            player_id = int(record["player"])
            state = _make_state(config, snapshot_rows[ref].get("snapshot") or {}, player_id)
            if validate_legal and not state.valid_moves(1)[action]:
                stats["illegal"] += 1
                continue
            obs = state.encode(1).astype(np.float32)
            if getattr(config, "observe_belief", False):
                zeros = np.zeros(
                    (
                        len(BELIEF_OBSERVATION_PLANES),
                        config.map_config.map_h,
                        config.map_config.map_w,
                    ),
                    dtype=np.float32,
                )
                obs = np.concatenate((obs, zeros), axis=0)
            samples.append(Sample(obs, action, str(record.get("event") or "")))
            stats["labeled"] += 1
    return samples, stats


def _checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, training_step: int) -> dict[str, Any]:
    return {
        "weights": models.dict_to_cpu(model.state_dict()),
        "optimizer_state": models.dict_to_cpu(optimizer.state_dict()),
        "total_reward": 0,
        "muzero_reward": 0,
        "opponent_reward": 0,
        "episode_length": 0,
        "mean_value": 0,
        "training_step": int(training_step),
        "lr": optimizer.param_groups[0]["lr"],
        "total_loss": 0,
        "value_loss": 0,
        "reward_loss": 0,
        "policy_loss": 0,
        "num_played_games": 0,
        "num_played_steps": 0,
        "num_reanalysed_games": 0,
        "terminate": False,
    }


def main() -> None:
    os.environ.setdefault("FREECIV_OBSERVE_BELIEF", "1")
    parser = argparse.ArgumentParser(description="Pretrain Freeciv MuZero policy from built-in AI trajectory JSONL.")
    parser.add_argument("trajectories", nargs="*", type=pathlib.Path)
    parser.add_argument("--samples", type=pathlib.Path, help="imitation_samples.jsonl built from ACTLOG.")
    parser.add_argument("--snapshots", type=pathlib.Path, help="snapshots.jsonl used by --samples.")
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--checkpoint-path", type=pathlib.Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-legal", action="store_true")
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = MuZeroConfig()
    if args.channels is not None:
        config.channels = args.channels
    if args.blocks is not None:
        config.blocks = args.blocks

    if args.samples is not None:
        if args.snapshots is None:
            raise SystemExit("--snapshots is required with --samples")
        samples, stats = load_indexed_samples(
            config,
            args.snapshots,
            args.samples,
            args.validate_legal,
        )
    else:
        if not args.trajectories:
            raise SystemExit("Provide trajectory JSONL files, or use --samples with --snapshots")
        samples, stats = load_samples(config, args.trajectories, args.validate_legal)
    if not samples:
        raise SystemExit(f"No imitation samples generated: {stats}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = models.MuZeroNetwork(config).to(device)
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        model.set_weights(checkpoint["weights"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.weight_decay)
    model.train()

    actions = np.array([sample.action for sample in samples], dtype=np.int64)
    event_counts: dict[str, int] = {}
    for sample in samples:
        event_counts[sample.event_type] = event_counts.get(sample.event_type, 0) + 1

    for step in range(1, args.steps + 1):
        batch_idx = np.random.randint(0, len(samples), size=args.batch_size)
        obs = torch.tensor(
            np.stack([samples[int(idx)].observation for idx in batch_idx]),
            dtype=torch.float32,
            device=device,
        )
        target = torch.tensor(actions[batch_idx], dtype=torch.long, device=device)
        _value, _reward, policy_logits, _hidden = model.initial_inference(obs)
        loss = torch.nn.functional.cross_entropy(policy_logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == args.steps or step % max(1, args.steps // 10) == 0:
            pred = policy_logits.argmax(dim=1)
            acc = (pred == target).float().mean().item()
            print(f"[imitation] step={step}/{args.steps} loss={loss.item():.4f} acc={acc:.3f}", flush=True)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT_DIR / "results" / "imitation_pretrain" / _dt.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint(model, optimizer, 0), output_dir / "model.checkpoint")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "samples": len(samples),
                "stats": stats,
                "event_counts": event_counts,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": str(device),
                "observation_shape": list(config.observation_shape),
                "action_space": len(config.action_space),
            },
            fp,
            indent=2,
            sort_keys=True,
        )
        fp.write("\n")
    print(f"[imitation] saved={output_dir / 'model.checkpoint'} samples={len(samples)} stats={stats}", flush=True)


if __name__ == "__main__":
    main()
