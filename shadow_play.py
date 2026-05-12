from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy
import torch

import models
from self_play import MCTS, SelfPlay

from games import freeciv as sim_game
from freeciv_alpha_zero.freeciv import live_agent as alpha_live
from freeciv_alpha_zero.freeciv.config import MapConfig
from freeciv_alpha_zero.freeciv.multihead_state import (
    MultiheadState,
    PRODUCTION_UNIT_NAMES,
)
from freeciv_alpha_zero.freeciv.providers import RandomMapProvider


def load_weights(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        return checkpoint["weights"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format; expected dict with weights.")


def _collect_live_units(client, player_id):
    controlled, player_id = alpha_live.discover_controlled_units(client, player_id)
    units = []
    if not controlled:
        return units, player_id
    for uid in sorted(controlled):
        pos_result = client.eval(alpha_live.simple_find_unit_pos(uid))
        pos_info = alpha_live.parse_position_result(pos_result)
        if pos_info is None:
            continue
        unit_type = alpha_live.get_unit_rule_name(client, uid) or ""
        units.append((uid, int(pos_info[0]), int(pos_info[1]), unit_type))
    return units, player_id


def _can_build_city(unit_type: str) -> bool:
    label = (unit_type or "").lower()
    return "settler" in label or "migrant" in label


def _action_to_string(state: MultiheadState, action_number: int) -> str:
    if action_number == state.PASS_ACTION:
        return "pass"
    if action_number < state.MOVE_SIZE:
        unit_idx = action_number // state.MOVE_PER_UNIT
        dir_idx = action_number % state.MOVE_PER_UNIT
        return f"move_u{unit_idx}_d{dir_idx}"
    if action_number < state.MOVE_SIZE + state.ATTACK_SIZE:
        rel = action_number - state.MOVE_SIZE
        unit_idx = rel // state.ATTACK_PER_UNIT
        dir_idx = rel % state.ATTACK_PER_UNIT
        return f"attack_u{unit_idx}_d{dir_idx}"
    econ_idx = action_number - (state.MOVE_SIZE + state.ATTACK_SIZE)
    if 0 <= econ_idx < len(state.RESEARCH_TECHS):
        tech = state.RESEARCH_TECHS[econ_idx]
        return f"research_{tech}"
    if state.ECON_BUILD_CITY_OFFSET <= econ_idx < state.ECON_PRODUCTION_OFFSET:
        unit_idx = econ_idx - state.ECON_BUILD_CITY_OFFSET
        return f"build_city_u{unit_idx}"
    if state.ECON_PRODUCTION_OFFSET <= econ_idx < state.ECON_PASS_OFFSET:
        rel = econ_idx - state.ECON_PRODUCTION_OFFSET
        city_slot = rel // state.PRODUCTION_UNIT_COUNT
        unit_idx = rel % state.PRODUCTION_UNIT_COUNT
        unit_name = PRODUCTION_UNIT_NAMES[unit_idx]
        return f"produce_c{city_slot}_{unit_name}"
    return str(action_number)


def _mask_legal_actions(
    state: MultiheadState,
    live_units,
    live_cities,
    research_locked: bool,
    production_locked: set[int],
    acted_unit_slots: set[int],
):
    valid = state.valid_moves(1).astype(bool)

    # Disable move/attack/build for unit slots without a live unit.
    for slot_idx in range(state.max_units):
        has_unit = slot_idx < len(live_units)
        if not has_unit:
            move_start = slot_idx * state.MOVE_PER_UNIT
            move_end = move_start + state.MOVE_PER_UNIT
            atk_start = state.MOVE_SIZE + slot_idx * state.ATTACK_PER_UNIT
            atk_end = atk_start + state.ATTACK_PER_UNIT
            valid[move_start:move_end] = 0
            valid[atk_start:atk_end] = 0
            econ_offset = state.MOVE_SIZE + state.ATTACK_SIZE
            build_idx = econ_offset + state.ECON_BUILD_CITY_OFFSET + slot_idx
            if 0 <= build_idx < len(valid):
                valid[build_idx] = 0
        else:
            unit_type = live_units[slot_idx][3]
            if not _can_build_city(unit_type):
                econ_offset = state.MOVE_SIZE + state.ATTACK_SIZE
                build_idx = econ_offset + state.ECON_BUILD_CITY_OFFSET + slot_idx
                if 0 <= build_idx < len(valid):
                    valid[build_idx] = 0
        if slot_idx in acted_unit_slots:
            move_start = slot_idx * state.MOVE_PER_UNIT
            move_end = move_start + state.MOVE_PER_UNIT
            atk_start = state.MOVE_SIZE + slot_idx * state.ATTACK_PER_UNIT
            atk_end = atk_start + state.ATTACK_PER_UNIT
            valid[move_start:move_end] = 0
            valid[atk_start:atk_end] = 0
            econ_offset = state.MOVE_SIZE + state.ATTACK_SIZE
            build_idx = econ_offset + state.ECON_BUILD_CITY_OFFSET + slot_idx
            if 0 <= build_idx < len(valid):
                valid[build_idx] = 0

    if not live_cities:
        econ_offset = state.MOVE_SIZE + state.ATTACK_SIZE
        research_end = econ_offset + state.ECON_BUILD_CITY_OFFSET
        prod_start = econ_offset + state.ECON_PRODUCTION_OFFSET
        prod_end = econ_offset + state.ECON_PASS_OFFSET
        build_start = econ_offset + state.ECON_BUILD_CITY_OFFSET
        build_end = econ_offset + state.ECON_PRODUCTION_OFFSET
        valid[econ_offset:research_end] = 0
        valid[prod_start:prod_end] = 0
        build_candidates = [
            idx
            for idx in range(build_start, build_end)
            if 0 <= idx < len(valid) and valid[idx]
        ]
        if build_candidates:
            forced = numpy.zeros_like(valid)
            for idx in build_candidates:
                forced[idx] = 1
            valid = forced

    if research_locked:
        econ_offset = state.MOVE_SIZE + state.ATTACK_SIZE
        research_end = econ_offset + state.ECON_BUILD_CITY_OFFSET
        valid[econ_offset:research_end] = 0

    if production_locked and live_cities:
        econ_offset = state.MOVE_SIZE + state.ATTACK_SIZE
        prod_start = econ_offset + state.ECON_PRODUCTION_OFFSET
        for slot_idx, (city_id, _cx, _cy) in enumerate(live_cities):
            if city_id not in production_locked:
                continue
            start = prod_start + slot_idx * state.PRODUCTION_UNIT_COUNT
            end = start + state.PRODUCTION_UNIT_COUNT
            valid[start:end] = 0

    non_pass = valid.copy()
    non_pass[state.PASS_ACTION] = 0
    if non_pass.any() and live_units:
        valid[state.PASS_ACTION] = 0

    return [idx for idx, allowed in enumerate(valid) if allowed]


def _apply_to_live(
    client,
    movement,
    dir_ids,
    state: MultiheadState,
    action: int,
    live_units,
    live_cities,
    player_id,
    production_locked,
):
    if action == state.PASS_ACTION:
        return "pass"
    if action < state.MOVE_SIZE:
        unit_idx = action // state.MOVE_PER_UNIT
        dir_idx = action % state.MOVE_PER_UNIT
        if unit_idx >= len(live_units) or dir_idx >= len(dir_ids):
            return "move_skip"
        uid = live_units[unit_idx][0]
        client.move_dir_id(uid, dir_ids[dir_idx])
        return f"move_u{unit_idx}_d{dir_idx}"
    if action < state.MOVE_SIZE + state.ATTACK_SIZE:
        rel = action - state.MOVE_SIZE
        unit_idx = rel // state.ATTACK_PER_UNIT
        dir_idx = rel % state.ATTACK_PER_UNIT
        if unit_idx >= len(live_units):
            return "attack_skip"
        if dir_idx >= len(dir_ids):
            return "attack_skip"
        uid, ux, uy, _utype = live_units[unit_idx]
        neighbors = movement.get_native_neighbors(ux, uy)
        nx, ny = neighbors[dir_idx]
        if nx is None or ny is None:
            return "attack_skip"
        client.attack_target(uid, int(nx), int(ny))
        return f"attack_u{unit_idx}_d{dir_idx}"
    econ_idx = action - (state.MOVE_SIZE + state.ATTACK_SIZE)
    if 0 <= econ_idx < len(state.RESEARCH_TECHS):
        if not live_cities:
            return "research_skip"
        tech = state.RESEARCH_TECHS[econ_idx]
        alpha_live.set_research_to_target(client, player_id, tech_name=tech)
        return f"research_{tech}"
    if state.ECON_BUILD_CITY_OFFSET <= econ_idx < state.ECON_PRODUCTION_OFFSET:
        unit_idx = econ_idx - state.ECON_BUILD_CITY_OFFSET
        if unit_idx >= len(live_units):
            return "build_skip"
        uid = live_units[unit_idx][0]
        city_name = f"MuZeroCity{len(live_cities) + 1}"
        built = client.found_city(uid, city_name)
        if not built:
            client.build_city(uid)
        return f"build_city_u{unit_idx}"
    if state.ECON_PRODUCTION_OFFSET <= econ_idx < state.ECON_PASS_OFFSET:
        rel = econ_idx - state.ECON_PRODUCTION_OFFSET
        city_slot = rel // state.PRODUCTION_UNIT_COUNT
        unit_idx = rel % state.PRODUCTION_UNIT_COUNT
        if city_slot >= len(live_cities):
            return "produce_skip"
        city_id = live_cities[city_slot][0]
        unit_name = PRODUCTION_UNIT_NAMES[unit_idx]
        if client.set_city_production(city_id, "UnitType", unit_name):
            production_locked.add(city_id)
        return f"produce_c{city_slot}_{unit_name}"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run MuZero on the simulator and mirror actions to Freeciv."
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4444)
    ap.add_argument("--map-width", type=int, default=4)
    ap.add_argument("--map-height", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=128)
    ap.add_argument("--player-id", type=int, default=0)
    ap.add_argument("--dir-ids", default="0,1,4,7,6,3")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--num-simulations", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint file {checkpoint_path} not found.")

    config = sim_game.MuZeroConfig()
    if (
        args.map_width != config.map_config.map_w
        or args.map_height != config.map_config.map_h
        or args.max_turns != config.map_config.max_turns
    ):
        raise SystemExit(
            "Map config mismatch. Use 2x9/128 to match the trained checkpoint."
        )
    config.players = [0]
    config.num_simulations = args.num_simulations

    model = models.MuZeroNetwork(config)
    model.set_weights(load_weights(checkpoint_path))
    model.eval()

    map_cfg = MapConfig(args.map_width, args.map_height, args.max_turns)
    provider = RandomMapProvider(map_cfg.map_w, map_cfg.map_h)
    state = MultiheadState(
        map_cfg,
        provider,
        max_units=config.max_units,
        max_cities=config.max_cities,
    )

    client = alpha_live.LuaRemoteClient(args.host, args.port, timeout=2.5)
    client.connect()
    movement = alpha_live.FreecivMovement(map_cfg.map_w, map_cfg.map_h)
    dir_ids = alpha_live.parse_dir_ids(args.dir_ids)

    production_locked = set()
    research_locked = False
    acted_unit_slots: set[int] = set()
    steps = 0
    while steps < args.max_steps and state.terminal_reason is None:
        live_units, args.player_id = _collect_live_units(client, args.player_id)
        live_cities = alpha_live.discover_player_cities(client, args.player_id)

        current_research = alpha_live.query_player_research(client, args.player_id)
        research_locked = (
            isinstance(current_research, str)
            and current_research.startswith("__TECH__")
        )

        legal_actions = _mask_legal_actions(
            state,
            live_units,
            live_cities,
            research_locked,
            production_locked,
            acted_unit_slots,
        )
        if not legal_actions:
            action = state.PASS_ACTION
        else:
            observation = state.encode(1)
            with torch.no_grad():
                root, _info = MCTS(config).run(
                    model=model,
                    observation=observation,
                    legal_actions=legal_actions,
                    to_play=0,
                    add_exploration_noise=False,
                )
                action = SelfPlay.select_action(root, args.temperature)

        prev_turn = state.turn
        state.step(1, action)
        live_desc = _apply_to_live(
            client,
            movement,
            dir_ids,
            state,
            action,
            live_units,
            live_cities,
            args.player_id,
            production_locked,
        )
        if action < state.MOVE_SIZE:
            acted_unit_slots.add(action // state.MOVE_PER_UNIT)
        elif action < state.MOVE_SIZE + state.ATTACK_SIZE:
            rel = action - state.MOVE_SIZE
            acted_unit_slots.add(rel // state.ATTACK_PER_UNIT)
        else:
            econ_idx = action - (state.MOVE_SIZE + state.ATTACK_SIZE)
            if state.ECON_BUILD_CITY_OFFSET <= econ_idx < state.ECON_PRODUCTION_OFFSET:
                acted_unit_slots.add(econ_idx - state.ECON_BUILD_CITY_OFFSET)
        print(f"sim={_action_to_string(state, action)} live={live_desc}")

        if state.turn != prev_turn:
            acted_unit_slots.clear()
            try:
                client.end_turn()
            except Exception:
                pass
        if args.sleep:
            time.sleep(args.sleep)
        steps += 1

    try:
        client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
