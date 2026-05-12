from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from freeciv_alpha_zero.freeciv import live_agent as alpha_live
from freeciv_alpha_zero.freeciv.config import MapConfig
from freeciv_alpha_zero.freeciv.state import FreecivBoardState

from games.freeciv import MuZeroConfig, _observation_channels
import models
from self_play import MCTS, SelfPlay


def load_weights(checkpoint_path: Path) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        return checkpoint["weights"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format; expected dict with weights.")


def apply_action(
    client: alpha_live.LuaRemoteClient,
    action: int,
    unit_id: int,
    dir_ids: list[int],
    board_state: FreecivBoardState,
    owned_cities: list[Tuple[int, int, int]],
    player_id: Optional[int],
    research_flags: Dict[str, bool],
) -> None:
    if action == board_state.PASS_ACTION:
        return

    if action == board_state.BUILD_CITY_ACTION:
        city_name = f"MuZeroCity{len(owned_cities) + 1}"
        built = client.found_city(unit_id, city_name)
        if not built:
            client.build_city(unit_id)
        return

    if board_state.RESEARCH_ACTION_BASE <= action < board_state.RESEARCH_ACTION_BASE + board_state.RESEARCH_ACTION_COUNT:
        if not owned_cities:
            return
        tech_idx = action - board_state.RESEARCH_ACTION_BASE
        tech_name = board_state.RESEARCH_TECHS[tech_idx]
        alpha_live.set_research_to_target(
            client,
            player_id,
            research_flags=research_flags,
            tech_name=tech_name,
        )
        return

    if 0 <= action < len(dir_ids):
        dir_id = dir_ids[action]
        client.move_dir_id(unit_id, dir_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a MuZero Freeciv model via LuaRemote.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4444)
    ap.add_argument("--timeout", type=float, default=2.5)
    ap.add_argument("--unit-id", type=int, help="Control a single unit id (disables auto-discovery).")
    ap.add_argument("--player-id", type=int, help="Restrict auto-discovery to a specific player id.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--map-width", type=int, default=9)
    ap.add_argument("--map-height", type=int, default=9)
    ap.add_argument("--max-turns", type=int, default=64)
    ap.add_argument("--dir-ids", default="0,1,4,7,6,3")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--num-simulations", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint file {checkpoint_path} not found.")

    dir_ids = alpha_live.parse_dir_ids(args.dir_ids)
    map_cfg = MapConfig(map_w=args.map_width, map_h=args.map_height, max_turns=args.max_turns)

    config = MuZeroConfig()
    config.map_config = map_cfg
    config.observation_shape = (
        _observation_channels(),
        map_cfg.map_h,
        map_cfg.map_w,
    )
    config.max_moves = map_cfg.max_turns
    config.num_simulations = args.num_simulations
    config.selfplay_on_gpu = False

    model = models.MuZeroNetwork(config)
    model.set_weights(load_weights(checkpoint_path))
    model.eval()

    client = alpha_live.LuaRemoteClient(args.host, args.port, timeout=args.timeout)
    client.connect()

    player_id: Optional[int] = args.player_id
    if args.unit_id is not None:
        controlled_units = [args.unit_id]
        pos_result = client.eval(alpha_live.simple_find_unit_pos(args.unit_id))
        pos_info = alpha_live.parse_position_result(pos_result)
        if pos_info and pos_info[2] is not None and pos_info[2] >= 0:
            player_id = int(pos_info[2])
    else:
        controlled_units, player_id = alpha_live.discover_controlled_units(client, player_id)
        if not controlled_units:
            raise SystemExit("No controllable units found. Provide --unit-id or --player-id.")
        print(
            f"Discovered {len(controlled_units)} unit(s) for player {player_id if player_id is not None else 'unknown'}."
        )

    movement = alpha_live.FreecivMovement(map_width=map_cfg.map_w, map_height=map_cfg.map_h)
    known_tiles: Dict[Tuple[int, int], str] = {}
    known_enemy: Dict[Tuple[int, int], bool] = {}
    visited_tiles: Set[Tuple[int, int]] = set()

    mcts = MCTS(config)

    steps = 0
    while steps < args.max_steps and controlled_units:
        unit_id = controlled_units[0]
        try:
            snapshot, player_id = alpha_live.gather_snapshot(
                client=client,
                movement=movement,
                cfg=map_cfg,
                unit_id=unit_id,
                player_id=player_id,
                known_tiles=known_tiles,
                known_enemy=known_enemy,
                visited_tiles=visited_tiles,
            )
        except Exception:
            controlled_units, player_id = alpha_live.discover_controlled_units(client, player_id)
            time.sleep(args.sleep)
            continue

        visited_tiles.add(snapshot.player_pos)
        board_state = alpha_live.build_state(map_cfg, snapshot)
        observation = board_state.encode(1)
        valids = board_state.valid_moves(1)
        legal_actions = [idx for idx, allowed in enumerate(valids) if allowed]
        if not legal_actions:
            client.end_turn()
            steps += 1
            time.sleep(args.sleep)
            continue

        owned_cities = alpha_live.discover_player_cities(client, player_id)
        if not owned_cities and valids[board_state.BUILD_CITY_ACTION]:
            action = board_state.BUILD_CITY_ACTION
        else:
            with torch.no_grad():
                root, _info = mcts.run(
                    model=model,
                    observation=observation,
                    legal_actions=legal_actions,
                    to_play=0,
                    add_exploration_noise=False,
                )
                action = SelfPlay.select_action(root, args.temperature)

        apply_action(
            client=client,
            action=action,
            unit_id=unit_id,
            dir_ids=dir_ids,
            board_state=board_state,
            owned_cities=owned_cities,
            player_id=player_id,
            research_flags=snapshot.research_flags,
        )

        client.end_turn()
        steps += 1
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
