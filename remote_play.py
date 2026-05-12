from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from games import freeciv_remote
import models
from self_play import MCTS, SelfPlay


def _set_env(name: str, value) -> None:
    if value is None:
        return
    os.environ[name] = str(value)


def load_weights(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        return checkpoint["weights"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format; expected dict with weights.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a MuZero Freeciv model against a LuaRemote Freeciv client."
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4444)
    ap.add_argument("--map-width", type=int, default=4)
    ap.add_argument("--map-height", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=128)
    ap.add_argument("--player-id", type=int)
    ap.add_argument("--unit-id", type=int)
    ap.add_argument("--dir-ids", default="0,1,4,7,6,3")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--num-simulations", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint file {checkpoint_path} not found.")

    _set_env("FREECIV_HOST", args.host)
    _set_env("FREECIV_PORT", args.port)
    _set_env("FREECIV_MAP_W", args.map_width)
    _set_env("FREECIV_MAP_H", args.map_height)
    _set_env("FREECIV_MAX_TURNS", args.max_turns)
    _set_env("FREECIV_PLAYER_ID", args.player_id)
    _set_env("FREECIV_UNIT_ID", args.unit_id)
    _set_env("FREECIV_DIR_IDS", args.dir_ids)
    _set_env("FREECIV_SLEEP", args.sleep)

    config = freeciv_remote.MuZeroConfig()
    config.num_simulations = args.num_simulations

    model = models.MuZeroNetwork(config)
    model.set_weights(load_weights(checkpoint_path))
    model.eval()

    game = freeciv_remote.Game()
    mcts = MCTS(config)

    observation = game.reset()
    done = False
    steps = 0
    while not done and steps < args.max_steps:
        legal_actions = game.legal_actions()
        if not legal_actions:
            break
        with torch.no_grad():
            root, _info = mcts.run(
                model=model,
                observation=observation,
                legal_actions=legal_actions,
                to_play=game.to_play(),
                add_exploration_noise=False,
            )
            action = SelfPlay.select_action(root, args.temperature)
        turn = getattr(game, "turns", None)
        prefix = f"[step {steps}]" if turn is None else f"[turn {turn} step {steps}]"
        enemy_units = getattr(game, "visible_enemy_units", None)
        enemy_cities = getattr(game, "visible_enemy_cities", None)
        extra = ""
        if isinstance(enemy_units, list):
            extra += f" enemy_units={len(enemy_units)}"
        if isinstance(enemy_cities, list):
            extra += f" enemy_cities={len(enemy_cities)}"
        print(f"{prefix} action={game.action_to_string(action)}{extra}")
        observation, _reward, done = game.step(action)
        if args.render:
            game.render()
        steps += 1

    game.close()


if __name__ == "__main__":
    main()
