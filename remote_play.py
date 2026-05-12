from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import torch

from games import freeciv_remote
import models
from self_play import MCTS, SelfPlay


def _set_env(name: str, value) -> None:
    if value is None:
        return
    os.environ[name] = str(value)


def load_weights(checkpoint_path: Path, map_location) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        return checkpoint["weights"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format; expected dict with weights.")


def _parse_score_line(text: str):
    marker = "__SCORE__"
    if marker not in text and "**SCORE**" not in text:
        return None, None
    normalized = text.replace("**SCORE**", marker)
    payload = normalized.split(marker, 1)[-1].strip()
    if not payload:
        return None, None
    parts = payload.split()
    score = None
    winner = None
    if parts:
        if parts[0].lower() not in ("nil", "none"):
            try:
                score = int(float(parts[0]))
            except ValueError:
                score = None
    if len(parts) > 1:
        val = parts[1].strip().lower()
        if val in ("true", "false"):
            winner = val == "true"
        elif val not in ("nil", "none"):
            winner = None
    return score, winner


def _query_score(client, player_id):
    if client is None or player_id is None:
        return None, None
    lua = (
        "return (function() "
        f"local pl = find.player and find.player({int(player_id)}); "
        "if not pl then return '__SCORE__ nil nil' end; "
        "local score=nil; "
        "if pl.score_game then score = pl:score_game() end; "
        "local win=nil; "
        "if pl.is_winner then win = pl:is_winner() end; "
        "return string.format('__SCORE__ %s %s', tostring(score), tostring(win)) "
        "end)()"
    )
    try:
        res = client.eval(lua)
    except Exception:
        return None, None
    candidates = []
    try:
        last = res.last_return()
        if last:
            candidates.append(last)
    except Exception:
        pass
    for seq_name in ("returns", "lines"):
        try:
            for item in getattr(res, seq_name, []):
                candidates.append(item)
        except Exception:
            pass
    for item in candidates:
        if isinstance(item, str):
            score, winner = _parse_score_line(item)
            if score is not None or winner is not None:
                return score, winner
    return None, None


def _current_civ_score(game):
    state = getattr(game, "_last_state", None)
    if state is None:
        return None
    try:
        return state.civilization_score(1)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a MuZero Freeciv model against a LuaRemote Freeciv client."
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference (default: cuda if available).",
    )
    ap.add_argument("--port", type=int, default=4444)
    ap.add_argument("--map-width", type=int, default=4)
    ap.add_argument("--map-height", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=2000)
    ap.add_argument("--player-id", type=int)
    ap.add_argument("--unit-id", type=int)
    ap.add_argument("--dir-ids", default="0,1,4,7,6,3")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--max-moves", type=int, default=2000)
    ap.add_argument("--num-simulations", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit JSONL results (one line per episode plus a summary).",
    )
    ap.add_argument(
        "--json-out",
        help=(
            "Write JSONL to this file. "
            "If omitted, defaults to results/remote_play/YYYY-MM-DD--HH-MM-SS/remote_play.jsonl."
        ),
    )
    ap.add_argument(
        "--client-cmd",
        help=(
            "Optional command to launch a Freeciv client (headless or GUI). "
            "When set, FREECIV_CLIENT_CMD is exported before connecting."
        ),
    )
    ap.add_argument(
        "--no-client",
        action="store_true",
        help="Do not auto-start a client even if FREECIV_CLIENT_CMD is set.",
    )
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint file {checkpoint_path} not found.")

    if args.no_client:
        os.environ.pop("FREECIV_CLIENT_CMD", None)
        os.environ.pop("FREECIV_CLIENT_RESTART", None)
    elif args.client_cmd:
        _set_env("FREECIV_CLIENT_CMD", args.client_cmd)

    json_path = None
    json_fp = None
    json_records = None
    if args.json or args.json_out:
        args.json = True
        if args.json_out:
            json_path = Path(args.json_out).expanduser()
        else:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
            json_path = (
                Path(__file__).resolve().parent
                / "results"
                / "remote_play"
                / stamp
                / "remote_play.jsonl"
            )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_fp = json_path.open("w", encoding="utf-8")
        json_records = []
        print(f"JSONL output: {json_path}", file=sys.stderr)

    _set_env("FREECIV_HOST", args.host)
    _set_env("FREECIV_PORT", args.port)
    _set_env("FREECIV_LUAREMOTE_PORT", args.port)
    _set_env("FREECIV_MAP_W", args.map_width)
    _set_env("FREECIV_MAP_H", args.map_height)
    _set_env("FREECIV_MAX_TURNS", args.max_turns)
    _set_env("FREECIV_PLAYER_ID", args.player_id)
    _set_env("FREECIV_UNIT_ID", args.unit_id)
    _set_env("FREECIV_DIR_IDS", args.dir_ids)
    _set_env("FREECIV_SLEEP", args.sleep)

    config = freeciv_remote.MuZeroConfig()
    config.num_simulations = args.num_simulations

    device = torch.device(args.device)
    model = models.MuZeroNetwork(config)
    model.set_weights(load_weights(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    game = freeciv_remote.Game()
    mcts = MCTS(config)

    if args.episodes > 1 and not game.restart_on_reset:
        print(
            "Warning: FREECIV_CLIENT_RESTART=1 is not set; "
            "episodes will continue in the same client session.",
            file=sys.stderr,
        )

    emit_human = args.episodes > 1
    log_fp = sys.stderr if args.json else sys.stdout
    win_count = 0
    score_values = []
    for episode in range(args.episodes):
        observation = game.reset()
        done = False
        steps = 0
        start = time.monotonic()
        while not done and steps < args.max_moves:
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
            observation, _reward, done = game.step(action)
            turn = getattr(game, "turns", None)
            prefix = f"[step {steps}]" if turn is None else f"[turn {turn} step {steps}]"
            enemy_units = getattr(game, "visible_enemy_units", None)
            enemy_cities = getattr(game, "visible_enemy_cities", None)
            extra = ""
            if isinstance(enemy_units, list):
                extra += f" enemy_units={len(enemy_units)}"
            if isinstance(enemy_cities, list):
                extra += f" enemy_cities={len(enemy_cities)}"
            civ_score = _current_civ_score(game)
            fc_score, fc_winner = _query_score(game.client, getattr(game, "player_id", None))
            score_parts = []
            if civ_score is not None:
                score_parts.append(f"civ_score={civ_score:.2f}")
            if fc_score is not None:
                score_parts.append(f"fc_score={fc_score}")
            if fc_winner is not None:
                score_parts.append(f"fc_win={fc_winner}")
            line = f"{prefix} action={game.action_to_string(action)}{extra}"
            if score_parts:
                line += " " + " ".join(score_parts)
            print(line, file=log_fp)
            if args.render:
                game.render()
            steps += 1

        score, winner = _query_score(game.client, getattr(game, "player_id", None))
        if winner is True:
            win_count += 1
        if score is not None:
            score_values.append(score)
        elapsed = time.monotonic() - start
        turns = getattr(game, "turns", None)
        record = {
            "episode": episode + 1,
            "steps": steps,
            "turns": turns,
            "score": score,
            "winner": winner,
            "elapsed_sec": round(elapsed, 3),
        }
        if args.json:
            if json_records is not None:
                json_records.append(json.dumps(record, ensure_ascii=True))
        elif emit_human:
            print(record)

    score_mean = None
    if score_values:
        score_mean = sum(score_values) / len(score_values)
    win_rate = win_count / args.episodes if args.episodes else None
    summary = {
        "episodes": args.episodes,
        "wins": win_count,
        "win_rate": win_rate,
        "score_mean": score_mean,
        "score_min": min(score_values) if score_values else None,
        "score_max": max(score_values) if score_values else None,
    }
    if args.json:
        summary_line = json.dumps({"summary": summary}, ensure_ascii=True)
        if json_fp is not None:
            json_fp.write(summary_line + "\n")
            if json_records:
                for line in json_records:
                    json_fp.write(line + "\n")
            json_fp.flush()
        else:
            print(summary_line)
            if json_records:
                for line in json_records:
                    print(line)
    elif emit_human:
        print({"summary": summary})

    if json_fp is not None:
        json_fp.close()

    game.close()


if __name__ == "__main__":
    main()
