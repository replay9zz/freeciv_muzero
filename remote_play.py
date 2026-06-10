from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

import numpy
import torch

from games import freeciv_remote
import models
from self_play import MCTS, SelfPlay

REPO_ROOT = Path(__file__).resolve().parent
try:
    from freeciv_sim.remote.lua_queries import list_city_scores, list_player_scores
except Exception:  # pragma: no cover - optional dependency
    list_city_scores = None
    list_player_scores = None


def _set_env(name: str, value) -> None:
    if value is None:
        return
    os.environ[name] = str(value)


class _FormatContext(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _format_checkpoint_path(checkpoint_path: Path) -> str:
    try:
        if not checkpoint_path.is_absolute():
            return str(checkpoint_path)
        resolved = checkpoint_path.resolve()
        if REPO_ROOT == resolved or REPO_ROOT in resolved.parents:
            rel = resolved.relative_to(REPO_ROOT)
            if str(rel) == "results" or str(rel).startswith(f"results{os.sep}"):
                return str(rel)
        home = Path.home().resolve()
        if resolved == home or home in resolved.parents:
            return f"~/{resolved.relative_to(home)}"
        return str(resolved)
    except OSError:
        return str(checkpoint_path)


def _write_checkpoint_file(output_dir: Path, checkpoint_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    display_path = _format_checkpoint_path(checkpoint_path)
    (output_dir / "CHECKPOINT").write_text(f"{display_path}\n", encoding="utf-8")


def _resolve_server_scorefile_path() -> Path | None:
    explicit = os.environ.get("FREECIV_SERVER_SCOREFILE")
    if explicit:
        return Path(explicit).expanduser()

    server_cmd = os.environ.get("FREECIV_SERVER_CMD")
    if not server_cmd:
        return None
    context = _FormatContext(os.environ)
    context.setdefault("server_port", os.environ.get("FREECIV_SERVER_PORT", ""))
    context.setdefault("luaremote_port", os.environ.get("FREECIV_LUAREMOTE_PORT", ""))
    context.setdefault("host", os.environ.get("HOST", "127.0.0.1"))
    context.setdefault("server_host", os.environ.get("HOST", "127.0.0.1"))
    try:
        formatted = server_cmd.format_map(context)
        parts = shlex.split(formatted)
    except Exception:
        return None

    rc_path = None
    for idx, part in enumerate(parts[:-1]):
        if part == "-r":
            rc_path = Path(parts[idx + 1]).expanduser()
            break
    if rc_path is None or not rc_path.exists():
        return None

    scorefile = None
    try:
        for line in rc_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r'\s*set\s+scorefile\s+(?:"([^"]+)"|(\S+))', line)
            if match:
                scorefile = match.group(1) or match.group(2)
    except OSError:
        return None
    if not scorefile:
        return None

    path = Path(scorefile).expanduser()
    if path.is_absolute():
        return path
    server_cwd = REPO_ROOT
    if parts:
        cmd0 = Path(parts[0]).expanduser()
        if cmd0.is_absolute() and cmd0.exists():
            server_cwd = cmd0.parent
    return server_cwd / path


def _read_server_scorefile_scores(
    scorefile_path: Path | None = None,
) -> dict[int, tuple[float | None, None, str]]:
    path = scorefile_path or _resolve_server_scorefile_path()
    if path is None or not path.exists():
        return {}
    tags: dict[int, str] = {}
    names: dict[int, str] = {}
    best_turn: dict[int, int] = {}
    scores: dict[int, tuple[float | None, None, str]] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in lines:
        if line.startswith("tag "):
            parts = line.split(maxsplit=2)
            if len(parts) == 3:
                try:
                    tags[int(parts[1])] = parts[2]
                except ValueError:
                    pass
        elif line.startswith("addplayer "):
            parts = line.split(maxsplit=3)
            if len(parts) >= 4:
                try:
                    names[int(parts[2])] = parts[3]
                except ValueError:
                    pass
        elif line.startswith("data "):
            parts = line.split(maxsplit=4)
            if len(parts) != 5:
                continue
            try:
                turn = int(parts[1])
                tag = int(parts[2])
                pid = int(parts[3])
                value = float(parts[4])
            except ValueError:
                continue
            if tags.get(tag) != "score":
                continue
            if turn >= best_turn.get(pid, -1):
                best_turn[pid] = turn
                scores[pid] = (value, None, names.get(pid, ""))
    return scores


def _resolve_checkpoint_dir_from_env() -> Path | None:
    for root_key, run_key in (("LOG_ROOT", "RUN_ID"), ("MZ_LOG_ROOT", "MZ_RUN_ID")):
        root = os.getenv(root_key)
        run_id = os.getenv(run_key)
        if root and run_id:
            return Path(root).expanduser() / run_id
    for root_key in ("SERVER_LOG_ROOT", "MZ_SERVER_LOG_ROOT"):
        root = os.getenv(root_key)
        if root:
            return Path(root).expanduser()
    return None


def load_weights(checkpoint_path: Path, map_location) -> dict:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        return checkpoint["weights"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format; expected dict with weights.")


def _infer_in_channels(weights: dict) -> int | None:
    for key in (
        "representation_network.module.conv.weight",
        "representation_network.conv.weight",
    ):
        tensor = weights.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 2:
            return int(tensor.shape[1])
    for key, tensor in weights.items():
        if "representation_network" not in key or "conv.weight" not in key:
            continue
        if hasattr(tensor, "shape") and len(tensor.shape) >= 2:
            return int(tensor.shape[1])
    for key, tensor in weights.items():
        if "representation_network" not in key or "weight" not in key:
            continue
        if hasattr(tensor, "dim") and tensor.dim() == 4:
            return int(tensor.shape[1])
    return None


def _build_obs_adapter(native_channels: int, model_channels: int, num_techs: int):
    if native_channels == model_channels:
        return None
    native_base = native_channels - (2 * num_techs + 1)
    model_base = model_channels - (2 * num_techs + 1)
    if native_base <= 0 or model_base <= 0:
        return None

    if native_base == model_base + 8:
        def adapter(obs):
            if obs.shape[0] != native_channels:
                return obs
            base_head = obs[:6]
            base_tail = obs[native_base - 8:native_base]
            research = obs[native_base:native_base + 2 * num_techs]
            final_plane = obs[native_base + 2 * num_techs:native_base + 2 * num_techs + 1]
            return numpy.concatenate((base_head, base_tail, research, final_plane), axis=0)

        return adapter

    if model_base == native_base + 8:
        def adapter(obs):
            if obs.shape[0] != native_channels:
                return obs
            base_head = obs[:6]
            pad = numpy.zeros((8, obs.shape[1], obs.shape[2]), dtype=obs.dtype)
            base_tail = obs[6:native_base]
            research = obs[native_base:native_base + 2 * num_techs]
            final_plane = obs[native_base + 2 * num_techs:native_base + 2 * num_techs + 1]
            return numpy.concatenate((base_head, pad, base_tail, research, final_plane), axis=0)

        return adapter

    def adapter(obs):
        if obs.shape[0] == model_channels:
            return obs
        if obs.shape[0] > model_channels:
            return obs[:model_channels]
        pad = numpy.zeros(
            (model_channels - obs.shape[0], obs.shape[1], obs.shape[2]),
            dtype=obs.dtype,
        )
        return numpy.concatenate((obs, pad), axis=0)

    return adapter


def _adapt_checkpoint_weights(weights: dict, model: torch.nn.Module) -> dict:
    model_weights = model.state_dict()
    adapted = dict(weights)
    changed = []

    for key, source in weights.items():
        target = model_weights.get(key)
        if target is None or source.shape == target.shape:
            continue
        if not hasattr(source, "dim"):
            continue

        patched = target.detach().clone()
        if "fc_policy" in key and source.dim() >= 1 and source.shape[1:] == target.shape[1:]:
            rows = min(int(source.shape[0]), int(target.shape[0]))
            patched[:rows] = source[:rows].to(device=patched.device, dtype=patched.dtype)
            adapted[key] = patched
            changed.append((key, tuple(source.shape), tuple(target.shape)))
            continue

        if (
            source.dim() == 2
            and target.dim() == 2
            and source.shape[0] == target.shape[0]
            and (
                ".fc." in key
                or ".fc_value." in key
                or ".fc_policy." in key
            )
        ):
            cols = min(int(source.shape[1]), int(target.shape[1]))
            patched[:, :cols] = source[:, :cols].to(
                device=patched.device,
                dtype=patched.dtype,
            )
            adapted[key] = patched
            changed.append((key, tuple(source.shape), tuple(target.shape)))

    for key, old_shape, new_shape in changed:
        print(
            "Using checkpoint compatibility adapter: "
            f"{key} checkpoint={old_shape}, remote={new_shape}",
            file=sys.stderr,
        )

    return adapted


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


def _query_player_scores(client=None):
    server_scores = _read_server_scorefile_scores()
    if server_scores:
        return server_scores
    if client is None or list_player_scores is None:
        return {}
    try:
        return list_player_scores(client)
    except Exception:
        return {}


def _query_player_score(client, player_id):
    if player_id is None:
        return None, None
    scores = _query_player_scores(client)
    if isinstance(scores, dict):
        score = scores.get(int(player_id), (None, None, ""))[0]
        if score is not None:
            return score, None
    return _query_score(client, player_id)


def _query_city_scores(client):
    if client is None or list_city_scores is None:
        return []
    try:
        return list_city_scores(client)
    except Exception:
        return []


def _city_score_data(game):
    cities = _query_city_scores(getattr(game, "client", None))
    state = getattr(game, "_last_state", None)
    cfg = getattr(state, "cfg", None)
    pop_weight = float(getattr(cfg, "score_population", 1.0))
    for city in cities:
        size = city.get("size")
        if isinstance(size, (int, float)):
            city["population_score"] = float(size) * pop_weight
    return cities


def _current_civ_score(game):
    state = getattr(game, "_last_state", None)
    if state is None:
        return None
    try:
        return state.civilization_score(1)
    except Exception:
        return None


def _pick_unit_move_action(game, legal_actions, step: int):
    state = getattr(game, "_last_state", None)
    if state is None:
        return None
    move_actions = []
    for action in legal_actions:
        action = int(action)
        if action < 0 or action >= state.MOVE_SIZE:
            continue
        if action % state.MOVE_PER_UNIT == state.HOLD_DIR:
            continue
        move_actions.append(action)
    if not move_actions:
        return None
    move_actions.sort()
    return move_actions[step % len(move_actions)]


def _run_direct_unit_move_demo(game, args) -> None:
    dir_ids = freeciv_remote.alpha_live.parse_dir_ids(args.dir_ids)
    max_actions = max(1, int(args.max_actions_per_turn or 6))
    for episode in range(args.episodes):
        game.reset()
        for turn in range(args.max_turns):
            try:
                game._sync_state()
            except Exception:
                break
            moved = 0
            units = [uid for uid in getattr(game, "unit_slots", []) if uid is not None]
            for unit_idx, unit_id in enumerate(units):
                if moved >= max_actions:
                    break
                for offset in range(len(dir_ids)):
                    dir_id = dir_ids[(turn + unit_idx + offset) % len(dir_ids)]
                    before = game.client.get_unit_pos(unit_id)
                    ok = game.client.move_dir_id(unit_id, dir_id)
                    if args.sleep:
                        time.sleep(args.sleep)
                    after = game.client.get_unit_pos(unit_id)
                    changed = before != after and after is not None
                    print(
                        f"[turn {turn} direct] unit_id={unit_id} dir_id={dir_id} "
                        f"success={ok} before={before} after={after}",
                        file=sys.stdout,
                    )
                    if ok and changed:
                        moved += 1
                        break
            try:
                success = game.client.end_turn()
                print(
                    f"[turn {turn} direct] end_turn success={success} moved={moved}",
                    file=sys.stdout,
                )
            except Exception:
                print(f"[turn {turn} direct] end_turn exception moved={moved}", file=sys.stdout)
                break
            game.turns += 1
            if args.sleep:
                time.sleep(args.sleep)


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
    ap.add_argument(
        "--no-sea-units",
        action="store_true",
        help="Disable naval unit production for maps without sea.",
    )
    ap.add_argument(
        "--max-actions-per-turn",
        type=int,
        help="Cap actions per turn (default: max_units*2).",
    )
    ap.add_argument("--player-id", type=int)
    ap.add_argument("--unit-id", type=int)
    ap.add_argument("--dir-ids", default="1,2,7,6,5,0")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--max-moves", type=int)
    ap.add_argument("--num-simulations", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--prefer-unit-move",
        action="store_true",
        help="Prefer legal non-hold unit movement actions; useful for short movement recordings.",
    )
    ap.add_argument(
        "--direct-unit-move-demo",
        action="store_true",
        help="Bypass MCTS and directly issue unit move commands for recording demos.",
    )
    ap.add_argument("--render", action="store_true")
    ap.add_argument(
        "--belief-tensorboard",
        action="store_true",
        help="Log belief heatmaps to TensorBoard once per turn.",
    )
    ap.add_argument(
        "--belief-tensorboard-dir",
        help="TensorBoard log directory for belief heatmaps.",
    )
    ap.add_argument(
        "--belief-tensorboard-interval",
        type=int,
        default=1,
        help="Log belief heatmaps every N turns.",
    )
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--score-log", help="Write civ scores to JSONL every N turns.")
    ap.add_argument("--score-log-interval", type=int, default=25)
    ap.add_argument("--city-score-log", help="Write per-city civ scores to JSONL every N turns.")
    ap.add_argument("--city-score-log-interval", type=int, default=25)
    ap.add_argument(
        "--turn-score-csv",
        help="Write per-turn player civ scores to this CSV (episode,turn,p0,p1).",
    )
    ap.add_argument(
        "--take-player",
        help='Send /take "<player>" via chat before LuaRemote control (ex: Condor).',
    )
    ap.add_argument(
        "--take-command",
        help="Send a raw server command via chat before LuaRemote control.",
    )
    ap.add_argument(
        "--start-after-take",
        action="store_true",
        help="Send /start after taking a pregame player, before LuaRemote control.",
    )
    ap.add_argument(
        "--start-command",
        help="Send a raw server command after take, before LuaRemote control.",
    )
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
    checkpoint_written = False
    default_checkpoint_dir = _resolve_checkpoint_dir_from_env()
    if default_checkpoint_dir is not None:
        _write_checkpoint_file(default_checkpoint_dir, checkpoint_path)
        checkpoint_written = True

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
        _write_checkpoint_file(json_path.parent, checkpoint_path)
        checkpoint_written = True
        json_fp = json_path.open("w", encoding="utf-8")
        json_records = []
        print(f"JSONL output: {json_path}", file=sys.stderr)

    turn_score_path = None
    turn_score_fp = None
    turn_score_writer = None
    if args.turn_score_csv:
        if list_player_scores is None:
            print(
                "Warning: turn score CSV requested but freeciv_rl is unavailable.",
                file=sys.stderr,
            )
        else:
            turn_score_path = Path(args.turn_score_csv).expanduser()
            _write_checkpoint_file(turn_score_path.parent, checkpoint_path)
            checkpoint_written = True
            file_exists = turn_score_path.exists()
            turn_score_fp = turn_score_path.open("a", newline="", encoding="utf-8")
            turn_score_writer = csv.writer(turn_score_fp)
            try:
                if not file_exists or turn_score_path.stat().st_size == 0:
                    turn_score_writer.writerow(
                        ["episode", "turn", "player0_score", "player1_score"]
                    )
                    turn_score_fp.flush()
            except OSError:
                pass

    score_log_path = None
    score_log_fp = None
    last_score_turn = None
    if args.score_log:
        score_log_path = Path(args.score_log).expanduser()
        _write_checkpoint_file(score_log_path.parent, checkpoint_path)
        checkpoint_written = True
        score_log_path.parent.mkdir(parents=True, exist_ok=True)
        score_log_fp = score_log_path.open("a", encoding="utf-8")
        try:
            if score_log_path.stat().st_size == 0:
                score_log_fp.write(f"# checkpoint: {args.checkpoint}\n")
                score_log_fp.flush()
        except OSError:
            pass
        print(f"Score log output: {score_log_path}", file=sys.stderr)

    city_score_log_path = None
    city_score_log_fp = None
    last_city_score_turn = None
    if args.city_score_log:
        if list_city_scores is None:
            print(
                "Warning: city score log requested but Lua score queries are unavailable.",
                file=sys.stderr,
            )
        else:
            city_score_log_path = Path(args.city_score_log).expanduser()
            _write_checkpoint_file(city_score_log_path.parent, checkpoint_path)
            checkpoint_written = True
            city_score_log_path.parent.mkdir(parents=True, exist_ok=True)
            city_score_log_fp = city_score_log_path.open("a", encoding="utf-8")
            try:
                if city_score_log_path.stat().st_size == 0:
                    city_score_log_fp.write(f"# checkpoint: {args.checkpoint}\n")
                    city_score_log_fp.flush()
            except OSError:
                pass
            print(f"City score log output: {city_score_log_path}", file=sys.stderr)

    if not checkpoint_written:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        default_dir = (
            Path(__file__).resolve().parent
            / "results"
            / "remote_play"
            / stamp
        )
        _write_checkpoint_file(default_dir, checkpoint_path)

    _set_env("FREECIV_HOST", args.host)
    _set_env("FREECIV_PORT", args.port)
    _set_env("FREECIV_LUAREMOTE_PORT", args.port)
    _set_env("FREECIV_MAP_W", args.map_width)
    _set_env("FREECIV_MAP_H", args.map_height)
    _set_env("FREECIV_NO_SEA_UNITS", "1" if args.no_sea_units else None)
    _set_env("FREECIV_MAX_TURNS", args.max_turns)
    _set_env("FREECIV_MAX_ACTIONS_PER_TURN", args.max_actions_per_turn)
    _set_env("FREECIV_PLAYER_ID", args.player_id)
    _set_env("FREECIV_UNIT_ID", args.unit_id)
    _set_env("FREECIV_TAKE_PLAYER", args.take_player)
    _set_env("FREECIV_TAKE_COMMAND", args.take_command)
    _set_env("FREECIV_START_AFTER_TAKE", "1" if args.start_after_take else None)
    _set_env("FREECIV_START_COMMAND", args.start_command)
    _set_env("FREECIV_DIR_IDS", args.dir_ids)
    _set_env("FREECIV_SLEEP", args.sleep)
    _set_env("FREECIV_BELIEF_TENSORBOARD", "1" if args.belief_tensorboard else None)
    _set_env("FREECIV_BELIEF_TENSORBOARD_DIR", args.belief_tensorboard_dir)
    _set_env("FREECIV_BELIEF_TENSORBOARD_INTERVAL", args.belief_tensorboard_interval)

    device = torch.device(args.device)
    weights = load_weights(checkpoint_path, map_location=device)

    config = freeciv_remote.MuZeroConfig()
    config.num_simulations = args.num_simulations
    native_obs_shape = config.observation_shape
    checkpoint_channels = _infer_in_channels(weights)
    obs_adapter = None
    if checkpoint_channels is not None and checkpoint_channels != native_obs_shape[0]:
        num_techs = len(freeciv_remote.MultiheadState.RESEARCH_TECHS)
        obs_adapter = _build_obs_adapter(
            native_obs_shape[0],
            checkpoint_channels,
            num_techs,
        )
        if obs_adapter is None:
            raise SystemExit(
                "Checkpoint observation channels do not match the remote encoder "
                f"({checkpoint_channels} vs {native_obs_shape[0]})."
            )
        config.observation_shape = (
            checkpoint_channels,
            native_obs_shape[1],
            native_obs_shape[2],
        )
        print(
            "Using observation compatibility adapter: "
            f"checkpoint={checkpoint_channels}, remote={native_obs_shape[0]}",
            file=sys.stderr,
        )

    model = models.MuZeroNetwork(config)
    weights = _adapt_checkpoint_weights(weights, model)
    model.set_weights(weights)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "<unset>")
        cuda_index = torch.cuda.current_device()
        cuda_name = torch.cuda.get_device_name(cuda_index)
        print(
            "Inference device: "
            f"{device} visible={visible_devices} cuda_index={cuda_index} name={cuda_name}",
            file=sys.stderr,
        )
    else:
        print(f"Inference device: {device}", file=sys.stderr)

    if args.max_moves is None:
        if hasattr(config, "max_actions_per_turn"):
            args.max_moves = args.max_turns * config.max_actions_per_turn
        else:
            args.max_moves = args.max_turns

    game = freeciv_remote.Game()
    if args.direct_unit_move_demo:
        _run_direct_unit_move_demo(game, args)
        game.close()
        return
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
    score_log_interval = int(args.score_log_interval or 0)
    city_score_log_interval = int(args.city_score_log_interval or 0)
    for episode in range(args.episodes):
        observation = game.reset()
        if args.belief_tensorboard and getattr(game, "belief_tb_enabled", False):
            log_dir = getattr(game, "belief_tb_dir", None)
            if log_dir:
                print(f"Belief TensorBoard log dir: {log_dir}", file=sys.stderr)
        if obs_adapter is not None:
            observation = obs_adapter(observation)
        done = False
        steps = 0
        start = time.monotonic()
        last_turn_csv = None
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
            if args.prefer_unit_move:
                move_action = _pick_unit_move_action(game, legal_actions, steps)
                if move_action is not None:
                    action = move_action
            observation, _reward, done = game.step(action)
            if obs_adapter is not None:
                observation = obs_adapter(observation)
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
            fc_score, fc_winner = _query_player_score(
                game.client, getattr(game, "player_id", None)
            )
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
            if (
                turn_score_writer is not None
                and turn is not None
                and turn > 0
                and turn != last_turn_csv
            ):
                scores = _query_player_scores(game.client)
                p0 = scores.get(0, (None, None, ""))[0]
                p1 = scores.get(1, (None, None, ""))[0]
                turn_score_writer.writerow([episode + 1, turn, p0, p1])
                turn_score_fp.flush()
                last_turn_csv = turn
            if (
                score_log_fp is not None
                and turn is not None
                and score_log_interval > 0
                and turn > 0
                and turn % score_log_interval == 0
                and turn != last_score_turn
            ):
                scores = _query_player_scores(game.client)
                p0 = scores.get(0, (None, None, ""))[0]
                p1 = scores.get(1, (None, None, ""))[0]
                payload = {
                    "turn": turn,
                    "player0_score": p0,
                    "player1_score": p1,
                }
                score_log_fp.write(json.dumps(payload) + "\n")
                score_log_fp.flush()
                last_score_turn = turn
            if (
                city_score_log_fp is not None
                and list_city_scores is not None
                and turn is not None
                and city_score_log_interval > 0
                and turn > 0
                and turn % city_score_log_interval == 0
                and turn != last_city_score_turn
            ):
                payload = {
                    "episode": episode + 1,
                    "turn": turn,
                    "players": _query_player_scores(game.client),
                    "cities": _city_score_data(game),
                }
                city_score_log_fp.write(json.dumps(payload) + "\n")
                city_score_log_fp.flush()
                last_city_score_turn = turn
            if args.render:
                game.render()
            steps += 1

        score, winner = _query_player_score(game.client, getattr(game, "player_id", None))
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
    if score_log_fp is not None:
        score_log_fp.close()
    if city_score_log_fp is not None:
        city_score_log_fp.close()
    if turn_score_fp is not None:
        turn_score_fp.close()

    game.close()


if __name__ == "__main__":
    main()
