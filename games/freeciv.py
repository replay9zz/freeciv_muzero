import datetime
import os
import pathlib
import sys
from collections import deque

import numpy
import torch

from .abstract_game import AbstractGame

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from freeciv_sim.state.config import MapConfig
from freeciv_sim.state.providers import RandomMapProvider
from freeciv_sim.rules.research import TECH_PREREQS
from freeciv_sim.rules.ruleset_loader import export_ruleset_config
from freeciv_sim.state.multihead_state import (
    MultiheadState,
    PRODUCTION_UNIT_NAMES,
    UNIT_TECHS,
)
from .tech_policy import pick_next_priority_tech


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _make_map_config(max_turns: int | None = None) -> MapConfig:
    if max_turns is None:
        max_turns = _env_int("FREECIV_MAX_TURNS", 100)
    allow_sea_units = not _env_bool("FREECIV_NO_SEA_UNITS", False)
    cfg = MapConfig(
        map_w=4,
        map_h=16,
        max_turns=max_turns,
        allow_sea_units=allow_sea_units,
    )
    # cfg = MapConfig(map_w=4, map_h=16, max_turns=2000)
    cfg.attack_reward = 0.2
    cfg.city_capture_reward = 8.0
    cfg.elimination_bonus = 4.5
    cfg.city_defense_multiplier = 1.5
    cfg.city_walls_defense_multiplier = 2.0
    cfg.production_queue_add = 3
    return cfg


class MuZeroConfig:
    def __init__(self):
        export_ruleset_config(pathlib.Path(__file__).resolve().parents[1] / "config")
        # fmt: off
        self.seed = 0
        self.max_num_gpus = 1

        self.map_config = _make_map_config()
        self.max_units = 6
        self.max_cities = 3
        self.max_actions_per_turn = _env_int(
            "FREECIV_MAX_ACTIONS_PER_TURN",
            max(1, self.max_units * 2),
        )

        ### Game
        tmp_state = MultiheadState(
            self.map_config,
            RandomMapProvider(self.map_config.map_w, self.map_config.map_h, p_open=1.0),
            max_units=self.max_units,
            max_cities=self.max_cities,
        )
        self.observation_shape = tmp_state.encode(1).shape
        self.action_space = list(range(tmp_state.ACTION_SIZE))
        self.players = list(range(2))
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0
        self.opponent = "self"

        ### Self-Play
        self.num_workers = 1
        self.selfplay_on_gpu = True
        self.max_turns = self.map_config.max_turns
        self.max_moves = self.max_turns * self.max_actions_per_turn
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = None
        self.use_stochastic_muzero = _env_bool("MUZERO_STOCHASTIC", False)

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.mcts_backup_operator = os.getenv("MUZERO_MCTS_BACKUP_OPERATOR", "wasserstein")
        self.mcts_wasserstein_power = _env_float("MUZERO_MCTS_WASSERSTEIN_POWER", 1.0)
        self.mcts_wasserstein_selection = os.getenv(
            "MUZERO_MCTS_WASSERSTEIN_SELECTION", "optimistic"
        )
        self.mcts_wasserstein_uncertainty_coef = _env_float(
            "MUZERO_MCTS_WASSERSTEIN_UNCERTAINTY_COEF", 0.25
        )
        self.mcts_wasserstein_min_std = _env_float(
            "MUZERO_MCTS_WASSERSTEIN_MIN_STD", 1e-6
        )
        self.mcts_wasserstein_shift_epsilon = _env_float(
            "MUZERO_MCTS_WASSERSTEIN_SHIFT_EPSILON", 1e-6
        )

        ### Network
        self.network = "resnet"
        self.support_size = 10

        # Residual Network
        self.downsample = False
        self.use_freeciv_hex_conv = _env_bool("FREECIV_HEX_CONV", False)
        self.blocks = 2
        self.channels = 32
        self.reduced_channels_reward = 2
        self.reduced_channels_value = 2
        self.reduced_channels_policy = 4
        self.resnet_fc_reward_layers = [64]
        self.resnet_fc_value_layers = [64]
        self.resnet_fc_policy_layers = [64]

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [64]
        self.fc_reward_layers = [64]
        self.fc_value_layers = [64]
        self.fc_policy_layers = [64]

        ### Training
        self.results_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "results"
            / pathlib.Path(__file__).stem
            / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )
        self.save_model = True
        self.training_steps = 3000
        self.batch_size = 128
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = True

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = 0.02
        self.lr_decay_rate = 0.8
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 1000
        self.num_unroll_steps = 20
        self.td_steps = 50
        self.PER = True
        self.PER_alpha = 0.5

        # Reanalyze
        self.use_last_model_value = True
        self.reanalyse_on_gpu = True

        ### Adjust the self play / training ratio
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = 1.5
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        if trained_steps < 0.75 * self.training_steps:
            return 0.5
        return 0.25


class Game(AbstractGame):
    def __init__(self, seed=None, config: MuZeroConfig | None = None):
        if config is not None and hasattr(config, "map_config"):
            self.config = config.map_config
            self.max_units = getattr(config, "max_units", 6)
            self.max_cities = getattr(config, "max_cities", 3)
        else:
            self.config = _make_map_config()
            self.max_units = 6
            self.max_cities = 3
        rng = numpy.random.default_rng(seed) if seed is not None else None
        self.provider = RandomMapProvider(
            self.config.map_w,
            self.config.map_h,
            p_open=1.0,
            rng=rng,
        )
        self.state = MultiheadState(
            self.config,
            self.provider,
            max_units=self.max_units,
            max_cities=self.max_cities,
        )
        self.player = 1

    def _score_diff(self) -> float:
        return float(
            self.state.civilization_score(1) - self.state.civilization_score(-1)
        )

    def _observation(self):
        return self.state.encode(self.player)

    def step(self, action):
        prev_score = float(self.state.civilization_score(self.player))
        visited_before = (
            int(self.state.visited[self.player].sum())
            if self.state.visited.get(self.player) is not None
            else 0
        )
        current_player = self.player
        prev_turn = self.state.turn
        self.state.step(current_player, action)
        done = self.state.terminal_reason is not None
        reward = float(self.state.civilization_score(current_player)) - prev_score
        if self.state.visited.get(current_player) is not None:
            visited_after = int(self.state.visited[current_player].sum())
            reward += (visited_after - visited_before) * self.config.move_reward
        if self.state.turn != prev_turn:
            self.player = -current_player
        return self._observation(), reward, done

    def to_play(self):
        return 0 if self.player == 1 else 1

    def _production_is_worker_like(self, name: str) -> bool:
        label = (name or "").lower()
        return any(tag in label for tag in ("worker", "engineer", "migrant"))

    def _deprioritize_worker_production(self, valid):
        state = self.state
        player = self.player
        if not state.cities[player]:
            return valid
        econ_offset = state.MOVE_SIZE + state.ATTACK_SIZE
        prod_start = econ_offset + state.ECON_PRODUCTION_OFFSET
        prod_end = econ_offset + state.ECON_PASS_OFFSET
        if prod_start >= len(valid):
            return valid
        for city_idx in range(len(state.cities[player])):
            start = prod_start + city_idx * state.PRODUCTION_ITEM_COUNT
            end = start + state.PRODUCTION_ITEM_COUNT
            if start >= len(valid):
                break
            end = min(end, len(valid), prod_end)
            if not valid[start:end].any():
                continue
            worker_actions = []
            other_unit_actions = []
            for item_idx in range(end - start):
                action_idx = start + item_idx
                if action_idx >= len(valid) or not valid[action_idx]:
                    continue
                kind, name = state.PRODUCTION_ITEM_NAMES[item_idx]
                if kind != "unit":
                    continue
                if self._production_is_worker_like(name):
                    worker_actions.append(action_idx)
                else:
                    other_unit_actions.append(action_idx)
            if other_unit_actions and worker_actions:
                for action_idx in worker_actions:
                    valid[action_idx] = 0
        return valid

    def _city_site_ok(self, x: int, y: int, player: int) -> bool:
        state = self.state
        if state.gt is None:
            return False
        if state.gt.au_map[y, x] != "A":
            return False
        if state._city_at(x, y, player) is not None:
            return False
        if state._city_at(x, y, -player) is not None:
            return False
        for p in (1, -1):
            for u in state.units[p]:
                if u.alive and u.x == x and u.y == y:
                    return False
        if state.movement is not None:
            for nx, ny in state.movement.get_native_neighbors(x, y):
                if nx is None or ny is None:
                    continue
                for u in state.units[-player]:
                    if u.alive and u.x == nx and u.y == ny:
                        return False
        return state._city_spacing_ok(player, x, y)

    def _settler_first_step(self, start: tuple[int, int], player: int):
        state = self.state
        if state.movement is None or state.gt is None:
            return None
        sx, sy = start
        neighbors = state.movement.get_native_neighbors(sx, sy)
        seen = {(sx, sy)}
        queue = deque()
        for dir_idx, (nx, ny) in enumerate(neighbors):
            if nx is None or ny is None:
                continue
            if state.gt.au_map[ny, nx] != "A":
                continue
            seen.add((nx, ny))
            queue.append((nx, ny, dir_idx, 1))
        while queue:
            cx, cy, first_dir, dist = queue.popleft()
            if self._city_site_ok(cx, cy, player):
                return first_dir, dist
            for nx, ny in state.movement.get_native_neighbors(cx, cy):
                if nx is None or ny is None:
                    continue
                if (nx, ny) in seen:
                    continue
                if state.gt.au_map[ny, nx] != "A":
                    continue
                seen.add((nx, ny))
                queue.append((nx, ny, first_dir, dist + 1))
        return None

    def _force_settler_city_actions(self, valid):
        state = self.state
        player = self.player
        if len(state.cities[player]) >= state.max_cities:
            return None
        econ_offset = state.MOVE_SIZE + state.ATTACK_SIZE
        build_base = econ_offset + state.ECON_BUILD_CITY_OFFSET
        build_actions = []
        best_move = None
        for idx, u in enumerate(state.units[player]):
            if not u.alive or not u.can_build_city:
                continue
            build_action = build_base + idx
            if 0 <= build_action < len(valid) and valid[build_action]:
                build_actions.append(build_action)
                continue
            step = self._settler_first_step((u.x, u.y), player)
            if step is None:
                continue
            dir_idx, dist = step
            action = idx * state.MOVE_PER_UNIT + dir_idx
            if 0 <= action < len(valid) and valid[action]:
                if best_move is None or dist < best_move[0]:
                    best_move = (dist, action)
        if build_actions:
            return build_actions
        if best_move is not None:
            return [best_move[1]]
        return None

    def legal_actions(self):
        valid = self.state.valid_moves(self.player)
        valid = self._deprioritize_worker_production(valid)
        forced = self._force_settler_city_actions(valid)
        if forced:
            return forced
        legal = [idx for idx, allowed in enumerate(valid) if allowed]
        if not legal:
            return []
        if not self.state.cities[self.player]:
            econ_offset = self.state.MOVE_SIZE + self.state.ATTACK_SIZE
            build_start = econ_offset + self.state.ECON_BUILD_CITY_OFFSET
            build_end = econ_offset + self.state.ECON_PRODUCTION_OFFSET
            build_candidates = [
                idx
                for idx in range(build_start, build_end)
                if 0 <= idx < len(valid) and valid[idx]
            ]
            if build_candidates:
                return build_candidates
        return legal

    def reset(self):
        self.state = MultiheadState(
            self.config,
            self.provider,
            max_units=self.max_units,
            max_cities=self.max_cities,
        )
        self.player = 1
        return self._observation()

    def render(self):
        print(self.state.string())
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        if action_number == self.state.PASS_ACTION:
            return "pass"
        if action_number < self.state.MOVE_SIZE:
            unit_idx = action_number // self.state.MOVE_PER_UNIT
            dir_idx = action_number % self.state.MOVE_PER_UNIT
            if dir_idx == self.state.HOLD_DIR:
                return f"hold_u{unit_idx}"
            return f"move_u{unit_idx}_d{dir_idx}"
        if action_number < self.state.MOVE_SIZE + self.state.ATTACK_SIZE:
            rel = action_number - self.state.MOVE_SIZE
            unit_idx = rel // self.state.ATTACK_PER_UNIT
            dir_idx = rel % self.state.ATTACK_PER_UNIT
            return f"attack_u{unit_idx}_d{dir_idx}"
        econ_idx = action_number - (self.state.MOVE_SIZE + self.state.ATTACK_SIZE)
        if 0 <= econ_idx < len(self.state.RESEARCH_TECHS):
            tech = self.state.RESEARCH_TECHS[econ_idx]
            return f"research_{tech}"
        if self.state.ECON_BUILD_CITY_OFFSET <= econ_idx < self.state.ECON_PRODUCTION_OFFSET:
            unit_idx = econ_idx - self.state.ECON_BUILD_CITY_OFFSET
            return f"build_city_u{unit_idx}"
        if self.state.ECON_PRODUCTION_OFFSET <= econ_idx < self.state.ECON_PASS_OFFSET:
            rel = econ_idx - self.state.ECON_PRODUCTION_OFFSET
            city_slot = rel // self.state.PRODUCTION_ITEM_COUNT
            item_idx = rel % self.state.PRODUCTION_ITEM_COUNT
            kind, name = self.state.PRODUCTION_ITEM_NAMES[item_idx]
            if kind == "unit":
                return f"produce_c{city_slot}_{name}"
            return f"build_c{city_slot}_{name}"
        return str(action_number)

    def expert_agent(self):
        state = self.state
        player = self.player
        valid = state.valid_moves(player)
        legal = self.legal_actions()
        if not legal:
            return state.PASS_ACTION

        econ_base = state.MOVE_SIZE + state.ATTACK_SIZE
        if not state.cities[player]:
            build_start = econ_base + state.ECON_BUILD_CITY_OFFSET
            build_end = econ_base + state.ECON_PRODUCTION_OFFSET
            build_candidates = [
                idx
                for idx in range(build_start, build_end)
                if 0 <= idx < len(valid) and valid[idx]
            ]
            if build_candidates:
                return int(numpy.random.choice(build_candidates))
        own_cities = {(c.x, c.y) for c in state.cities[player]}
        enemy_cities = {(c.x, c.y) for c in state.cities[-player]}
        friend_units = [
            (idx, u) for idx, u in enumerate(state.units[player]) if u.alive
        ]
        friend_positions = {(u.x, u.y) for _, u in friend_units}
        enemy_positions = {(u.x, u.y) for u in state.units[-player] if u.alive}
        garrison_units = {
            idx for idx, u in friend_units if (u.x, u.y) in own_cities
        }
        unguarded_cities = [pos for pos in own_cities if pos not in friend_positions]

        def best_bfs_step(start, target):
            if start == target or state.movement is None or state.gt is None:
                return None
            neighbors = state.movement.get_native_neighbors(*start)
            seen = {start}
            queue = deque()
            for dir_idx, (nx, ny) in enumerate(neighbors):
                if nx is None or ny is None:
                    continue
                if state.gt.au_map[ny, nx] != "A":
                    continue
                seen.add((nx, ny))
                queue.append((nx, ny, dir_idx))
            while queue:
                cx, cy, first_dir = queue.popleft()
                if (cx, cy) == target:
                    return first_dir
                for nx, ny in state.movement.get_native_neighbors(cx, cy):
                    if nx is None or ny is None:
                        continue
                    if (nx, ny) in seen:
                        continue
                    if state.gt.au_map[ny, nx] != "A":
                        continue
                    seen.add((nx, ny))
                    queue.append((nx, ny, first_dir))
            return None

        def approx_dist(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def friendly_support(target):
            count = 0
            for nx, ny in state.movement.get_native_neighbors(*target):
                if nx is None or ny is None:
                    continue
                if (nx, ny) in friend_positions:
                    count += 1
            return count

        # 1) Immediate garrison if a city is empty and reachable in one move.
        for city_pos in unguarded_cities:
            for unit_idx, u in friend_units:
                for dir_idx, (nx, ny) in enumerate(
                    state.movement.get_native_neighbors(u.x, u.y)
                ):
                    if (nx, ny) != city_pos:
                        continue
                    action = unit_idx * state.MOVE_PER_UNIT + dir_idx
                    if valid[action]:
                        return action

        # 2) Only attack when enough friendly units are adjacent to the target.
        attack_candidates = []
        for unit_idx, u in friend_units:
            for dir_idx, (nx, ny) in enumerate(
                state.movement.get_native_neighbors(u.x, u.y)
            ):
                if nx is None or ny is None:
                    continue
                action = (
                    state.MOVE_SIZE + unit_idx * state.ATTACK_PER_UNIT + dir_idx
                )
                if action >= len(valid) or not valid[action]:
                    continue
                target = (nx, ny)
                if target not in enemy_cities and target not in enemy_positions:
                    continue
                support = friendly_support(target)
                if target in enemy_cities:
                    if support >= 3:
                        attack_candidates.append((support + 10, action))
                else:
                    if support >= 2:
                        attack_candidates.append((support + 3, action))
        if attack_candidates:
            attack_candidates.sort(reverse=True)
            return attack_candidates[0][1]

        # 3) Prefer research along the priority chain when possible.
        archer_techs = UNIT_TECHS.get("Archers", [])
        archers_unlocked = (
            not archer_techs
            or all(state.research_done[player].get(tech, False) for tech in archer_techs)
        )
        next_tech = pick_next_priority_tech(
            state.research_done[player],
            TECH_PREREQS,
            state.RESEARCH_TECHS,
        )
        if next_tech and next_tech in state.RESEARCH_TECHS:
            tech_idx = state.RESEARCH_TECHS.index(next_tech)
            action = econ_base + tech_idx
            if action < len(valid) and valid[action]:
                return action

        # 4) Prefer producing archers once available.
        if archers_unlocked and "Archers" in state.PRODUCTION_UNIT_NAMES:
            arch_idx = state.PRODUCTION_UNIT_NAMES.index("Archers")
            for city_idx, city in enumerate(state.cities[player]):
                if city.production_target == "Archers":
                    continue
                action = (
                    econ_base
                    + state.ECON_PRODUCTION_OFFSET
                    + city_idx * state.PRODUCTION_ITEM_COUNT
                    + arch_idx
                )
                if valid[action]:
                    return action

        # 5) Move toward unguarded cities if possible.
        best_move = None
        for city_pos in unguarded_cities:
            for unit_idx, u in friend_units:
                if unit_idx in garrison_units:
                    continue
                dir_idx = best_bfs_step((u.x, u.y), city_pos)
                if dir_idx is None:
                    continue
                action = unit_idx * state.MOVE_PER_UNIT + dir_idx
                if not valid[action]:
                    continue
                dist = approx_dist((u.x, u.y), city_pos)
                if best_move is None or dist < best_move[0]:
                    best_move = (dist, action)
        if best_move is not None:
            return best_move[1]

        # 6) Advance toward the nearest enemy city once we have a small force.
        combat_units = [
            (idx, u) for idx, u in friend_units if not u.can_build_city
        ]
        if archers_unlocked and enemy_cities and len(combat_units) >= 2:
            best_attack_move = None
            for unit_idx, u in combat_units:
                if unit_idx in garrison_units:
                    continue
                for city_pos in enemy_cities:
                    dir_idx = best_bfs_step((u.x, u.y), city_pos)
                    if dir_idx is None:
                        continue
                    action = unit_idx * state.MOVE_PER_UNIT + dir_idx
                    if not valid[action]:
                        continue
                    dist = approx_dist((u.x, u.y), city_pos)
                    if best_attack_move is None or dist < best_attack_move[0]:
                        best_attack_move = (dist, action)
            if best_attack_move is not None:
                return best_attack_move[1]

        return int(numpy.random.choice(legal))
