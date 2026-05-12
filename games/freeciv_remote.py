import datetime
import os
import pathlib
import sys
import time

import numpy
import torch

from .abstract_game import AbstractGame

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from freeciv_alpha_zero.freeciv import live_agent as alpha_live
from freeciv_alpha_zero.freeciv.config import MapConfig
from freeciv_alpha_zero.freeciv.multihead_state import (
    City,
    MHUnit,
    MultiheadState,
    PRODUCTION_UNIT_NAMES,
    UNIT_SPECS,
)
from freeciv_alpha_zero.freeciv.providers import GroundTruth, RandomMapProvider


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0
        self.max_num_gpus = 0

        map_w = _env_int("FREECIV_MAP_W", 4)
        map_h = _env_int("FREECIV_MAP_H", 16)
        max_turns = _env_int("FREECIV_MAX_TURNS", 128)
        self.map_config = MapConfig(map_w=map_w, map_h=map_h, max_turns=max_turns)
        self.max_units = _env_int("FREECIV_MAX_UNITS", 6)
        self.max_cities = _env_int("FREECIV_MAX_CITIES", 3)

        ### Game
        tmp_state = MultiheadState(
            self.map_config,
            RandomMapProvider(self.map_config.map_w, self.map_config.map_h),
            max_units=self.max_units,
            max_cities=self.max_cities,
        )
        self.observation_shape = tmp_state.encode(1).shape
        self.action_space = list(range(tmp_state.ACTION_SIZE))
        self.players = list(range(1))
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0
        self.opponent = "self"

        ### Self-Play
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = self.map_config.max_turns
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"
        self.support_size = 10

        # Residual Network
        self.downsample = False
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
        self.training_steps = 0
        self.batch_size = 128
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = False

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = 0.02
        self.lr_decay_rate = 0.8
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 1
        self.num_unroll_steps = 20
        self.td_steps = 50
        self.PER = False
        self.PER_alpha = 0.5

        # Reanalyze
        self.use_last_model_value = False
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        return 0


class Game(AbstractGame):
    def __init__(self, seed=None):
        map_w = _env_int("FREECIV_MAP_W", 4)
        map_h = _env_int("FREECIV_MAP_H", 16)
        max_turns = _env_int("FREECIV_MAX_TURNS", 128)
        self.config = MapConfig(map_w=map_w, map_h=map_h, max_turns=max_turns)
        self.max_units = _env_int("FREECIV_MAX_UNITS", 6)
        self.max_cities = _env_int("FREECIV_MAX_CITIES", 3)
        self.dir_ids = alpha_live.parse_dir_ids(
            os.getenv("FREECIV_DIR_IDS", "0,1,4,7,6,3")
        )
        self.sleep = _env_float("FREECIV_SLEEP", 0.1)

        host = os.getenv("FREECIV_HOST", "127.0.0.1")
        port = _env_int("FREECIV_PORT", 4444)
        timeout = _env_float("FREECIV_TIMEOUT", 2.5)
        self.client = alpha_live.LuaRemoteClient(host, port, timeout=timeout)
        self.client.connect()

        self.player_id = _env_int("FREECIV_PLAYER_ID", None)
        self.unit_id = _env_int("FREECIV_UNIT_ID", None)
        self.controlled_units = []
        self.unit_slots = []
        self.unit_positions = []
        self.city_slots = []
        self.production_locked = set()
        self.current_research = None
        if self.unit_id is not None:
            pos_result = self.client.eval(alpha_live.simple_find_unit_pos(self.unit_id))
            pos_info = alpha_live.parse_position_result(pos_result)
            if pos_info and pos_info[2] is not None and pos_info[2] >= 0:
                self.player_id = int(pos_info[2])
        self._refresh_controlled_units()

        self.movement = alpha_live.FreecivMovement(
            map_width=self.config.map_w, map_height=self.config.map_h
        )
        self.known_tiles = {}
        self.known_enemy = {}
        self.visited_tiles = set()
        self.turns = 0
        self.actions_this_turn = 0
        self.max_actions_per_turn = _env_int(
            "FREECIV_MAX_ACTIONS_PER_TURN", max(1, self.max_units * 2)
        )
        self._last_snapshot = None
        self._last_state = None
        self.previous_pos = None

    def _refresh_controlled_units(self):
        controlled, self.player_id = alpha_live.discover_controlled_units(
            self.client, self.player_id
        )
        if not controlled and self.unit_id is not None:
            controlled = [self.unit_id]
        if not controlled:
            owned_cities = alpha_live.discover_player_cities(self.client, self.player_id)
            if owned_cities:
                self.controlled_units = []
                self.unit_id = None
                return
            raise RuntimeError(
                "No controllable units found. Set FREECIV_UNIT_ID or FREECIV_PLAYER_ID."
            )
        self.controlled_units = sorted(controlled)
        self.unit_id = self.controlled_units[0]

    def _try_refresh_controlled_units(self):
        try:
            controlled, self.player_id = alpha_live.discover_controlled_units(
                self.client, self.player_id
            )
        except Exception:
            return False
        if not controlled:
            return False
        self.controlled_units = sorted(controlled)
        self.unit_id = self.controlled_units[0]
        return True

    def _collect_unit_info(self):
        unit_entries = []
        unit_type_labels = {}
        for uid in self.controlled_units:
            pos_result = self.client.eval(alpha_live.simple_find_unit_pos(uid))
            pos_info = alpha_live.parse_position_result(pos_result)
            if pos_info is None:
                continue
            unit_type = alpha_live.get_unit_rule_name(self.client, uid) or ""
            unit_type_labels[uid] = unit_type
            unit_entries.append((uid, int(pos_info[0]), int(pos_info[1]), unit_type))
        unit_entries.sort(key=lambda entry: entry[0])
        return unit_entries, unit_type_labels

    def _visible_tiles_from_player(self):
        if self.player_id is None:
            return []
        lua = (
            "return (function() "
            f"local pl = find.player and find.player({int(self.player_id)}); "
            "if not pl then return '' end; "
            "local parts = {}; "
            f"for x=0,{int(self.config.map_w) - 1} do "
            f"  for y=0,{int(self.config.map_h) - 1} do "
            "    local t = find.tile and find.tile(x, y); "
            "    if t and t.seen and t:seen(pl) then "
            "      parts[#parts + 1] = string.format('%d,%d', x, y); "
            "    end "
            "  end "
            "end; "
            "return table.concat(parts, ';') "
            "end)()"
        )
        try:
            result = self.client.eval(lua)
            payload = result.last_return() if result else None
        except Exception:
            payload = None
        tiles = []
        if payload:
            for entry in str(payload).split(";"):
                if not entry:
                    continue
                parts = entry.split(",", 1)
                if len(parts) != 2:
                    continue
                try:
                    tiles.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
        return tiles

    def _snapshot_from_city(self, owned_cities):
        if not owned_cities:
            raise RuntimeError("No controllable units or cities available.")
        _, cx, cy = owned_cities[0]
        if self._last_snapshot is not None:
            au_map = self._last_snapshot.au_map.copy()
            enemy_map = self._last_snapshot.enemy_map.copy()
            visited = self._last_snapshot.visited.copy()
            revealed = self._last_snapshot.revealed.copy()
            status_lookup = dict(self._last_snapshot.status_lookup)
        else:
            au_map = numpy.full(
                (self.config.map_h, self.config.map_w), "U", dtype="<U1"
            )
            enemy_map = numpy.zeros((self.config.map_h, self.config.map_w), dtype=bool)
            visited = numpy.zeros((self.config.map_h, self.config.map_w), dtype=bool)
            revealed = numpy.zeros((self.config.map_h, self.config.map_w), dtype=bool)
            status_lookup = {}
        cx, cy = int(cx), int(cy)
        if 0 <= cy < self.config.map_h and 0 <= cx < self.config.map_w:
            visited[cy, cx] = True
            revealed[cy, cx] = True
        visible_tiles = self._visible_tiles_from_player()
        for vx, vy in visible_tiles:
            if 0 <= vy < self.config.map_h and 0 <= vx < self.config.map_w:
                visited[vy, vx] = True
                revealed[vy, vx] = True
        if visible_tiles:
            self.visited_tiles.update(visible_tiles)

        research_name = None
        research_done = False
        research_flags = {tech: False for tech in MultiheadState.RESEARCH_TECHS}
        if self.player_id is not None:
            try:
                research_name = alpha_live.query_player_research(self.client, self.player_id)
            except Exception:
                research_name = None
            try:
                research_done = alpha_live.is_target_researched(self.client, self.player_id)
            except Exception:
                research_done = False
            for tech in MultiheadState.RESEARCH_TECHS:
                try:
                    research_flags[tech] = alpha_live.simple_knows_tech(
                        self.client, self.player_id, tech
                    )
                except Exception:
                    continue

        snapshot = alpha_live.Snapshot(
            au_map=au_map,
            enemy_map=enemy_map,
            visited=visited,
            revealed=revealed,
            player_pos=(cx, cy),
            enemy_pos=(-1, -1),
            status_lookup=status_lookup,
            research_name=research_name,
            research_done=research_done,
            research_flags=research_flags,
        )
        return snapshot

    def _build_state(self, snapshot, owned_cities):
        unit_entries, _unit_type_labels = self._collect_unit_info()
        unit_names = {name.lower(): name for name in UNIT_SPECS.keys()}
        state = MultiheadState.__new__(MultiheadState)  # type: ignore[misc]
        state.cfg = self.config
        state.provider = None
        state.max_units = self.max_units
        state.max_cities = self.max_cities
        state.rng = numpy.random.default_rng()
        state.movement = alpha_live.FreecivMovement(self.config.map_w, self.config.map_h)
        au_map = snapshot.au_map.copy()
        au_map[au_map == "U"] = "A"
        state.gt = GroundTruth(au_map, snapshot.enemy_map.copy())
        state.units = {1: [], -1: []}
        state.cities = {1: [], -1: []}
        state.research_done = {
            1: {
                tech: snapshot.research_flags.get(tech, False)
                for tech in MultiheadState.RESEARCH_TECHS
            },
            -1: {tech: False for tech in MultiheadState.RESEARCH_TECHS},
        }
        state.turn = self.turns
        state.actions_this_turn = 0
        state.max_actions_per_turn = max(1, self.max_units * 2)
        state.kills = {1: 0, -1: 0}
        state.scores = {1: 0.0, -1: 0.0}
        state.winner = None
        state.terminal_reason = None
        state.RESEARCH_TECHS = MultiheadState.RESEARCH_TECHS
        state.MOVE_PER_UNIT = MultiheadState.MOVE_PER_UNIT
        state.ATTACK_PER_UNIT = MultiheadState.ATTACK_PER_UNIT
        state.MOVE_SIZE = self.max_units * state.MOVE_PER_UNIT
        state.ATTACK_SIZE = self.max_units * state.ATTACK_PER_UNIT
        state.ECON_RESEARCH_OFFSET = 0
        state.ECON_BUILD_CITY_OFFSET = len(state.RESEARCH_TECHS)
        state.ECON_PRODUCTION_OFFSET = state.ECON_BUILD_CITY_OFFSET + self.max_units
        state.PRODUCTION_UNIT_COUNT = len(PRODUCTION_UNIT_NAMES)
        state.ECON_PASS_OFFSET = (
            state.ECON_PRODUCTION_OFFSET + self.max_cities * state.PRODUCTION_UNIT_COUNT
        )
        state.ECON_SIZE = state.ECON_PASS_OFFSET + 1
        state.ACTION_SIZE = state.MOVE_SIZE + state.ATTACK_SIZE + state.ECON_SIZE
        state.PASS_ACTION = state.ACTION_SIZE - 1

        self.unit_slots = []
        self.unit_positions = []
        can_build_flags = []
        for uid, ux, uy, unit_type in unit_entries[: self.max_units]:
            unit_key = (unit_type or "").strip().lower()
            unit_name = unit_names.get(unit_key)
            if unit_name is None and unit_key.endswith("s"):
                unit_name = unit_names.get(unit_key[:-1])
            if unit_name is None and ("settler" in unit_key or "migrant" in unit_key):
                unit_name = "Settlers"
            spec = UNIT_SPECS.get(unit_name or "") or UNIT_SPECS.get("Warriors")
            if spec is None:
                unit = MHUnit(ux, uy, 10, 2, 1, 1, unit_name or "Warriors", True, False, None)
            else:
                unit = MHUnit(
                    ux,
                    uy,
                    spec.hp,
                    spec.atk,
                    spec.df,
                    spec.firepower,
                    spec.name,
                    True,
                    spec.can_build_city or ("settler" in unit_key or "migrant" in unit_key),
                    None,
                )
            can_build_flags.append(unit.can_build_city)
            state.units[1].append(unit)
            self.unit_slots.append(uid)
            self.unit_positions.append((ux, uy))
        while len(state.units[1]) < self.max_units:
            state.units[1].append(MHUnit(0, 0, 0, 0, 0, 1, "None", False, False, None))
            self.unit_slots.append(None)
            self.unit_positions.append(None)

        if not owned_cities and state.units[1] and not any(can_build_flags):
            state.units[1][0].can_build_city = True

        enemy_coords = [
            (int(x), int(y)) for (y, x) in numpy.argwhere(snapshot.enemy_map)
        ]
        for ex, ey in enemy_coords[: self.max_units]:
            spec = UNIT_SPECS.get("Warriors")
            if spec is None:
                enemy = MHUnit(ex, ey, 10, 2, 1, 1, "Warriors", True, False, None)
            else:
                enemy = MHUnit(
                    ex,
                    ey,
                    spec.hp,
                    spec.atk,
                    spec.df,
                    spec.firepower,
                    spec.name,
                    True,
                    False,
                    None,
                )
            state.units[-1].append(enemy)
        while len(state.units[-1]) < self.max_units:
            state.units[-1].append(MHUnit(0, 0, 0, 0, 0, 1, "None", False, False, None))

        self.city_slots = []
        for city_id, cx, cy in owned_cities[: self.max_cities]:
            state.cities[1].append(City(x=int(cx), y=int(cy), size=1))
            self.city_slots.append(city_id)

        return state

    def _sync_state(self):
        if self.unit_id is None:
            owned_cities = alpha_live.discover_player_cities(self.client, self.player_id)
            snapshot = self._snapshot_from_city(owned_cities)
        else:
            try:
                snapshot, self.player_id = alpha_live.gather_snapshot(
                    client=self.client,
                    movement=self.movement,
                    cfg=self.config,
                    unit_id=self.unit_id,
                    player_id=self.player_id,
                    known_tiles=self.known_tiles,
                    known_enemy=self.known_enemy,
                    visited_tiles=self.visited_tiles,
                )
            except Exception:
                self._refresh_controlled_units()
                self.previous_pos = None
                if self.unit_id is None:
                    owned_cities = alpha_live.discover_player_cities(
                        self.client, self.player_id
                    )
                    snapshot = self._snapshot_from_city(owned_cities)
                else:
                    snapshot, self.player_id = alpha_live.gather_snapshot(
                        client=self.client,
                        movement=self.movement,
                        cfg=self.config,
                        unit_id=self.unit_id,
                        player_id=self.player_id,
                        known_tiles=self.known_tiles,
                        known_enemy=self.known_enemy,
                        visited_tiles=self.visited_tiles,
                    )
        self.visited_tiles.add(snapshot.player_pos)
        self._last_snapshot = snapshot
        self._try_refresh_controlled_units()
        owned_cities = alpha_live.discover_player_cities(self.client, self.player_id)
        self._last_state = self._build_state(snapshot, owned_cities)
        self.production_locked.intersection_update(set(self.city_slots))
        self.current_research = None
        if isinstance(snapshot.research_name, str):
            if snapshot.research_name.startswith("__TECH__"):
                self.current_research = snapshot.research_name.replace("__TECH__", "", 1).strip()
        return self._last_state

    def _apply_action(self, action, board_state, owned_cities):
        if action == board_state.PASS_ACTION:
            return

        if action < board_state.MOVE_SIZE:
            unit_idx = action // board_state.MOVE_PER_UNIT
            dir_idx = action % board_state.MOVE_PER_UNIT
            if unit_idx >= len(self.unit_slots):
                return
            unit_id = self.unit_slots[unit_idx]
            unit_pos = self.unit_positions[unit_idx]
            if unit_id is None or unit_pos is None:
                return
            if dir_idx >= len(self.dir_ids):
                return
            dir_id = self.dir_ids[dir_idx]
            success = self.client.move_dir_id(unit_id, dir_id)
            if success and self._last_snapshot is not None:
                self.previous_pos = self._last_snapshot.player_pos
            return

        if action < board_state.MOVE_SIZE + board_state.ATTACK_SIZE:
            rel = action - board_state.MOVE_SIZE
            unit_idx = rel // board_state.ATTACK_PER_UNIT
            dir_idx = rel % board_state.ATTACK_PER_UNIT
            if unit_idx >= len(self.unit_slots):
                return
            unit_id = self.unit_slots[unit_idx]
            unit_pos = self.unit_positions[unit_idx]
            if unit_id is None or unit_pos is None:
                return
            neighbors = self.movement.get_native_neighbors(unit_pos[0], unit_pos[1])
            if dir_idx >= len(neighbors):
                return
            nx, ny = neighbors[dir_idx]
            if nx is None or ny is None:
                return
            self.client.attack_target(unit_id, int(nx), int(ny))
            return

        econ_idx = action - (board_state.MOVE_SIZE + board_state.ATTACK_SIZE)
        if 0 <= econ_idx < len(board_state.RESEARCH_TECHS):
            if not owned_cities:
                return
            tech_name = board_state.RESEARCH_TECHS[econ_idx]
            alpha_live.set_research_to_target(
                self.client,
                self.player_id,
                research_flags=self._last_snapshot.research_flags,
                tech_name=tech_name,
            )
            return
        if board_state.ECON_BUILD_CITY_OFFSET <= econ_idx < board_state.ECON_PRODUCTION_OFFSET:
            unit_idx = econ_idx - board_state.ECON_BUILD_CITY_OFFSET
            if unit_idx >= len(self.unit_slots):
                return
            unit_id = self.unit_slots[unit_idx]
            if unit_id is None:
                return
            city_name = f"MuZeroCity{len(owned_cities) + 1}"
            built = self.client.found_city(unit_id, city_name)
            if not built:
                self.client.build_city(unit_id)
            return
        if board_state.ECON_PRODUCTION_OFFSET <= econ_idx < board_state.ECON_PASS_OFFSET:
            rel = econ_idx - board_state.ECON_PRODUCTION_OFFSET
            city_slot = rel // board_state.PRODUCTION_UNIT_COUNT
            unit_idx = rel % board_state.PRODUCTION_UNIT_COUNT
            if city_slot >= len(self.city_slots):
                return
            if unit_idx >= len(PRODUCTION_UNIT_NAMES):
                return
            city_id = self.city_slots[city_slot]
            unit_name = PRODUCTION_UNIT_NAMES[unit_idx]
            if self.client.set_city_production(city_id, "UnitType", unit_name):
                self.production_locked.add(city_id)
            return

    def step(self, action):
        prev_visited = len(self.visited_tiles)
        try:
            board_state = self._sync_state()
        except Exception:
            return self.reset(), 0.0, True

        owned_cities = alpha_live.discover_player_cities(self.client, self.player_id)
        valid_actions = board_state.valid_moves(1)
        if not owned_cities:
            econ_offset = board_state.MOVE_SIZE + board_state.ATTACK_SIZE
            build_offset = econ_offset + board_state.ECON_BUILD_CITY_OFFSET
            for unit_idx in range(self.max_units):
                action_idx = build_offset + unit_idx
                if 0 <= action_idx < len(valid_actions) and valid_actions[action_idx]:
                    action = action_idx
                    break
        self._apply_action(action, board_state, owned_cities)
        self.actions_this_turn += 1

        end_turn = action == board_state.PASS_ACTION
        if self.actions_this_turn >= self.max_actions_per_turn:
            end_turn = True

        if end_turn:
            try:
                self.client.end_turn()
            except Exception:
                pass
            self.turns += 1
            self.actions_this_turn = 0
            if self.sleep:
                time.sleep(self.sleep)

        done = self.turns >= self.config.max_turns
        try:
            board_state = self._sync_state()
        except Exception:
            done = True
            board_state = self._last_state

        new_visited = len(self.visited_tiles)
        reward = float(new_visited - prev_visited)
        if board_state is not None:
            observation = board_state.encode(1)
        else:
            observation = numpy.zeros(
                (_observation_channels(), self.config.map_h, self.config.map_w),
                dtype=numpy.float32,
            )
        return observation, reward, done

    def to_play(self):
        return 0

    def legal_actions(self):
        if self._last_state is None:
            try:
                self._sync_state()
            except Exception:
                return []
        valid = self._last_state.valid_moves(1)
        alive_units = any(u.alive for u in self._last_state.units[1])
        if not self.city_slots:
            econ_offset = self._last_state.MOVE_SIZE + self._last_state.ATTACK_SIZE
            research_end = econ_offset + self._last_state.ECON_BUILD_CITY_OFFSET
            prod_start = econ_offset + self._last_state.ECON_PRODUCTION_OFFSET
            prod_end = econ_offset + self._last_state.ECON_PASS_OFFSET
            build_start = econ_offset + self._last_state.ECON_BUILD_CITY_OFFSET
            build_end = econ_offset + self._last_state.ECON_PRODUCTION_OFFSET
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
        if self.current_research:
            econ_offset = self._last_state.MOVE_SIZE + self._last_state.ATTACK_SIZE
            research_end = econ_offset + self._last_state.ECON_BUILD_CITY_OFFSET
            valid[econ_offset:research_end] = 0
        if self.production_locked and self.city_slots:
            econ_offset = self._last_state.MOVE_SIZE + self._last_state.ATTACK_SIZE
            prod_start = econ_offset + self._last_state.ECON_PRODUCTION_OFFSET
            for slot_idx, city_id in enumerate(self.city_slots):
                if city_id not in self.production_locked:
                    continue
                start = prod_start + slot_idx * self._last_state.PRODUCTION_UNIT_COUNT
                end = start + self._last_state.PRODUCTION_UNIT_COUNT
                valid[start:end] = 0
        non_pass = valid.copy()
        non_pass[self._last_state.PASS_ACTION] = 0
        if non_pass.any() and alive_units:
            valid[self._last_state.PASS_ACTION] = 0
        return [idx for idx, allowed in enumerate(valid) if allowed]

    def reset(self):
        self.turns = 0
        state = self._sync_state()
        return state.encode(1)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    def render(self):
        if self._last_state is None:
            self._sync_state()
        if self._last_state is not None:
            print(self._last_state.string())

    def action_to_string(self, action_number):
        tmp_state = self._last_state
        if tmp_state is None:
            tmp_state = MultiheadState(
                self.config,
                RandomMapProvider(self.config.map_w, self.config.map_h),
                max_units=self.max_units,
                max_cities=self.max_cities,
            )
        if action_number == tmp_state.PASS_ACTION:
            return "pass"
        if action_number < tmp_state.MOVE_SIZE:
            unit_idx = action_number // tmp_state.MOVE_PER_UNIT
            dir_idx = action_number % tmp_state.MOVE_PER_UNIT
            return f"move_u{unit_idx}_d{dir_idx}"
        if action_number < tmp_state.MOVE_SIZE + tmp_state.ATTACK_SIZE:
            rel = action_number - tmp_state.MOVE_SIZE
            unit_idx = rel // tmp_state.ATTACK_PER_UNIT
            dir_idx = rel % tmp_state.ATTACK_PER_UNIT
            return f"attack_u{unit_idx}_d{dir_idx}"
        econ_idx = action_number - (tmp_state.MOVE_SIZE + tmp_state.ATTACK_SIZE)
        if 0 <= econ_idx < len(tmp_state.RESEARCH_TECHS):
            tech = tmp_state.RESEARCH_TECHS[econ_idx]
            return f"research_{tech}"
        if tmp_state.ECON_BUILD_CITY_OFFSET <= econ_idx < tmp_state.ECON_PRODUCTION_OFFSET:
            unit_idx = econ_idx - tmp_state.ECON_BUILD_CITY_OFFSET
            return f"build_city_u{unit_idx}"
        if tmp_state.ECON_PRODUCTION_OFFSET <= econ_idx < tmp_state.ECON_PASS_OFFSET:
            rel = econ_idx - tmp_state.ECON_PRODUCTION_OFFSET
            city_slot = rel // tmp_state.PRODUCTION_UNIT_COUNT
            unit_idx = rel % tmp_state.PRODUCTION_UNIT_COUNT
            unit_name = PRODUCTION_UNIT_NAMES[unit_idx]
            return f"produce_c{city_slot}_{unit_name}"
        return str(action_number)
