import datetime
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from typing import Optional

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
    BUILDING_REQ_BUILDINGS,
    BUILDING_TECHS,
    GREAT_WONDER_NAMES,
    PRODUCTION_ITEM_NAMES,
    PRODUCTION_BUILDING_NAMES,
    PRODUCTION_UNIT_NAMES,
    UnitSpec,
    UNIT_GENERATION_CHAINS,
    UNIT_TECHS,
    UNIT_SPECS,
    UNIT_OBSOLETE_BY,
)
from freeciv_alpha_zero.freeciv.research_policy import TECH_PREREQS, build_tech_costs
from freeciv_alpha_zero.freeciv.providers import GroundTruth, RandomMapProvider
from freeciv_rl.lua_helper import (
    auto_settler as lua_auto_settler,
    list_city_adjacent_water,
    list_city_buildings,
    list_all_unit_status,
    list_player_scores,
    list_city_sizes,
    list_tile_owners,
    list_units_by_homecity,
    set_government as lua_set_government,
)

from .tech_policy import pick_next_priority_tech


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


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0
        self.max_num_gpus = 1

        map_w = _env_int("FREECIV_MAP_W", 4)
        map_h = _env_int("FREECIV_MAP_H", 16)
        max_turns = _env_int("FREECIV_MAX_TURNS", 2000)
        self.map_config = MapConfig(map_w=map_w, map_h=map_h, max_turns=max_turns)
        self.max_units = _env_int("FREECIV_MAX_UNITS", 6)
        self.max_cities = _env_int("FREECIV_MAX_CITIES", 3)
        self.luaremote_port_base = _env_int(
            "FREECIV_LUAREMOTE_PORT",
            _env_int("FREECIV_PORT", 4444),
        )
        self.luaremote_port_stride = _env_int("FREECIV_LUAREMOTE_PORT_STRIDE", 1)
        ### Game
        tmp_state = MultiheadState(
            self.map_config,
            RandomMapProvider(self.map_config.map_w, self.map_config.map_h, p_open=1.0),
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
        self.selfplay_on_gpu = True
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
        self.train_on_gpu = True

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
        self.reanalyse_on_gpu = True

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
        max_turns = _env_int("FREECIV_MAX_TURNS", 2000)
        self.config = MapConfig(map_w=map_w, map_h=map_h, max_turns=max_turns)
        self.max_units = _env_int("FREECIV_MAX_UNITS", 6)
        self.max_cities = _env_int("FREECIV_MAX_CITIES", 3)
        self.dir_ids = alpha_live.parse_dir_ids(
            os.getenv("FREECIV_DIR_IDS", "0,1,4,7,6,3")
        )
        self.sleep = _env_float("FREECIV_SLEEP", 0.1)

        self.host = os.getenv("FREECIV_HOST", "127.0.0.1")
        self.port = _env_int(
            "FREECIV_LUAREMOTE_PORT",
            _env_int("FREECIV_PORT", 4444),
        )
        self.timeout = _env_float("FREECIV_TIMEOUT", 2.5)
        self.client_cmd = os.getenv("FREECIV_CLIENT_CMD")
        self.restart_on_reset = _env_bool("FREECIV_CLIENT_RESTART", False)
        self.client_start_wait = _env_float("FREECIV_CLIENT_START_WAIT", 1.0)
        self.client_start_timeout = _env_float("FREECIV_CLIENT_START_TIMEOUT", 30.0)
        self._needs_restart = False
        self._client_process = None

        if self.client_cmd:
            self._start_client_process()

        self.client = None
        self._connect_client()

        self.player_id = _env_int("FREECIV_PLAYER_ID", None)
        self.unit_id = _env_int("FREECIV_UNIT_ID", None)
        self.controlled_units = []
        self.unit_slots = []
        self.unit_positions = []
        self.unit_slot_types = []
        self.unit_type_labels = {}
        self.unit_status: list[tuple[int, int, int, int, str, int, int]] = []
        self.city_slots = []
        self.city_sizes: dict[int, int] = {}
        self.production_queue: dict[int, list[tuple[str, str]]] = {}
        self.production_current: dict[int, tuple[str, str] | None] = {}
        self.max_production_queue = _env_int("FREECIV_PRODUCTION_QUEUE_MAX", 3)
        self.production_queue_add = _env_int("FREECIV_PRODUCTION_QUEUE_ADD", 3)
        self._city_buildings: dict[int, set[str]] = {}
        self._city_unit_counts: dict[int, dict[str, int]] = {}
        self._city_adjacent_water: dict[int, dict[str, bool]] = {}
        self.current_research = None
        self._last_research_flags: dict[str, bool] = {}
        self._buildable_units: set[str] = set()
        self._buildable_buildings: set[str] = set()
        self.autosettler_units: set[int] = set()
        self.gov_switch_sent: set[str] = set()
        self.visible_enemy_units: list[tuple[int, int]] = []
        self.visible_enemy_cities: list[tuple[int, int]] = []
        self.visible_enemy_city_ids: dict[tuple[int, int], int] = {}
        self.visible_enemy_unit_coords: set[tuple[int, int]] = set()
        self.tile_owners: dict[tuple[int, int], int] = {}
        self.player_scores: dict[int, tuple[Optional[float], Optional[bool], str]] = {}
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
        self.acted_unit_slots: set[int] = set()
        self.acted_production_cities: set[int] = set()
        self._last_snapshot = None
        self._last_state = None
        self.previous_pos = None
        self._tile_owner_refresh_pending = True

    def _start_client_process(self) -> None:
        if not self.client_cmd:
            return
        if self._client_process and self._client_process.poll() is None:
            return
        cmd = shlex.split(self.client_cmd)
        if not cmd:
            raise RuntimeError("FREECIV_CLIENT_CMD is empty.")
        self._client_process = subprocess.Popen(cmd, start_new_session=True)

    def _stop_client_process(self) -> None:
        if not self._client_process:
            return
        if self._client_process.poll() is not None:
            self._client_process = None
            return
        try:
            os.killpg(self._client_process.pid, signal.SIGTERM)
            self._client_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self._client_process.pid, signal.SIGKILL)
            self._client_process.wait(timeout=5)
        finally:
            self._client_process = None

    def _restart_client_process(self) -> None:
        self._stop_client_process()
        self._start_client_process()

    def _shutdown_client_for_restart(self) -> None:
        try:
            if self.client is not None:
                self.client.close()
        except Exception:
            pass
        self.client = None
        if self.client_cmd:
            self._stop_client_process()
        self._needs_restart = True

    def _connect_client(self) -> None:
        if self.client is None:
            self.client = alpha_live.LuaRemoteClient(
                self.host, self.port, timeout=self.timeout
            )
        deadline = time.monotonic() + self.client_start_timeout
        last_exc = None
        while time.monotonic() < deadline:
            try:
                self.client.connect()
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(self.client_start_wait)
        raise RuntimeError(
            f"Failed to connect to LuaRemote at {self.host}:{self.port}"
        ) from last_exc

    def _refresh_tile_owners(self) -> None:
        if self.client is None:
            return
        try:
            self.tile_owners = list_tile_owners(
                self.client, self.config.map_w, self.config.map_h
            )
            self._tile_owner_refresh_pending = False
        except Exception:
            self.tile_owners = {}
            self._tile_owner_refresh_pending = True

    def _refresh_unit_status(self) -> None:
        if self.client is None:
            self.unit_status = []
            return
        try:
            self.unit_status = list_all_unit_status(self.client)
        except Exception:
            self.unit_status = []

    def _refresh_player_scores(self) -> None:
        if self.client is None:
            self.player_scores = {}
            return
        try:
            self.player_scores = list_player_scores(self.client)
        except Exception:
            self.player_scores = {}

    def _unit_status_by_id(self) -> dict[int, tuple[int, int, int, str, int, int]]:
        status_by_id: dict[int, tuple[int, int, int, str, int, int]] = {}
        for uid, ux, uy, owner, name, hp, moves in self.unit_status:
            status_by_id[int(uid)] = (int(ux), int(uy), int(owner), name, int(hp), int(moves))
        return status_by_id

    def _refresh_controlled_units(self):
        controlled, self.player_id = alpha_live.discover_controlled_units(
            self.client, self.player_id
        )
        if not controlled and self.unit_id is not None:
            pos_result = self.client.eval(alpha_live.simple_find_unit_pos(self.unit_id))
            pos_info = alpha_live.parse_position_result(pos_result)
            if pos_info is not None:
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

    def _collect_unit_info(self, status_by_id):
        unit_entries = []
        unit_type_labels = {}
        for uid in self.controlled_units:
            unit_hp = None
            status = status_by_id.get(uid)
            if status is not None:
                ux, uy, _owner, unit_type, hp, _moves = status
                unit_type = unit_type or ""
                unit_hp = hp
                unit_type_labels[uid] = unit_type
                unit_entries.append((uid, int(ux), int(uy), unit_type, unit_hp))
                continue
            pos_result = self.client.eval(alpha_live.simple_find_unit_pos(uid))
            pos_info = alpha_live.parse_position_result(pos_result)
            if pos_info is None:
                continue
            unit_type = alpha_live.get_unit_rule_name(self.client, uid) or ""
            unit_type_labels[uid] = unit_type
            unit_entries.append((uid, int(pos_info[0]), int(pos_info[1]), unit_type, unit_hp))
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

    def _visible_enemy_cities(self):
        if self.player_id is None:
            return []
        try:
            cities = alpha_live.list_all_cities(self.client)
        except Exception:
            return []
        visible = []
        for city_id, cx, cy, owner, _name in cities:
            if owner == self.player_id:
                continue
            coord = (int(cx), int(cy))
            visible.append((int(city_id), coord[0], coord[1]))
        return visible

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

    def _build_state(self, snapshot, owned_cities, enemy_cities):
        status_by_id = self._unit_status_by_id()
        unit_entries, unit_type_labels = self._collect_unit_info(status_by_id)
        self.unit_type_labels = unit_type_labels
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
        state.research_target = {1: None, -1: None}
        state.research_progress = {1: 0.0, -1: 0.0}
        state.tech_costs = build_tech_costs(
            TECH_PREREQS,
            style=self.config.tech_cost_style,
            base_cost=self.config.base_tech_cost,
            min_cost=self.config.min_tech_cost,
            cost_factor=getattr(self.config, "tech_cost_factor", 1.0),
        )
        state.turn = self.turns
        state.actions_this_turn = 0
        state.max_actions_per_turn = max(1, self.max_units * 2)
        state.acted_unit_slots = {1: set(), -1: set()}
        state.acted_production_cities = {1: set(), -1: set()}
        state.visited = {
            1: snapshot.visited.copy(),
            -1: snapshot.visited.copy(),
        }
        state.units_built = {1: 0, -1: 0}
        state.future_techs = {1: 0, -1: 0}
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
        state.PRODUCTION_ITEM_COUNT = len(PRODUCTION_ITEM_NAMES)
        state.PRODUCTION_UNIT_COUNT = len(PRODUCTION_UNIT_NAMES)
        state.ECON_PASS_OFFSET = (
            state.ECON_PRODUCTION_OFFSET + self.max_cities * state.PRODUCTION_ITEM_COUNT
        )
        state.ECON_SIZE = state.ECON_PASS_OFFSET + 1
        state.ACTION_SIZE = state.MOVE_SIZE + state.ATTACK_SIZE + state.ECON_SIZE
        state.PASS_ACTION = state.ACTION_SIZE - 1

        self.unit_slots = []
        self.unit_positions = []
        self.unit_slot_types = []
        can_build_flags = []
        for uid, ux, uy, unit_type, unit_hp in unit_entries[: self.max_units]:
            unit_key = (unit_type or "").strip().lower()
            unit_name = unit_names.get(unit_key)
            if unit_name is None and unit_key.endswith("s"):
                unit_name = unit_names.get(unit_key[:-1])
            if unit_name is None and ("settler" in unit_key or "migrant" in unit_key):
                unit_name = "Settlers"
            spec = UNIT_SPECS.get(unit_name or "") or UNIT_SPECS.get("Warriors")
            if spec is None:
                hp_val = unit_hp if unit_hp is not None and unit_hp > 0 else 10
                unit = MHUnit(
                    ux, uy, hp_val, 2, 1, 1, unit_name or "Warriors", True, False, None
                )
                slot_label = (unit_name or unit_type or "Warriors").strip()
            else:
                hp_val = unit_hp if unit_hp is not None and unit_hp > 0 else spec.hp
                unit = MHUnit(
                    ux,
                    uy,
                    hp_val,
                    spec.atk,
                    spec.df,
                    spec.firepower,
                    spec.name,
                    True,
                    spec.can_build_city or ("settler" in unit_key),
                    None,
                )
                slot_label = spec.name
            can_build_flags.append(unit.can_build_city)
            state.units[1].append(unit)
            self.unit_slots.append(uid)
            self.unit_positions.append((ux, uy))
            self.unit_slot_types.append(slot_label)
        while len(state.units[1]) < self.max_units:
            state.units[1].append(MHUnit(0, 0, 0, 0, 0, 1, "None", False, False, None))
            self.unit_slots.append(None)
            self.unit_positions.append(None)
            self.unit_slot_types.append("")

        enemy_units = []
        if status_by_id:
            for uid, (ux, uy, owner, unit_type, hp, _moves) in status_by_id.items():
                if owner < 0:
                    continue
                if self.player_id is not None and owner == self.player_id:
                    continue
                enemy_units.append((uid, ux, uy, unit_type, hp))
            enemy_units.sort(key=lambda entry: entry[0])
        if enemy_units:
            for _uid, ex, ey, unit_type, unit_hp in enemy_units[: self.max_units]:
                unit_key = (unit_type or "").strip().lower()
                unit_name = unit_names.get(unit_key)
                if unit_name is None and unit_key.endswith("s"):
                    unit_name = unit_names.get(unit_key[:-1])
                if unit_name is None and ("settler" in unit_key or "migrant" in unit_key):
                    unit_name = "Settlers"
                spec = UNIT_SPECS.get(unit_name or "") or UNIT_SPECS.get("Warriors")
                if spec is None:
                    hp_val = unit_hp if unit_hp is not None and unit_hp > 0 else 10
                    enemy = MHUnit(
                        ex, ey, hp_val, 2, 1, 1, unit_name or "Warriors", True, False, None
                    )
                else:
                    hp_val = unit_hp if unit_hp is not None and unit_hp > 0 else spec.hp
                    enemy = MHUnit(
                        ex,
                        ey,
                        hp_val,
                        spec.atk,
                        spec.df,
                        spec.firepower,
                        spec.name,
                        True,
                        False,
                        None,
                    )
                state.units[-1].append(enemy)
        else:
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

        try:
            city_walls = alpha_live.list_city_walls(self.client)
        except Exception:
            city_walls = {}
        try:
            self.city_sizes = list_city_sizes(self.client)
        except Exception:
            self.city_sizes = {}

        self.city_slots = []
        for city_id, cx, cy in owned_cities[: self.max_cities]:
            state.cities[1].append(
                City(
                    x=int(cx),
                    y=int(cy),
                    size=int(self.city_sizes.get(int(city_id), 1)),
                    has_city_walls=bool(city_walls.get(city_id, False)),
                )
            )
            self.city_slots.append(city_id)
        for city_id, cx, cy in enemy_cities[: self.max_cities]:
            state.cities[-1].append(
                City(
                    x=int(cx),
                    y=int(cy),
                    size=int(self.city_sizes.get(int(city_id), 1)),
                    has_city_walls=bool(city_walls.get(city_id, False)),
                )
            )

        return state

    def _unit_slot_label(self, slot_idx: int) -> str:
        if 0 <= slot_idx < len(self.unit_slot_types):
            return (self.unit_slot_types[slot_idx] or "").lower()
        return ""

    def _unit_is_worker_like(self, label: str) -> bool:
        return any(tag in label for tag in ("worker", "engineer", "migrant", "settler"))

    def _unit_is_autosettler_candidate(self, label: str) -> bool:
        return any(tag in label for tag in ("worker", "engineer", "migrant"))

    def _production_is_worker_like(self, name: str) -> bool:
        label = (name or "").lower()
        return any(tag in label for tag in ("worker", "engineer", "migrant"))

    def _production_is_excluded(self, name: str) -> bool:
        label = (name or "").lower()
        return any(tag in label for tag in ("diplomat", "explorer"))

    def _worker_unit_count(self) -> int:
        for label in self.unit_slot_types:
            if self._production_is_worker_like(label or ""):
                return 1
        for city_counts in self._city_unit_counts.values():
            for name, units in city_counts.items():
                if self._production_is_worker_like(name):
                    return max(1, units)
        return 0

    def _unit_is_obsolete_exempt(self, name: str) -> bool:
        label = (name or "").lower()
        return any(tag in label for tag in ("settler", "worker", "engineer", "migrant"))

    def _unit_strength_score(self, spec: UnitSpec) -> int:
        return (
            spec.atk * 100
            + spec.df * 90
            + spec.hp * 10
            + spec.firepower * 50
            + spec.moves * 20
        )

    def _unit_best_upgrade(self, name: str) -> str:
        candidates: list[str] = []
        for chain in UNIT_GENERATION_CHAINS:
            if name not in chain:
                continue
            best = None
            for unit_name in chain:
                if unit_name in UNIT_SPECS and self._unit_unlocked(unit_name):
                    best = unit_name
            if best:
                candidates.append(best)
        if not candidates:
            return name

        def score(unit_name: str) -> int:
            spec = UNIT_SPECS.get(unit_name)
            if spec is None:
                return -1
            return self._unit_strength_score(spec)

        return max(candidates, key=score)

    def _unit_obsolete(self, name: str) -> bool:
        if self._unit_is_obsolete_exempt(name):
            return False
        if self._unit_best_upgrade(name) != name:
            return True
        obsolete_by = UNIT_OBSOLETE_BY.get(name)
        if not obsolete_by:
            return False
        return self._unit_unlocked(obsolete_by)

    def _upgrade_unit_name(self, name: str) -> str:
        upgraded = self._unit_best_upgrade(name)
        if upgraded != name:
            return upgraded
        seen = {name}
        while True:
            obsolete_by = UNIT_OBSOLETE_BY.get(name)
            if not obsolete_by or obsolete_by in seen:
                break
            if not self._unit_unlocked(obsolete_by):
                break
            name = obsolete_by
            seen.add(name)
        return name

    def _enemy_threats(self) -> set[tuple[int, int]]:
        threats = set(self.visible_enemy_unit_coords or [])
        if self.visible_enemy_units:
            threats.update(self.visible_enemy_units)
        return threats

    def _is_threat_adjacent(self, pos: tuple[int, int], threats: set[tuple[int, int]]) -> bool:
        if not threats or self.movement is None:
            return False
        ux, uy = pos
        for nx, ny in self.movement.get_native_neighbors(ux, uy):
            if nx is None or ny is None:
                continue
            if (int(nx), int(ny)) in threats:
                return True
        return False

    def _unit_unlocked(self, unit_name: str) -> bool:
        if self._buildable_units:
            return unit_name in self._buildable_units
        if self._last_state is None:
            return False
        try:
            return self._last_state._unit_unlocked(1, unit_name)
        except Exception:
            techs = UNIT_TECHS.get(unit_name, [])
            if not techs:
                return True
            return all(
                self._last_state.research_done.get(1, {}).get(tech, False)
                for tech in techs
            )

    def _player_can_build_unit(self, unit_name: str) -> bool:
        if self.client is None or self.player_id is None:
            return False
        safe_name = (unit_name or "").replace("\\", "\\\\").replace("'", "\\'")
        lua = (
            "return (function() "
            f"local pl = find.player and find.player({self.player_id}); "
            f"local ut = find.unit_type and find.unit_type('{safe_name}'); "
            "if pl and ut and pl.can_build_direct then "
            "  local ok,res = pcall(function() return pl:can_build_direct(ut) end); "
            "  if ok and res then return '__YES__' end "
            "end "
            "return '__NO__' "
            "end)()"
        )
        try:
            res = self.client.eval(lua)
            ret = res.last_return() if res else None
            return isinstance(ret, str) and "__YES__" in ret
        except Exception:
            return False

    def _refresh_buildable_units(self) -> None:
        if self.client is None or self.player_id is None:
            self._buildable_units = set()
            return
        buildable = set()
        for name in PRODUCTION_UNIT_NAMES:
            if self._production_is_excluded(name):
                continue
            if self._player_can_build_unit(name):
                buildable.add(name)
        self._buildable_units = buildable

    def _player_can_build_building(self, building_name: str) -> bool:
        if self.client is None or self.player_id is None:
            return False
        safe_name = (building_name or "").replace("\\", "\\\\").replace("'", "\\'")
        lua = (
            "return (function() "
            f"local pl = find.player and find.player({self.player_id}); "
            f"local bt = find.building_type and find.building_type('{safe_name}'); "
            "if pl and bt and pl.can_build_direct then "
            "  local ok,res = pcall(function() return pl:can_build_direct(bt) end); "
            "  if ok and res then return '__YES__' end "
            "end "
            "return '__NO__' "
            "end)()"
        )
        try:
            res = self.client.eval(lua)
            ret = res.last_return() if res else None
            return isinstance(ret, str) and "__YES__" in ret
        except Exception:
            return False

    def _refresh_buildable_buildings(self) -> None:
        if self.client is None or self.player_id is None:
            self._buildable_buildings = set()
            return
        buildable = set()
        for name in PRODUCTION_BUILDING_NAMES:
            if self._player_can_build_building(name):
                buildable.add(name)
        self._buildable_buildings = buildable

    def _log_production_options(
        self,
        *,
        reason: str,
        valid: Optional[numpy.ndarray] = None,
    ) -> None:
        if self._last_state is None or not self.city_slots:
            return
        econ_offset = self._last_state.MOVE_SIZE + self._last_state.ATTACK_SIZE
        prod_start = econ_offset + self._last_state.ECON_PRODUCTION_OFFSET
        for city_slot, city_id in enumerate(self.city_slots):
            if city_slot >= len(self._last_state.cities[1]):
                continue
            city = self._last_state.cities[1][city_slot]
            worker_count = self._worker_unit_count()
            buildable_units = []
            for name in PRODUCTION_UNIT_NAMES:
                if self._production_is_excluded(name):
                    continue
                if self._buildable_units:
                    if name not in self._buildable_units:
                        continue
                elif not self._unit_unlocked(name):
                    continue
                if worker_count > 0 and self._production_is_worker_like(name):
                    continue
                if name == "Settlers" and city.size < 3:
                    continue
                buildable_units.append(name)
            buildable_buildings = []
            for name in PRODUCTION_BUILDING_NAMES:
                if self._buildable_buildings:
                    if name not in self._buildable_buildings:
                        continue
                if name in self._city_buildings.get(city_id, set()):
                    continue
                if not self._building_allowed_by_water(city_id, name):
                    continue
                if not self._building_allowed_by_requirements(city_id, name):
                    continue
                buildable_buildings.append(name)
            legal_units = []
            legal_buildings = []
            if valid is not None:
                start = prod_start + city_slot * self._last_state.PRODUCTION_ITEM_COUNT
                end = start + self._last_state.PRODUCTION_ITEM_COUNT
                for item_idx, (kind, name) in enumerate(PRODUCTION_ITEM_NAMES):
                    action_idx = start + item_idx
                    if action_idx >= len(valid) or not valid[action_idx]:
                        continue
                    if kind == "unit":
                        legal_units.append(name)
                    else:
                        legal_buildings.append(name)
            print(
                "[production-options] "
                f"reason={reason} turn={self.turns} city={city_id} size={city.size} "
                f"units={','.join(buildable_units)} buildings={','.join(buildable_buildings)}",
                file=sys.stderr,
            )
            if valid is not None:
                print(
                    "[production-legal] "
                    f"reason={reason} turn={self.turns} city={city_id} "
                    f"units={','.join(legal_units)} buildings={','.join(legal_buildings)}",
                    file=sys.stderr,
                )

    def _select_production_unit(self) -> Optional[str]:
        if self._last_state is None:
            return None
        settler_count = sum(
            1 for name in self.unit_slot_types if "settler" in (name or "").lower()
        )
        city_sizes = [city.size for city in self._last_state.cities[1]]
        can_make_settler = any(size >= 3 for size in city_sizes)
        missing_cities = max(self.max_cities - len(self.city_slots), 0)
        if (
            can_make_settler
            and missing_cities > 0
            and settler_count < missing_cities
            and self._unit_unlocked("Settlers")
        ):
            return "Settlers"
        candidates = []
        for name in PRODUCTION_UNIT_NAMES:
            if not self._unit_unlocked(name):
                continue
            if self._unit_obsolete(name):
                continue
            if self._production_is_excluded(name):
                continue
            if name == "Settlers" or self._production_is_worker_like(name):
                continue
            spec = UNIT_SPECS.get(name)
            if spec is None:
                continue
            if spec.atk <= 0 and spec.df <= 0:
                continue
            score = spec.atk * 2.0 + spec.df * 1.5 + spec.hp * 0.1 + spec.moves * 0.5
            candidates.append((score, name))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        if self._unit_unlocked("Warriors"):
            return "Warriors"
        return None

    def _city_exists(self, city_id: int) -> bool:
        try:
            cities = alpha_live.list_all_cities(self.client)
        except Exception:
            return True
        for cid, _cx, _cy, _owner, _name in cities:
            if int(cid) == int(city_id):
                return True
        return False

    def _city_spacing_ok(self, pos: tuple[int, int]) -> bool:
        if self._last_state is None or self.movement is None:
            return True
        min_dist = getattr(self.config, "city_min_distance", 0)
        if min_dist <= 0:
            return True
        if not self._last_state.cities[1]:
            return True
        x, y = pos
        frontier = deque()
        frontier.append((x, y, 0))
        seen = {(x, y)}
        while frontier:
            cx, cy, dist = frontier.popleft()
            if dist >= min_dist:
                continue
            for city in self._last_state.cities[1]:
                if city.x == cx and city.y == cy:
                    return False
            for nx, ny in self.movement.get_native_neighbors(cx, cy):
                if nx is None or ny is None:
                    continue
                if (nx, ny) in seen:
                    continue
                if self._last_state.gt and self._last_state.gt.au_map[ny, nx] != "A":
                    continue
                seen.add((nx, ny))
                frontier.append((nx, ny, dist + 1))
        return True

    def _set_city_production(self, city_id: int, kind: str, name: str) -> bool:
        if self.client is None:
            return False
        prod_kind = "UnitType" if kind == "unit" else "Building"
        try:
            return bool(
                self.client.queue_city_production(city_id, prod_kind, name, 0)
            )
        except Exception:
            return False

    def _city_water_access(self, city_id: int) -> tuple[bool, bool]:
        info = self._city_adjacent_water.get(city_id)
        if not info:
            return False, False
        return bool(info.get("river", False)), bool(info.get("lake", False))

    def _building_allowed_by_water(self, city_id: int, name: str) -> bool:
        if not name:
            return True
        lowered = name.lower()
        if not lowered.startswith("aqueduct"):
            return True
        has_river, has_lake = self._city_water_access(city_id)
        if "river" in lowered:
            return has_river
        if "lake" in lowered:
            return has_lake
        return not has_river and not has_lake

    def _building_allowed_by_requirements(self, city_id: int, name: str) -> bool:
        if not name:
            return True
        lowered = name.lower()
        if "palace" in lowered:
            return False
        allowlist = getattr(self.config, "wonder_production_allowlist", ())
        blocklist = getattr(self.config, "wonder_production_blocklist", ())
        if name in GREAT_WONDER_NAMES:
            if allowlist:
                return name in allowlist
            if name in blocklist:
                return False
        if self._last_state is None:
            return True
        tech_flags = self._last_state.research_done.get(1, {})
        techs = BUILDING_TECHS.get(name, [])
        if techs and not all(tech_flags.get(tech, False) for tech in techs):
            return False
        req_buildings = BUILDING_REQ_BUILDINGS.get(name, [])
        if req_buildings:
            built = self._city_buildings.get(city_id, set())
            if not all(req in built for req in req_buildings):
                return False
        return True

    def _queue_city_production(self, city_id: int, kind: str, name: str, count: int = 1) -> int:
        queue = self.production_queue.setdefault(city_id, [])
        if self.max_production_queue > 0 and len(queue) >= self.max_production_queue:
            return 0
        if kind == "unit":
            name = self._upgrade_unit_name(name)
            if self._production_is_excluded(name):
                return 0
            if self._production_is_worker_like(name) and self._worker_unit_count() > 0:
                return 0
            if name == "Settlers":
                size = int(self.city_sizes.get(int(city_id), 1))
                if size < 3:
                    return 0
        if kind == "building":
            if not self._building_allowed_by_water(city_id, name):
                return 0
            if not self._building_allowed_by_requirements(city_id, name):
                return 0
            if name in self._city_buildings.get(city_id, set()):
                return 0
            queued_buildings = {
                queued_name
                for queued_kind, queued_name in queue
                if queued_kind == "building"
            }
            current_prod = self.production_current.get(city_id)
            if current_prod and current_prod[0] == "building":
                queued_buildings.add(current_prod[1])
            if name in queued_buildings:
                return 0
        added = 0
        to_add = max(1, int(count))
        if kind == "building":
            to_add = 1
        while added < to_add:
            if self.max_production_queue > 0 and len(queue) >= self.max_production_queue:
                break
            if self.client is not None:
                prod_kind = "UnitType" if kind == "unit" else "Building"
                position = 0 if self.production_current.get(city_id) is None and not queue else -1
                try:
                    ok = bool(
                        self.client.queue_city_production(city_id, prod_kind, name, position)
                    )
                except Exception:
                    ok = False
                if not ok:
                    break
            queue.append((kind, name))
            added += 1
        if added and self.production_current.get(city_id) is None and queue:
            self.production_current[city_id] = queue[0]
        return added

    def _production_completed(
        self,
        city_id: int,
        item: tuple[str, str],
        prev_buildings: dict[int, set[str]],
        prev_units: dict[int, dict[str, int]],
        buildings: dict[int, set[str]],
        units: dict[int, dict[str, int]],
    ) -> bool:
        kind, name = item
        if kind == "building":
            return (
                name in buildings.get(city_id, set())
                and name not in prev_buildings.get(city_id, set())
            )
        if kind == "unit":
            prev = prev_units.get(city_id, {}).get(name, 0)
            now = units.get(city_id, {}).get(name, 0)
            return now > prev
        return False

    def _refresh_production_queues(self) -> None:
        if self.client is None:
            return
        try:
            buildings = list_city_buildings(self.client, PRODUCTION_BUILDING_NAMES)
        except Exception:
            buildings = {}
        try:
            units = list_units_by_homecity(self.client)
        except Exception:
            units = {}
        prev_buildings = self._city_buildings
        prev_units = self._city_unit_counts
        self._city_buildings = buildings
        self._city_unit_counts = units
        for city_id in list(self.production_queue.keys()):
            if city_id not in self.city_slots:
                self.production_queue.pop(city_id, None)
                self.production_current.pop(city_id, None)
                continue
            queue = self.production_queue.get(city_id)
            if not queue:
                continue
            current = self.production_current.get(city_id)
            if current is None and queue:
                current = queue[0]
                if self._set_city_production(city_id, *current):
                    self.production_current[city_id] = current
            if current and self._production_completed(
                city_id,
                current,
                prev_buildings,
                prev_units,
                buildings,
                units,
            ):
                kind, name = current
                print(
                    f"[production] city={city_id} completed {kind} {name}",
                    file=sys.stderr,
                )
                if queue and queue[0] == current:
                    queue.pop(0)
                else:
                    try:
                        queue.remove(current)
                    except ValueError:
                        pass
                self.production_current[city_id] = None
            if queue and self.production_current.get(city_id) is None:
                next_item = queue[0]
                if self._set_city_production(city_id, *next_item):
                    self.production_current[city_id] = next_item
                    print(
                        f"[production] city={city_id} queued {next_item[0]} {next_item[1]}",
                        file=sys.stderr,
                    )

    def _refresh_city_adjacent_water(self) -> None:
        if self.client is None:
            return
        try:
            self._city_adjacent_water = list_city_adjacent_water(self.client)
        except Exception:
            self._city_adjacent_water = {}

    def _refresh_queue_on_research_change(self, research_flags: dict[str, bool]) -> None:
        if not self._buildable_units and not self._buildable_buildings:
            self._refresh_buildable_units()
            self._refresh_buildable_buildings()
        if research_flags == self._last_research_flags:
            return
        prev_flags = dict(self._last_research_flags)
        self._last_research_flags = dict(research_flags)
        self._refresh_buildable_units()
        self._refresh_buildable_buildings()
        completed = [
            tech
            for tech, done in research_flags.items()
            if done and not prev_flags.get(tech, False)
        ]
        if completed:
            print(
                f"[research] completed={','.join(completed)}",
                file=sys.stderr,
            )
            self._log_production_options(reason="research")
        for city_id, queue in self.production_queue.items():
            if not queue and self.production_current.get(city_id) is None:
                continue
            updated: list[tuple[str, str]] = []
            changed = False
            for idx, (kind, name) in enumerate(queue):
                if idx == 0:
                    updated.append((kind, name))
                    continue
                if kind == "unit":
                    upgraded = self._upgrade_unit_name(name)
                    if upgraded != name:
                        name = upgraded
                        changed = True
                updated.append((kind, name))
            if changed:
                self.production_queue[city_id] = updated
            current = self.production_current.get(city_id)
            if current is None and self.production_queue.get(city_id):
                current = self.production_queue[city_id][0]
            if not current:
                continue
            kind, name = current
            if kind != "unit":
                continue
            upgraded = self._upgrade_unit_name(name)
            if upgraded == name:
                continue
            existing_units = {
                queued_name
                for queued_kind, queued_name in self.production_queue.get(city_id, [])
                if queued_kind == "unit"
            }
            if upgraded in existing_units:
                continue
            if self.max_production_queue > 0:
                if len(self.production_queue.get(city_id, [])) >= self.max_production_queue:
                    continue
            added = self._queue_city_production(city_id, "unit", upgraded, count=1)
            if added:
                print(
                    f"[production] city={city_id} queued unit {upgraded} after research",
                    file=sys.stderr,
                )

    def _prefer_production_actions(self, valid):
        if self._last_state is None or not self.city_slots:
            return valid
        econ_offset = self._last_state.MOVE_SIZE + self._last_state.ATTACK_SIZE
        prod_start = econ_offset + self._last_state.ECON_PRODUCTION_OFFSET
        prod_end = econ_offset + self._last_state.ECON_PASS_OFFSET
        if prod_start >= len(valid):
            return valid
        desired_unit = self._select_production_unit()
        if not desired_unit:
            return valid
        desired_idx = None
        for idx, (kind, name) in enumerate(PRODUCTION_ITEM_NAMES):
            if kind == "unit" and name == desired_unit:
                desired_idx = idx
                break
        if desired_idx is None:
            return valid
        for slot_idx in range(len(self.city_slots)):
            if desired_unit == "Settlers":
                if slot_idx >= len(self._last_state.cities[1]):
                    continue
                if self._last_state.cities[1][slot_idx].size < 3:
                    continue
            start = prod_start + slot_idx * self._last_state.PRODUCTION_ITEM_COUNT
            end = start + self._last_state.PRODUCTION_ITEM_COUNT
            if start >= len(valid):
                break
            end = min(end, len(valid), prod_end)
            if not valid[start:end].any():
                continue
            desired_action = start + desired_idx
            if desired_action < end and valid[desired_action]:
                for action_idx in range(start, end):
                    if action_idx != desired_action:
                        valid[action_idx] = 0
        return valid

    def _force_production_actions(self, valid, raw_valid):
        return valid

    def _deprioritize_worker_production(self, valid):
        if self._last_state is None or not self.city_slots:
            return valid
        worker_count = 0
        for city_counts in self._city_unit_counts.values():
            for name, count in city_counts.items():
                if self._production_is_worker_like(name):
                    worker_count += count
        if worker_count <= 0:
            return valid
        econ_offset = self._last_state.MOVE_SIZE + self._last_state.ATTACK_SIZE
        prod_start = econ_offset + self._last_state.ECON_PRODUCTION_OFFSET
        prod_end = econ_offset + self._last_state.ECON_PASS_OFFSET
        if prod_start >= len(valid):
            return valid
        for slot_idx in range(len(self.city_slots)):
            start = prod_start + slot_idx * self._last_state.PRODUCTION_ITEM_COUNT
            end = start + self._last_state.PRODUCTION_ITEM_COUNT
            if start >= len(valid):
                break
            end = min(end, len(valid), prod_end)
            if not valid[start:end].any():
                continue
            worker_actions = []
            other_actions = []
            for item_idx in range(end - start):
                action_idx = start + item_idx
                if action_idx >= len(valid) or not valid[action_idx]:
                    continue
                kind, name = PRODUCTION_ITEM_NAMES[item_idx]
                if kind == "unit" and self._production_is_worker_like(name):
                    worker_actions.append(action_idx)
                else:
                    other_actions.append(action_idx)
            if other_actions and worker_actions:
                for action_idx in worker_actions:
                    valid[action_idx] = 0
        return valid

    def _prefer_research_actions(self, valid):
        if self._last_state is None:
            return valid
        if not self.city_slots or self.current_research:
            return valid
        flags = self._last_state.research_done.get(1, {})
        tech_name = pick_next_priority_tech(flags, TECH_PREREQS, self._last_state.RESEARCH_TECHS)
        if not tech_name:
            return valid
        try:
            tech_idx = self._last_state.RESEARCH_TECHS.index(tech_name)
        except ValueError:
            return valid
        econ_offset = self._last_state.MOVE_SIZE + self._last_state.ATTACK_SIZE
        research_end = econ_offset + self._last_state.ECON_BUILD_CITY_OFFSET
        if econ_offset >= len(valid):
            return valid
        action_idx = econ_offset + tech_idx
        if 0 <= action_idx < len(valid) and valid[action_idx]:
            valid[econ_offset:research_end] = 0
            valid[action_idx] = 1
        return valid

    def _maybe_switch_government(self, research_flags: dict[str, bool]) -> None:
        if not research_flags:
            return
        target = None
        if research_flags.get("Democracy", False) and "Democracy" not in self.gov_switch_sent:
            target = "Democracy"
        elif research_flags.get("Monarchy", False) and "Monarchy" not in self.gov_switch_sent:
            target = "Monarchy"
        if target is None:
            return
        try:
            ok = lua_set_government(self.client, target)
        except Exception:
            ok = False
        if ok:
            self.gov_switch_sent.add(target)

    def _maybe_enable_autosettlers(self) -> None:
        if not self.unit_slots:
            return
        threats = self._enemy_threats()
        for slot_idx, unit_id in enumerate(self.unit_slots):
            if unit_id is None:
                continue
            label = self._unit_slot_label(slot_idx)
            if not self._unit_is_autosettler_candidate(label):
                continue
            if unit_id in self.autosettler_units:
                continue
            pos = self.unit_positions[slot_idx] if slot_idx < len(self.unit_positions) else None
            if pos is not None and self._is_threat_adjacent(pos, threats):
                continue
            try:
                ok = lua_auto_settler(self.client, unit_id)
            except Exception:
                ok = False
            if ok:
                self.autosettler_units.add(unit_id)

    def _mask_autosettler_actions(self, valid):
        if self._last_state is None or not self.autosettler_units:
            return valid
        threats = self._enemy_threats()
        econ_offset = self._last_state.MOVE_SIZE + self._last_state.ATTACK_SIZE
        for slot_idx, unit_id in enumerate(self.unit_slots):
            if unit_id is None or unit_id not in self.autosettler_units:
                continue
            pos = self.unit_positions[slot_idx] if slot_idx < len(self.unit_positions) else None
            if pos is not None and self._is_threat_adjacent(pos, threats):
                continue
            move_start = slot_idx * self._last_state.MOVE_PER_UNIT
            move_end = move_start + self._last_state.MOVE_PER_UNIT
            atk_start = self._last_state.MOVE_SIZE + slot_idx * self._last_state.ATTACK_PER_UNIT
            atk_end = atk_start + self._last_state.ATTACK_PER_UNIT
            valid[move_start:move_end] = 0
            valid[atk_start:atk_end] = 0
            build_idx = econ_offset + self._last_state.ECON_BUILD_CITY_OFFSET + slot_idx
            if 0 <= build_idx < len(valid):
                valid[build_idx] = 0
        return valid

    def _city_site_ok(self, x: int, y: int, player: int) -> bool:
        if self._last_state is None or self._last_state.gt is None:
            return False
        if self._last_state.gt.au_map[y, x] != "A":
            return False
        if self._tile_owner_refresh_pending:
            self._refresh_tile_owners()
        if player == 1:
            owner = self.tile_owners.get((x, y))
            if owner is not None and self.player_id is not None and owner != self.player_id:
                return False
        if self._last_state._city_at(x, y, player) is not None:
            return False
        if self._last_state._city_at(x, y, -player) is not None:
            return False
        for p in (1, -1):
            for u in self._last_state.units[p]:
                if u.alive and u.x == x and u.y == y:
                    return False
        if self.movement is not None:
            for nx, ny in self.movement.get_native_neighbors(x, y):
                if nx is None or ny is None:
                    continue
                for u in self._last_state.units[-player]:
                    if u.alive and u.x == nx and u.y == ny:
                        return False
        return self._last_state._city_spacing_ok(player, x, y)

    def _settler_first_step(self, start: tuple[int, int], player: int):
        if self._last_state is None or self.movement is None or self._last_state.gt is None:
            return None
        sx, sy = start
        neighbors = self.movement.get_native_neighbors(sx, sy)
        seen = {(sx, sy)}
        queue = deque()
        for dir_idx, (nx, ny) in enumerate(neighbors):
            if nx is None or ny is None:
                continue
            if self._last_state.gt.au_map[ny, nx] != "A":
                continue
            seen.add((nx, ny))
            queue.append((nx, ny, dir_idx, 1))
        while queue:
            cx, cy, first_dir, dist = queue.popleft()
            if self._city_site_ok(cx, cy, player):
                return first_dir, dist
            for nx, ny in self.movement.get_native_neighbors(cx, cy):
                if nx is None or ny is None:
                    continue
                if (nx, ny) in seen:
                    continue
                if self._last_state.gt.au_map[ny, nx] != "A":
                    continue
                seen.add((nx, ny))
                queue.append((nx, ny, first_dir, dist + 1))
        return None

    def _force_settler_city_actions(self, valid):
        if self._last_state is None:
            return None
        player = 1
        if len(self._last_state.cities[player]) >= self._last_state.max_cities:
            return None
        econ_offset = self._last_state.MOVE_SIZE + self._last_state.ATTACK_SIZE
        build_base = econ_offset + self._last_state.ECON_BUILD_CITY_OFFSET
        build_actions = []
        best_move = None
        for idx, u in enumerate(self._last_state.units[player]):
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
            action = idx * self._last_state.MOVE_PER_UNIT + dir_idx
            if 0 <= action < len(valid) and valid[action]:
                if best_move is None or dist < best_move[0]:
                    best_move = (dist, action)
        if build_actions:
            return build_actions
        if best_move is not None:
            return [best_move[1]]
        return None

    def _prefer_worker_evasion(self, valid):
        if self._last_state is None or self.movement is None:
            return valid
        threats = self._enemy_threats()
        if not threats:
            return valid
        forced = valid.copy()
        for slot_idx, pos in enumerate(self.unit_positions):
            if pos is None:
                continue
            label = self._unit_slot_label(slot_idx)
            if not self._unit_is_worker_like(label):
                continue
            if not self._is_threat_adjacent(pos, threats):
                continue
            unit_id = self.unit_slots[slot_idx] if slot_idx < len(self.unit_slots) else None
            if unit_id is not None and unit_id in self.autosettler_units:
                self.autosettler_units.discard(unit_id)
            ux, uy = pos
            cur_dist = min(abs(ux - tx) + abs(uy - ty) for (tx, ty) in threats)
            move_start = slot_idx * self._last_state.MOVE_PER_UNIT
            move_end = move_start + self._last_state.MOVE_PER_UNIT
            safe_moves = []
            for dir_idx, (nx, ny) in enumerate(self.movement.get_native_neighbors(ux, uy)):
                if nx is None or ny is None:
                    continue
                action_idx = move_start + dir_idx
                if action_idx >= len(forced) or not forced[action_idx]:
                    continue
                dist = min(abs(nx - tx) + abs(ny - ty) for (tx, ty) in threats)
                if dist > cur_dist:
                    safe_moves.append(action_idx)
            if safe_moves:
                for action_idx in range(move_start, min(move_end, len(forced))):
                    forced[action_idx] = 0
                for action_idx in safe_moves:
                    forced[action_idx] = 1
        return forced

    def _prefer_attack_actions(self, valid):
        if self._last_state is None:
            return valid
        attack_start = self._last_state.MOVE_SIZE
        attack_end = attack_start + self._last_state.ATTACK_SIZE
        if attack_start >= len(valid):
            return valid
        attack_indices = [
            idx
            for idx in range(attack_start, min(attack_end, len(valid)))
            if valid[idx]
        ]
        if not attack_indices:
            return valid
        archer_actions = []
        other_actions = []
        phalanx_actions = []
        for action in attack_indices:
            rel = action - attack_start
            unit_idx = rel // self._last_state.ATTACK_PER_UNIT
            unit_label = self._unit_slot_label(unit_idx)
            if "archer" in unit_label:
                archer_actions.append(action)
            elif "phalanx" in unit_label:
                phalanx_actions.append(action)
            else:
                other_actions.append(action)
        preferred = archer_actions or other_actions or phalanx_actions
        if not preferred:
            return valid
        forced = numpy.zeros_like(valid)
        for idx in preferred:
            forced[idx] = 1
        return forced

    def _sync_state(self):
        if self.client is None:
            self._connect_client()
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
        self._maybe_switch_government(snapshot.research_flags)
        self._try_refresh_controlled_units()
        self._refresh_unit_status()
        self._refresh_player_scores()
        owned_cities = alpha_live.discover_player_cities(self.client, self.player_id)
        enemy_cities = self._visible_enemy_cities()
        self._last_state = self._build_state(snapshot, owned_cities, enemy_cities)
        self._refresh_queue_on_research_change(snapshot.research_flags)
        self._refresh_city_adjacent_water()
        self._tile_owner_refresh_pending = True
        enemy_units = []
        for _uid, ux, uy, owner, _name, _hp, _moves in self.unit_status:
            if owner < 0:
                continue
            if self.player_id is not None and owner == self.player_id:
                continue
            enemy_units.append((int(ux), int(uy)))
        if enemy_units:
            self.visible_enemy_units = enemy_units
        else:
            self.visible_enemy_units = [
                (int(x), int(y)) for (y, x) in numpy.argwhere(snapshot.enemy_map)
            ]
        self.visible_enemy_cities = [(cx, cy) for _cid, cx, cy in enemy_cities]
        self.visible_enemy_city_ids = {
            (int(cx), int(cy)): int(city_id) for city_id, cx, cy in enemy_cities
        }
        self.visible_enemy_unit_coords = {
            (int(x), int(y))
            for (x, y), status in snapshot.status_lookup.items()
            if status and len(status) >= 3 and status[2]
        }
        if self.visible_enemy_units:
            self.visible_enemy_unit_coords.update(self.visible_enemy_units)
        self.autosettler_units.intersection_update(set(self.unit_slots))
        self._maybe_enable_autosettlers()
        self._refresh_production_queues()
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
            target = (int(nx), int(ny))
            city_id = self.visible_enemy_city_ids.get(target)
            if city_id is not None:
                self.client.conquer_city(unit_id, city_id)
                if not self._city_exists(city_id):
                    return
            if dir_idx < len(self.dir_ids):
                dir_id = self.dir_ids[dir_idx]
                if self.client.attack_dir_id(unit_id, dir_id):
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
            unit_pos = self.unit_positions[unit_idx]
            if unit_id is None or unit_pos is None:
                return
            if not self._city_spacing_ok(unit_pos):
                return
            city_name = f"MuZeroCity{len(owned_cities) + 1}"
            built = self.client.found_city(unit_id, city_name)
            if not built:
                self.client.build_city(unit_id)
            return
        if board_state.ECON_PRODUCTION_OFFSET <= econ_idx < board_state.ECON_PASS_OFFSET:
            rel = econ_idx - board_state.ECON_PRODUCTION_OFFSET
            city_slot = rel // board_state.PRODUCTION_ITEM_COUNT
            item_idx = rel % board_state.PRODUCTION_ITEM_COUNT
            if city_slot >= len(self.city_slots):
                return
            if item_idx >= len(PRODUCTION_ITEM_NAMES):
                return
            city_id = self.city_slots[city_slot]
            kind, name = PRODUCTION_ITEM_NAMES[item_idx]
            queued = self._queue_city_production(
                city_id,
                kind,
                name,
                count=self.production_queue_add,
            )
            if queued:
                print(
                    f"[production] city={city_id} append {kind} {name} x{queued}",
                    file=sys.stderr,
                )
            self.acted_production_cities.add(city_id)
            return

    def step(self, action):
        prev_visited = len(self.visited_tiles)
        try:
            board_state = self._sync_state()
        except Exception:
            if self.restart_on_reset and self.client_cmd:
                self._shutdown_client_for_restart()
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
        if action < board_state.MOVE_SIZE:
            self.acted_unit_slots.add(action // board_state.MOVE_PER_UNIT)
        elif action < board_state.MOVE_SIZE + board_state.ATTACK_SIZE:
            rel = action - board_state.MOVE_SIZE
            self.acted_unit_slots.add(rel // board_state.ATTACK_PER_UNIT)
        else:
            econ_idx = action - (board_state.MOVE_SIZE + board_state.ATTACK_SIZE)
            if board_state.ECON_BUILD_CITY_OFFSET <= econ_idx < board_state.ECON_PRODUCTION_OFFSET:
                self.acted_unit_slots.add(econ_idx - board_state.ECON_BUILD_CITY_OFFSET)
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
            self.acted_unit_slots.clear()
            self.acted_production_cities.clear()
            if self.sleep:
                time.sleep(self.sleep)

        done = self.turns >= self.config.max_turns
        try:
            board_state = self._sync_state()
        except Exception:
            done = True
            board_state = self._last_state
        if done and self.restart_on_reset and self.client_cmd:
            self._shutdown_client_for_restart()

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
        if self.city_slots:
            econ_offset = self._last_state.MOVE_SIZE + self._last_state.ATTACK_SIZE
            prod_start = econ_offset + self._last_state.ECON_PRODUCTION_OFFSET
            min_units = int(getattr(self.config, "city_unit_min", 0))
            free_units = int(getattr(self.config, "city_unit_free", 0))
            for city_slot, city_id in enumerate(self.city_slots):
                if city_id in self.acted_production_cities:
                    start = (
                        prod_start
                        + city_slot * self._last_state.PRODUCTION_ITEM_COUNT
                    )
                    end = start + self._last_state.PRODUCTION_ITEM_COUNT
                    if start < len(valid):
                        valid[start:min(end, len(valid))] = 0
                    continue
                unit_count = sum(
                    self._city_unit_counts.get(city_id, {}).values()
                )
                garrison_count = 0
                queued_buildings = {
                    queued_name
                    for queued_kind, queued_name in self.production_queue.get(city_id, [])
                    if queued_kind == "building"
                }
                current_prod = self.production_current.get(city_id)
                if current_prod and current_prod[0] == "building":
                    queued_buildings.add(current_prod[1])
                if (
                    self._last_state is not None
                    and city_slot < len(self._last_state.cities[1])
                ):
                    city = self._last_state.cities[1][city_slot]
                    tile_count = sum(
                        1
                        for u in self._last_state.units[1]
                        if u.alive and u.x == city.x and u.y == city.y
                    )
                    garrison_count = tile_count
                    if tile_count > unit_count:
                        unit_count = tile_count
                city_size = None
                if (
                    self._last_state is not None
                    and city_slot < len(self._last_state.cities[1])
                ):
                    city_size = self._last_state.cities[1][city_slot].size
                for item_idx, (kind, name) in enumerate(PRODUCTION_ITEM_NAMES):
                    if kind != "unit":
                        continue
                    if self._production_is_excluded(name):
                        action_idx = (
                            prod_start
                            + city_slot * self._last_state.PRODUCTION_ITEM_COUNT
                            + item_idx
                        )
                        if 0 <= action_idx < len(valid):
                            valid[action_idx] = 0
                    elif name == "Settlers" and city_size is not None and city_size < 3:
                        action_idx = (
                            prod_start
                            + city_slot * self._last_state.PRODUCTION_ITEM_COUNT
                            + item_idx
                        )
                        if 0 <= action_idx < len(valid):
                            valid[action_idx] = 0
                for item_idx, (kind, name) in enumerate(PRODUCTION_ITEM_NAMES):
                    if kind != "building":
                        continue
                    if not self._building_allowed_by_water(city_id, name):
                        action_idx = (
                            prod_start
                            + city_slot * self._last_state.PRODUCTION_ITEM_COUNT
                            + item_idx
                        )
                        if 0 <= action_idx < len(valid):
                            valid[action_idx] = 0
                    elif not self._building_allowed_by_requirements(city_id, name):
                        action_idx = (
                            prod_start
                            + city_slot * self._last_state.PRODUCTION_ITEM_COUNT
                            + item_idx
                        )
                        if 0 <= action_idx < len(valid):
                            valid[action_idx] = 0
                    elif name in queued_buildings:
                        action_idx = (
                            prod_start
                            + city_slot * self._last_state.PRODUCTION_ITEM_COUNT
                            + item_idx
                        )
                        if 0 <= action_idx < len(valid):
                            valid[action_idx] = 0
                    elif min_units > 0 and garrison_count < min_units:
                        action_idx = (
                            prod_start
                            + city_slot * self._last_state.PRODUCTION_ITEM_COUNT
                            + item_idx
                        )
                        if 0 <= action_idx < len(valid):
                            valid[action_idx] = 0
                if (
                    free_units > 0
                    and unit_count >= free_units
                    and garrison_count >= min_units
                ):
                    for item_idx, (kind, _name) in enumerate(PRODUCTION_ITEM_NAMES):
                        if kind != "unit":
                            continue
                        if _name == "Settlers" and city_size is not None and city_size >= 3:
                            continue
                        action_idx = (
                            prod_start
                            + city_slot * self._last_state.PRODUCTION_ITEM_COUNT
                            + item_idx
                        )
                        if 0 <= action_idx < len(valid):
                            valid[action_idx] = 0
            if self.max_production_queue > 0:
                prod_end = econ_offset + self._last_state.ECON_PASS_OFFSET
                for city_slot, city_id in enumerate(self.city_slots):
                    queue = self.production_queue.get(city_id, [])
                    if len(queue) < self.max_production_queue:
                        continue
                    start = (
                        prod_start
                        + city_slot * self._last_state.PRODUCTION_ITEM_COUNT
                    )
                    end = start + self._last_state.PRODUCTION_ITEM_COUNT
                    if start >= len(valid):
                        break
                    end = min(end, len(valid), prod_end)
                    valid[start:end] = 0
        valid = self._prefer_research_actions(valid)
        valid = self._deprioritize_worker_production(valid)
        if self.acted_unit_slots:
            econ_offset = self._last_state.MOVE_SIZE + self._last_state.ATTACK_SIZE
            for slot_idx in self.acted_unit_slots:
                if slot_idx < 0 or slot_idx >= self.max_units:
                    continue
                move_start = slot_idx * self._last_state.MOVE_PER_UNIT
                move_end = move_start + self._last_state.MOVE_PER_UNIT
                atk_start = self._last_state.MOVE_SIZE + slot_idx * self._last_state.ATTACK_PER_UNIT
                atk_end = atk_start + self._last_state.ATTACK_PER_UNIT
                valid[move_start:move_end] = 0
                valid[atk_start:atk_end] = 0
                build_idx = econ_offset + self._last_state.ECON_BUILD_CITY_OFFSET + slot_idx
                if 0 <= build_idx < len(valid):
                    valid[build_idx] = 0
        valid = self._prefer_worker_evasion(valid)
        valid = self._mask_autosettler_actions(valid)
        forced = self._force_settler_city_actions(valid)
        if forced:
            return forced
        valid = self._prefer_attack_actions(valid)
        non_pass = valid.copy()
        non_pass[self._last_state.PASS_ACTION] = 0
        if non_pass.any() and alive_units:
            valid[self._last_state.PASS_ACTION] = 0
        return [idx for idx, allowed in enumerate(valid) if allowed]

    def reset(self):
        if self._needs_restart:
            if self.client_cmd:
                self._start_client_process()
            self._needs_restart = False
            self._connect_client()
        self.turns = 0
        self.acted_unit_slots.clear()
        self.acted_production_cities.clear()
        self.production_queue.clear()
        self.production_current.clear()
        self._city_buildings.clear()
        self._city_unit_counts.clear()
        self._city_adjacent_water.clear()
        state = self._sync_state()
        return state.encode(1)

    def close(self):
        try:
            if self.client is not None:
                self.client.close()
        except Exception:
            pass
        self.client = None
        if self.client_cmd:
            self._stop_client_process()

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
            city_slot = rel // tmp_state.PRODUCTION_ITEM_COUNT
            item_idx = rel % tmp_state.PRODUCTION_ITEM_COUNT
            kind, name = PRODUCTION_ITEM_NAMES[item_idx]
            if kind == "unit":
                return f"produce_c{city_slot}_{name}"
            return f"build_c{city_slot}_{name}"
        return str(action_number)
