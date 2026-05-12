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
from freeciv_alpha_zero.freeciv.research_policy import RESEARCH_TECHS
from freeciv_alpha_zero.freeciv.state import FreecivBoardState


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


def _observation_channels() -> int:
    base_channels = 11
    return base_channels + 2 * len(RESEARCH_TECHS)


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0
        self.max_num_gpus = 0

        map_w = _env_int("FREECIV_MAP_W", 4)
        map_h = _env_int("FREECIV_MAP_H", 16)
        max_turns = _env_int("FREECIV_MAX_TURNS", 128)
        self.map_config = MapConfig(map_w=map_w, map_h=map_h, max_turns=max_turns)

        ### Game
        self.observation_shape = (
            _observation_channels(),
            self.map_config.map_h,
            self.map_config.map_w,
        )
        self.action_space = list(range(FreecivBoardState.ACTION_SIZE + 1))
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
        if self.unit_id is not None:
            pos_result = self.client.eval(alpha_live.simple_find_unit_pos(self.unit_id))
            pos_info = alpha_live.parse_position_result(pos_result)
            if pos_info and pos_info[2] is not None and pos_info[2] >= 0:
                self.player_id = int(pos_info[2])
        else:
            controlled, self.player_id = alpha_live.discover_controlled_units(
                self.client, self.player_id
            )
            if not controlled:
                raise RuntimeError(
                    "No controllable units found. Set FREECIV_UNIT_ID or FREECIV_PLAYER_ID."
                )
            self.unit_id = controlled[0]

        self.movement = alpha_live.FreecivMovement(
            map_width=self.config.map_w, map_height=self.config.map_h
        )
        self.known_tiles = {}
        self.known_enemy = {}
        self.visited_tiles = set()
        self.turns = 0
        self._last_snapshot = None
        self._last_state = None

    def _sync_state(self):
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
        self._last_state = alpha_live.build_state(self.config, snapshot)
        return self._last_state

    def _apply_action(self, action, board_state, owned_cities):
        if action == board_state.PASS_ACTION:
            return

        if action == board_state.BUILD_CITY_ACTION:
            city_name = f"MuZeroCity{len(owned_cities) + 1}"
            built = self.client.found_city(self.unit_id, city_name)
            if not built:
                self.client.build_city(self.unit_id)
            return

        if (
            board_state.RESEARCH_ACTION_BASE
            <= action
            < board_state.RESEARCH_ACTION_BASE + board_state.RESEARCH_ACTION_COUNT
        ):
            if not owned_cities:
                return
            tech_idx = action - board_state.RESEARCH_ACTION_BASE
            tech_name = board_state.RESEARCH_TECHS[tech_idx]
            alpha_live.set_research_to_target(
                self.client,
                self.player_id,
                research_flags=self._last_snapshot.research_flags,
                tech_name=tech_name,
            )
            return

        if 0 <= action < len(self.dir_ids):
            dir_id = self.dir_ids[action]
            self.client.move_dir_id(self.unit_id, dir_id)

    def step(self, action):
        prev_visited = len(self.visited_tiles)
        try:
            board_state = self._sync_state()
        except Exception:
            return self.reset(), 0.0, True

        owned_cities = alpha_live.discover_player_cities(self.client, self.player_id)
        self._apply_action(action, board_state, owned_cities)
        try:
            self.client.end_turn()
        except Exception:
            pass
        self.turns += 1
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
        if action_number == FreecivBoardState.PASS_ACTION:
            return "pass"
        if action_number == FreecivBoardState.BUILD_CITY_ACTION:
            return "build_city"
        if action_number < FreecivBoardState.SETTLER_MOVE_COUNT:
            return f"move_{action_number}"
        if (
            FreecivBoardState.RESEARCH_ACTION_BASE
            <= action_number
            < FreecivBoardState.RESEARCH_ACTION_BASE
            + FreecivBoardState.RESEARCH_ACTION_COUNT
        ):
            tech_idx = action_number - FreecivBoardState.RESEARCH_ACTION_BASE
            return f"research_{RESEARCH_TECHS[tech_idx]}"
        return str(action_number)
