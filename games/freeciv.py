import datetime
import pathlib
import sys

import numpy
import torch

from .abstract_game import AbstractGame

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from freeciv_alpha_zero.freeciv.config import MapConfig
from freeciv_alpha_zero.freeciv.providers import RandomMapProvider
from freeciv_alpha_zero.freeciv.research_policy import RESEARCH_TECHS
from freeciv_alpha_zero.freeciv.state import FreecivBoardState


def _observation_channels() -> int:
    base_channels = 11
    return base_channels + 2 * len(RESEARCH_TECHS)


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0
        self.max_num_gpus = None

        self.map_config = MapConfig(map_w=4, map_h=16, max_turns=128)

        ### Game
        self.observation_shape = (
            _observation_channels(),
            self.map_config.map_h,
            self.map_config.map_w,
        )
        self.action_space = list(range(FreecivBoardState.ACTION_SIZE + 1))
        self.players = list(range(2))
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
        self.training_steps = 5000
        self.batch_size = 128
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = torch.cuda.is_available()

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
        self.reanalyse_on_gpu = False

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
    def __init__(self, seed=None):
        self.config = MapConfig(map_w=4, map_h=16, max_turns=128)
        rng = numpy.random.default_rng(seed) if seed is not None else None
        self.provider = RandomMapProvider(
            self.config.map_w,
            self.config.map_h,
            rng=rng,
        )
        self.state = FreecivBoardState(self.config, self.provider)
        self.player = 1

    def _score_diff(self) -> float:
        return float(self.state.scores[1] - self.state.scores[-1])

    def _observation(self):
        return self.state.encode(self.player)

    def step(self, action):
        prev_diff = self._score_diff()
        current_player = self.player
        self.state.step(current_player, action)
        done = self.state.terminal_reason is not None
        new_diff = self._score_diff()
        reward = new_diff - prev_diff
        self.player = -current_player
        return self._observation(), reward, done

    def to_play(self):
        return 0 if self.player == 1 else 1

    def legal_actions(self):
        valid = self.state.valid_moves(self.player)
        return [idx for idx, allowed in enumerate(valid) if allowed]

    def reset(self):
        self.state = FreecivBoardState(self.config, self.provider)
        self.player = 1
        return self._observation()

    def render(self):
        print(self.state.string())
        input("Press enter to take a step ")

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

    def expert_agent(self):
        legal = self.legal_actions()
        return int(numpy.random.choice(legal)) if legal else 0
