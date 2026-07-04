import copy
import importlib
import json
import math
import os
import pathlib
import pickle
import shutil
import sys
import time

import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import drive_sync
import models
import replay_buffer
import self_play
import shared_storage
import trainer


def _csv_items(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _actor_runtime_env_for_gpu(gpu_id):
    if gpu_id is None or str(gpu_id).strip() == "":
        return None
    return {"env_vars": {"CUDA_VISIBLE_DEVICES": str(gpu_id).strip()}}


class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        config (dict, MuZeroConfig, optional): Override the default config of the game.

        split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True)
    """

    def __init__(self, game_name, config=None, split_resources_in=1):
        self.game_name = game_name
        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + game_name)
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        if isinstance(config, dict):
            env_overrides = config.pop("env", None)
            if env_overrides is not None:
                if not isinstance(env_overrides, dict):
                    raise ValueError("config['env'] must be a dict of environment overrides")
                for key, value in env_overrides.items():
                    if value is None:
                        os.environ.pop(str(key), None)
                    else:
                        os.environ[str(key)] = str(value)

        self.Game = game_module.Game
        self.config = game_module.MuZeroConfig()
        results_path = os.getenv("MUZERO_RESULTS_PATH")
        if results_path:
            self.config.results_path = pathlib.Path(results_path).expanduser()

        # Overwrite the config
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    if hasattr(self.config, param):
                        setattr(self.config, param, value)
                    else:
                        raise AttributeError(
                            f"{game_name} config has no attribute '{param}'. Check the config file for the complete list of parameters."
                        )
                if hasattr(self.config, "map_config") and "max_turns" in config:
                    self.config.map_config.max_turns = self.config.max_turns
                if hasattr(self.config, "max_moves") and hasattr(self.config, "max_turns"):
                    if "max_turns" in config and "max_moves" not in config:
                        if hasattr(self.config, "max_actions_per_turn"):
                            self.config.max_moves = (
                                self.config.max_turns * self.config.max_actions_per_turn
                            )
                        else:
                            self.config.max_moves = self.config.max_turns
                    elif "max_moves" in config and "max_turns" not in config:
                        self.config.max_turns = self.config.max_moves
                    if "max_actions_per_turn" in config and "max_moves" not in config:
                        self.config.max_moves = (
                            self.config.max_turns * self.config.max_actions_per_turn
                        )
                if isinstance(getattr(self.config, "results_path", None), str):
                    self.config.results_path = pathlib.Path(
                        self.config.results_path
                    ).expanduser()
            else:
                self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Manage GPUs
        self.train_gpu_id = str(getattr(self.config, "train_gpu_id", "") or "").strip()
        self.selfplay_gpu_ids = _csv_items(
            getattr(self.config, "selfplay_gpu_ids", "")
        )
        self.reanalyse_gpu_id = str(
            getattr(self.config, "reanalyse_gpu_id", "") or ""
        ).strip()
        self.role_gpu_pinning = bool(
            self.train_gpu_id or self.selfplay_gpu_ids or self.reanalyse_gpu_id
        )
        if self.config.max_num_gpus == 0 and (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ) and not self.role_gpu_pinning:
            raise ValueError(
                "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
            )
        if self.role_gpu_pinning:
            total_gpus = 0
        elif (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        ray.init(
            num_gpus=total_gpus,
            ignore_reinit_error=True,
            include_dashboard=False,
            # dashboard_host="0.0.0.0",
            # dashboard_port=8265,
        )

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }
        self.replay_buffer = {}

        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        self._progress_tty = None
        self._progress_tty_disabled = False
        self._progress_tty_rows = None

    def train(self, log_in_tensorboard=True):
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        if os.getenv("MUZERO_DISABLE_TENSORBOARD", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        ):
            log_in_tensorboard = False
        if (
            self.game_name == "freeciv_remote"
            and os.getenv("MUZERO_ENABLE_LIVE_TEST_WORKER", "").strip().lower()
            not in ("1", "true", "yes", "y", "on")
        ):
            log_in_tensorboard = False
        if log_in_tensorboard or self.config.save_model:
            self.config.results_path.mkdir(parents=True, exist_ok=True)

        # Manage GPUs
        if self.role_gpu_pinning:
            num_gpus_per_worker = 0
        elif 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + log_in_tensorboard * self.config.selfplay_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0

        # Initialize workers
        training_options = {
            "num_cpus": 0,
            "num_gpus": num_gpus_per_worker if self.config.train_on_gpu else 0,
        }
        training_runtime_env = _actor_runtime_env_for_gpu(self.train_gpu_id)
        if self.config.train_on_gpu and training_runtime_env:
            training_options["runtime_env"] = training_runtime_env
        self.training_worker = trainer.Trainer.options(**training_options).remote(
            self.checkpoint, self.config
        )

        self.shared_storage_worker = shared_storage.SharedStorage.remote(
            self.checkpoint,
            self.config,
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
            self.checkpoint, self.replay_buffer, self.config
        )

        if self.config.use_last_model_value:
            reanalyse_options = {
                "num_cpus": 0,
                "num_gpus": num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
            }
            reanalyse_runtime_env = _actor_runtime_env_for_gpu(self.reanalyse_gpu_id)
            if self.config.reanalyse_on_gpu and reanalyse_runtime_env:
                reanalyse_options["runtime_env"] = reanalyse_runtime_env
            self.reanalyse_worker = replay_buffer.Reanalyse.options(
                **reanalyse_options
            ).remote(self.checkpoint, self.config)

        self.self_play_workers = []
        for seed in range(self.config.num_workers):
            selfplay_options = {
                "num_cpus": 0,
                "num_gpus": num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            }
            if self.config.selfplay_on_gpu and self.selfplay_gpu_ids:
                gpu_id = self.selfplay_gpu_ids[seed % len(self.selfplay_gpu_ids)]
                selfplay_options["runtime_env"] = _actor_runtime_env_for_gpu(gpu_id)
            self.self_play_workers.append(
                self_play.SelfPlay.options(**selfplay_options).remote(
                    self.checkpoint,
                    self.Game,
                    self.config,
                    self.config.seed + seed,
                )
            )

        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_play_worker in self.self_play_workers
        ]
        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )
        if self.config.use_last_model_value:
            self.reanalyse_worker.reanalyse.remote(
                self.replay_buffer_worker, self.shared_storage_worker
            )

        if log_in_tensorboard:
            start_time = time.time()
            self.logging_loop(
                num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
                start_time=start_time,
            )
        else:
            start_time = time.time()
            self.wait_for_training_end(start_time=start_time)

    def logging_loop(self, num_gpus, start_time=None):
        """
        Keep track of the training performance.
        """
        if start_time is None:
            start_time = time.time()
        # Launch the test worker to get performance metrics
        self.test_worker = self_play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(
            self.checkpoint,
            self.Game,
            self.config,
            self.config.seed + self.config.num_workers,
        )
        self.test_worker.continuous_self_play.remote(
            self.shared_storage_worker, None, True
        )

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path)

        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Save model representation
        writer.add_text(
            "Model summary",
            self.summary,
        )
        # Loop for updating the training performance
        counter = 0
        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                writer.add_scalar(
                    "1.Total_reward/1.Total_reward",
                    info["total_reward"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/2.Mean_value",
                    info["mean_value"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/3.Episode_length",
                    info["episode_length"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/4.MuZero_reward",
                    info["muzero_reward"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/5.Opponent_reward",
                    info["opponent_reward"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/1.Self_played_games",
                    info["num_played_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/2.Training_steps", info["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/3.Self_played_steps", info["num_played_steps"], counter
                )
                writer.add_scalar(
                    "2.Workers/4.Reanalysed_games",
                    info["num_reanalysed_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/5.Training_steps_per_self_played_step_ratio",
                    info["training_step"] / max(1, info["num_played_steps"]),
                    counter,
                )
                writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
                writer.add_scalar(
                    "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
                )
                writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
                self._emit_training_progress(info, include_reward=True)
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        self._finalize_training(start_time=start_time)

    def wait_for_training_end(self, start_time=None):
        """
        Block until training completes when TensorBoard logging is disabled.
        """
        if start_time is None:
            start_time = time.time()
        print("\nTraining (no TensorBoard)...\n")
        keys = ["training_step", "num_played_games", "total_loss"]
        try:
            while True:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                self._emit_training_progress(info)
                if info["training_step"] >= self.config.training_steps:
                    break
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        self._finalize_training(start_time=start_time)

    def _emit_training_progress(self, info, include_reward=False):
        if self._write_terminal_progress_bar(info):
            return
        print(self._training_progress_line(info, include_reward=include_reward), flush=True)

    def _training_progress_line(self, info, include_reward=False):
        training_steps = max(1, int(self.config.training_steps))
        training_step = int(info.get("training_step", 0))
        clamped_step = min(max(training_step, 0), training_steps)
        ratio = clamped_step / training_steps
        bar_width = 30
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)

        parts = [
            f"Training step: {training_step}/{self.config.training_steps}",
            f"[{bar}] {ratio * 100:6.2f}%",
            f'Played games: {info.get("num_played_games", 0)}',
            f'Loss: {float(info.get("total_loss", 0.0)):.2f}',
        ]
        if include_reward:
            parts.insert(3, f'Reward: {float(info.get("total_reward", 0.0)):.2f}')
        return " | ".join(parts)

    def _terminal_progress_allowed(self):
        value = os.getenv("MUZERO_TERMINAL_PROGRESS_BAR", "1").strip().lower()
        return value not in ("0", "false", "no", "off")

    def _init_terminal_progress_bar(self):
        if self._progress_tty or self._progress_tty_disabled:
            return bool(self._progress_tty)
        if not self._terminal_progress_allowed():
            self._progress_tty_disabled = True
            return False
        try:
            self._progress_tty = open("/dev/tty", "w", buffering=1)
        except OSError:
            self._progress_tty_disabled = True
            return False
        self._resize_terminal_progress_bar()
        return True

    def _resize_terminal_progress_bar(self):
        if not self._progress_tty:
            return None
        try:
            size = os.get_terminal_size(self._progress_tty.fileno())
        except OSError:
            size = shutil.get_terminal_size(fallback=(80, 24))
        rows = max(2, size.lines)
        if rows != self._progress_tty_rows:
            self._progress_tty.write(f"\0337\033[1;{rows - 1}r\033[{rows};1H\033[2K\0338")
            self._progress_tty.flush()
            self._progress_tty_rows = rows
        return rows, max(20, size.columns)

    def _write_terminal_progress_bar(self, info):
        if not self._init_terminal_progress_bar():
            return False
        size = self._resize_terminal_progress_bar()
        if not size:
            return False
        rows, columns = size

        training_steps = max(1, int(self.config.training_steps))
        training_step = int(info.get("training_step", 0))
        clamped_step = min(max(training_step, 0), training_steps)
        ratio = clamped_step / training_steps
        percent = ratio * 100

        text = (
            f" Training Step {training_step}/{self.config.training_steps}"
            f"  {percent:5.1f}%"
            f"  games {info.get('num_played_games', 0)}"
            f"  loss {float(info.get('total_loss', 0.0)):.2f}"
        )
        if len(text) > columns:
            text = text[:columns]
        text = text.ljust(columns)

        filled = min(columns, int(columns * ratio))
        if ratio > 0 and filled == 0:
            filled = 1
        filled_text = text[:filled]
        empty_text = text[filled:]
        line = f"\033[7m{filled_text}\033[0m{empty_text}"

        self._progress_tty.write(f"\0337\033[{rows};1H{line}\0338")
        self._progress_tty.flush()
        return True

    def _clear_terminal_progress_bar(self):
        if not self._progress_tty:
            return
        try:
            try:
                rows = os.get_terminal_size(self._progress_tty.fileno()).lines
            except OSError:
                rows = self._progress_tty_rows or 24
            self._progress_tty.write(f"\0337\033[r\033[{rows};1H\033[2K\0338")
            self._progress_tty.flush()
        finally:
            self._progress_tty.close()
            self._progress_tty = None
            self._progress_tty_rows = None

    def _finalize_training(self, start_time):
        self._clear_terminal_progress_bar()
        self.terminate_workers()

        if self.config.save_model:
            # Persist replay buffer to disk
            path = self.config.results_path / "replay_buffer.pkl"
            print(f"\n\nPersisting replay buffer games to disk at {path}")
            pickle.dump(
                {
                    "buffer": self.replay_buffer,
                    "num_played_games": self.checkpoint["num_played_games"],
                    "num_played_steps": self.checkpoint["num_played_steps"],
                    "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
                },
                open(path, "wb"),
            )
            drive_sync.sync_path(self.config.results_path, force=True)
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"\nTraining duration: {hours}h {minutes}m {seconds}s")

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        # Ask self-play workers to close their game clients promptly.
        if self.self_play_workers:
            for worker in self.self_play_workers:
                try:
                    worker.close_game.remote()
                except Exception:
                    pass
        if self.test_worker:
            try:
                self.test_worker.close_game.remote()
            except Exception:
                pass

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def test(
        self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0
    ):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """
        opponent = opponent if opponent else self.config.opponent
        muzero_player = muzero_player if muzero_player else self.config.muzero_player
        self_play_worker = self_play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(self.checkpoint, self.Game, self.config, numpy.random.randint(10000))
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(
                ray.get(
                    self_play_worker.play_game.remote(
                        0,
                        0,
                        render,
                        opponent,
                        muzero_player,
                    )
                )
            )
        self_play_worker.close_game.remote()

        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [
                    sum(
                        reward
                        for i, reward in enumerate(history.reward_history)
                        if history.to_play_history[i - 1] == muzero_player
                    )
                    for history in results
                ]
            )
        return result

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            checkpoint_path = pathlib.Path(checkpoint_path)
            self.checkpoint = torch.load(checkpoint_path)
            print(f"\nUsing checkpoint from {checkpoint_path}")

        # Load replay buffer
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_played_steps"] = replay_buffer_infos[
                "num_played_steps"
            ]
            self.checkpoint["num_played_games"] = replay_buffer_infos[
                "num_played_games"
            ]
            self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                "num_reanalysed_games"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")
        else:
            print(f"Using empty buffer.")
            self.replay_buffer = {}
            if not checkpoint_path:
                self.checkpoint["training_step"] = 0
            self.checkpoint["num_played_steps"] = 0
            self.checkpoint["num_played_games"] = 0
            self.checkpoint["num_reanalysed_games"] = 0

    def diagnose_model(self, horizon):
        """
        Play a game only with the learned model then play the same trajectory in the real
        environment and display information.

        Args:
            horizon (int): Number of timesteps for which we collect information.
        """
        try:
            import diagnose_model
        except ImportError as exc:
            raise RuntimeError(
                "diagnose_model requires optional plotting dependencies. "
                "Install seaborn/matplotlib to use this command."
            ) from exc
        try:
            game = self.Game(self.config.seed, config=self.config)
        except TypeError:
            game = self.Game(self.config.seed)
        obs = game.reset()
        dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
        dm.compare_virtual_with_real_trajectories(obs, game, horizon)
        input("Press enter to close all plots")
        dm.close_all()


@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = models.MuZeroNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary


def hyperparameter_search(
    game_name, parametrization, budget, parallel_experiments, num_tests
):
    """
    Search for hyperparameters by launching parallel experiments.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        parametrization : Nevergrad parametrization, please refer to nevergrad documentation.

        budget (int): Number of experiments to launch in total.

        parallel_experiments (int): Number of experiments to launch in parallel.

        num_tests (int): Number of games to average for evaluating an experiment.
    """
    try:
        import nevergrad
    except ImportError as exc:
        raise RuntimeError(
            "Hyperparameter search requires nevergrad. Install it to use this feature."
        ) from exc
    optimizer = nevergrad.optimizers.OnePlusOne(
        parametrization=parametrization, budget=budget
    )

    running_experiments = []
    best_training = None
    try:
        # Launch initial experiments
        for i in range(parallel_experiments):
            if 0 < budget:
                param = optimizer.ask()
                print(f"Launching new experiment: {param.value}")
                muzero = MuZero(game_name, param.value, parallel_experiments)
                muzero.param = param
                muzero.train(False)
                running_experiments.append(muzero)
                budget -= 1

        while 0 < budget or any(running_experiments):
            for i, experiment in enumerate(running_experiments):
                if experiment and experiment.config.training_steps <= ray.get(
                    experiment.shared_storage_worker.get_info.remote("training_step")
                ):
                    experiment.terminate_workers()
                    result = experiment.test(False, num_tests=num_tests)
                    if not best_training or best_training["result"] < result:
                        best_training = {
                            "result": result,
                            "config": experiment.config,
                            "checkpoint": experiment.checkpoint,
                        }
                    print(f"Parameters: {experiment.param.value}")
                    print(f"Result: {result}")
                    optimizer.tell(experiment.param, -result)

                    if 0 < budget:
                        param = optimizer.ask()
                        print(f"Launching new experiment: {param.value}")
                        muzero = MuZero(game_name, param.value, parallel_experiments)
                        muzero.param = param
                        muzero.train(False)
                        running_experiments[i] = muzero
                        budget -= 1
                    else:
                        running_experiments[i] = None

    except KeyboardInterrupt:
        for experiment in running_experiments:
            if isinstance(experiment, MuZero):
                experiment.terminate_workers()

    recommendation = optimizer.provide_recommendation()
    print("Best hyperparameters:")
    print(recommendation.value)
    if best_training:
        # Save best training weights (but it's not the recommended weights)
        best_training["config"].results_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            best_training["checkpoint"],
            best_training["config"].results_path / "model.checkpoint",
        )
        # Save the recommended hyperparameters
        text_file = open(
            best_training["config"].results_path / "best_parameters.txt",
            "w",
        )
        text_file.write(str(recommendation.value))
        text_file.close()
    return recommendation.value


def load_model_menu(muzero, game_name):
    # Configure running options
    options = ["Specify paths manually"] + sorted(
        (pathlib.Path("results") / game_name).glob("*/")
    )
    options.reverse()
    print()
    for i in range(len(options)):
        print(f"{i}. {options[i]}")

    choice = input("Enter a number to choose a model to load: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)

    if choice == (len(options) - 1):
        # manual path option
        checkpoint_path = input(
            "Enter a path to the model.checkpoint, or ENTER if none: "
        )
        while checkpoint_path and not pathlib.Path(checkpoint_path).is_file():
            checkpoint_path = input("Invalid checkpoint path. Try again: ")
        replay_buffer_path = input(
            "Enter a path to the replay_buffer.pkl, or ENTER if none: "
        )
        while replay_buffer_path and not pathlib.Path(replay_buffer_path).is_file():
            replay_buffer_path = input("Invalid replay buffer path. Try again: ")
    else:
        checkpoint_path = options[choice] / "model.checkpoint"
        replay_buffer_path = options[choice] / "replay_buffer.pkl"

    muzero.load_model(
        checkpoint_path=checkpoint_path,
        replay_buffer_path=replay_buffer_path,
    )


def _load_resume_paths(config):
    checkpoint_path = config.pop("checkpoint_path", None)
    replay_buffer_path = config.pop("replay_buffer_path", None)
    return checkpoint_path, replay_buffer_path


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Train directly with: python muzero.py cartpole
        muzero = MuZero(sys.argv[1])
        muzero.train()
    elif len(sys.argv) == 3:
        # Train directly with: python muzero.py cartpole '{"lr_init": 0.01}'
        config = json.loads(sys.argv[2])
        checkpoint_path, replay_buffer_path = _load_resume_paths(config)
        muzero = MuZero(sys.argv[1], config)
        if checkpoint_path or replay_buffer_path:
            muzero.load_model(
                checkpoint_path=checkpoint_path,
                replay_buffer_path=replay_buffer_path,
            )
        muzero.train()
    else:
        print("\nWelcome to MuZero! Here's a list of games:")
        # Let user pick a game
        games = [
            filename.stem
            for filename in sorted(list((pathlib.Path.cwd() / "games").glob("*.py")))
            if filename.name != "abstract_game.py"
        ]
        for i in range(len(games)):
            print(f"{i}. {games[i]}")
        choice = input("Enter a number to choose the game: ")
        valid_inputs = [str(i) for i in range(len(games))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")

        # Initialize MuZero
        choice = int(choice)
        game_name = games[choice]
        muzero = MuZero(game_name)

        while True:
            # Configure running options
            options = [
                "Train",
                "Load pretrained model",
                "Diagnose model",
                "Render some self play games",
                "Play against MuZero",
                "Test the game manually",
                "Hyperparameter search",
                "Exit",
            ]
            print()
            for i in range(len(options)):
                print(f"{i}. {options[i]}")

            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
            if choice == 0:
                muzero.train()
            elif choice == 1:
                load_model_menu(muzero, game_name)
            elif choice == 2:
                muzero.diagnose_model(30)
            elif choice == 3:
                muzero.test(render=True, opponent="self", muzero_player=None)
            elif choice == 4:
                muzero.test(render=True, opponent="human", muzero_player=0)
            elif choice == 5:
                try:
                    env = muzero.Game(config=muzero.config)
                except TypeError:
                    env = muzero.Game()
                env.reset()
                env.render()

                done = False
                while not done:
                    action = env.human_to_action()
                    observation, reward, done = env.step(action)
                    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                    env.render()
            elif choice == 6:
                # Define here the parameters to tune
                # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
                try:
                    import nevergrad
                except ImportError as exc:
                    raise RuntimeError(
                        "Hyperparameter search requires nevergrad. Install it to use this feature."
                    ) from exc
                muzero.terminate_workers()
                del muzero
                budget = 20
                parallel_experiments = 2
                lr_init = nevergrad.p.Log(lower=0.0001, upper=0.1)
                discount = nevergrad.p.Log(lower=0.95, upper=0.9999)
                parametrization = nevergrad.p.Dict(lr_init=lr_init, discount=discount)
                best_hyperparameters = hyperparameter_search(
                    game_name, parametrization, budget, parallel_experiments, 20
                )
                muzero = MuZero(game_name, best_hyperparameters)
            else:
                break
            print("\nDone")

    ray.shutdown()
