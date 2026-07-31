import math
import os
import sys
import time

import numpy
import ray
import torch

import models


@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        if hasattr(self.config, "luaremote_port_base"):
            base = int(self.config.luaremote_port_base)
            stride = int(getattr(self.config, "luaremote_port_stride", 1))
            offset = seed - self.config.seed
            if offset < 0:
                offset = 0
            port = base + (offset * stride)
            os.environ["FREECIV_LUAREMOTE_PORT"] = str(port)
            os.environ["FREECIV_PORT"] = str(port)
        if hasattr(self.config, "server_port_base"):
            base = int(self.config.server_port_base)
            stride = int(getattr(self.config, "server_port_stride", 1))
            offset = seed - self.config.seed
            if offset < 0:
                offset = 0
            port = base + (offset * stride)
            os.environ["FREECIV_SERVER_PORT"] = str(port)
            os.environ["FREECIV_GAME_PORT"] = str(port)
        try:
            self.game = Game(seed, config=self.config)
        except TypeError:
            self.game = Game(seed)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()
        if self.config.selfplay_on_gpu and torch.cuda.is_available():
            print(
                "[gpu] selfplay "
                f"visible={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')} "
                f"device={torch.cuda.current_device()} "
                f"name={torch.cuda.get_device_name(torch.cuda.current_device())}",
                file=sys.stderr,
            )

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )

                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                if 1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.muzero_player
                            ),
                        }
                    )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        with torch.no_grad():
            last_logged_turn = None
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )

                # Choose the action
                if opponent == "self" or muzero_player == self.game.to_play():
                    root, mcts_info = MCTS(self.config).run(
                        self.model,
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.to_play(),
                        True,
                    )
                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )

                    if render:
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(
                            f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                        )
                else:
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations
                    )

                observation, reward, done = self.game.step(action)
                turn = getattr(self.game, "turns", None)
                if turn is not None and turn != last_logged_turn:
                    last_logged_turn = turn
                    muzero_score = None
                    state = getattr(self.game, "_last_state", None)
                    if state is not None and hasattr(state, "muzero_score"):
                        try:
                            muzero_score = state.muzero_score(1)
                        except Exception:
                            muzero_score = None
                    score_line = f"[selfplay] turn={turn}"
                    player_scores = getattr(self.game, "player_scores", None)
                    if isinstance(player_scores, dict) and player_scores:
                        player_id = getattr(self.game, "player_id", None)
                        if player_id is not None:
                            civ_score = player_scores.get(
                                int(player_id), (None, None, "")
                            )[0]
                            if civ_score is not None:
                                score_line += f" civ_score={civ_score:.2f}"
                        if muzero_score is not None:
                            score_line += f" muzero_score={muzero_score:.2f}"
                        parts = []
                        for pid in sorted(player_scores.keys()):
                            score, win, name = player_scores[pid]
                            tag = f"{pid}"
                            if name:
                                tag += f":{name}"
                            if score is not None:
                                tag += f":{score:.0f}"
                            if win is True:
                                tag += ":W"
                            elif done and win is False:
                                tag += ":L"
                            parts.append(tag)
                        score_line += " civ_scores=[" + ",".join(parts) + "]"
                    elif muzero_score is not None:
                        score_line += f" muzero_score={muzero_score:.2f}"
                    print(score_line)

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())

        outcome = getattr(self.game, "last_outcome", None)
        if outcome is not None:
            own_score = "" if outcome.own_score is None else f"{outcome.own_score:.6f}"
            opponent_score = (
                "" if outcome.opponent_score is None else f"{outcome.opponent_score:.6f}"
            )
            print(
                "[selfplay-result] "
                f"outcome={outcome.value:.9f} win_point={outcome.win_point:.1f} "
                f"own_score={own_score} opponent_score={opponent_score} "
                f"decided_by={outcome.decided_by}",
                flush=True,
            )
        return game_history

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            (
                root_predicted_value,
                reward,
                policy_logits,
                hidden_state,
            ) = model.initial_inference(observation)
            root_predicted_value = models.support_to_scalar(
                root_predicted_value, self.config.support_size
            ).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            root.expand(
                legal_actions,
                to_play,
                reward,
                policy_logits,
                hidden_state,
                child_is_chance=self._use_stochastic_muzero(),
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                # Players play turn by turn
                if not node.is_chance:
                    virtual_to_play = self.next_player(virtual_to_play)

            parent = search_path[-2]
            if self._use_stochastic_muzero() and node.is_chance:
                outcome = 0
                value, reward, policy_logits, hidden_state = model.recurrent_inference(
                    parent.hidden_state,
                    torch.tensor([[action]]).to(parent.hidden_state.device),
                )
                value = models.support_to_scalar(value, self.config.support_size).item()
                reward = models.support_to_scalar(reward, self.config.support_size).item()
                node.reward = reward
                node.hidden_state = hidden_state
                outcome_node = Node(1.0)
                outcome_node.is_chance_outcome = True
                outcome_node.expand(
                    self.config.action_space,
                    self.next_player(virtual_to_play),
                    0,
                    policy_logits,
                    hidden_state,
                    child_is_chance=True,
                )
                node.children[outcome] = outcome_node
                virtual_to_play = outcome_node.to_play
            else:
                # Inside the search tree we use the dynamics function to obtain the next
                # hidden state given an action and the previous hidden state.
                value, reward, policy_logits, hidden_state = model.recurrent_inference(
                    parent.hidden_state,
                    torch.tensor([[action]]).to(parent.hidden_state.device),
                )
                value = models.support_to_scalar(value, self.config.support_size).item()
                reward = models.support_to_scalar(reward, self.config.support_size).item()
                node.expand(
                    self.config.action_space,
                    virtual_to_play,
                    reward,
                    policy_logits,
                    hidden_state,
                    child_is_chance=self._use_stochastic_muzero(),
                )

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        if node.is_chance:
            outcomes = list(node.children.keys())
            probabilities = numpy.array(
                [node.children[outcome].prior for outcome in outcomes],
                dtype="float64",
            )
            probabilities = probabilities / probabilities.sum()
            outcome = numpy.random.choice(outcomes, p=probabilities)
            return outcome, node.children[outcome]

        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def next_player(self, to_play):
        if to_play + 1 < len(self.config.players):
            return self.config.players[to_play + 1]
        return self.config.players[0]

    def _use_stochastic_muzero(self):
        return bool(getattr(self.config, "use_stochastic_muzero", False))

    def _use_wasserstein_mcts(self):
        return str(
            getattr(self.config, "mcts_backup_operator", "mean")
        ).strip().lower() in {"wasserstein", "w-mcts", "wmcts"}

    def _wasserstein_power_mean(self, values, weights, power):
        weighted_values = [
            (float(value), float(weight))
            for value, weight in zip(values, weights)
            if float(weight) > 0
        ]
        if not weighted_values:
            return 0.0
        weight_sum = sum(weight for _, weight in weighted_values)
        normalized = [(value, weight / weight_sum) for value, weight in weighted_values]
        if abs(power - 1.0) < 1e-8:
            return sum(weight * value for value, weight in normalized)

        # Power means with non-integer powers are undefined for negative values.
        # Shift all children into the positive domain, aggregate, then shift back.
        min_value = min(value for value, _ in normalized)
        shift = max(0.0, -min_value) + getattr(
            self.config, "mcts_wasserstein_shift_epsilon", 1e-6
        )
        powered = sum(weight * ((value + shift) ** power) for value, weight in normalized)
        return max(powered, 0.0) ** (1.0 / power) - shift

    def _refresh_wasserstein_value(self, node):
        if not node.children:
            return
        visited_children = [
            child for child in node.children.values() if child.visit_count > 0
        ]
        if not visited_children:
            return
        weights = [child.visit_count for child in visited_children]
        q_means = [
            child.reward + self.config.discount * child.value()
            for child in visited_children
        ]
        q_stds = [
            self.config.discount * child.value_std()
            for child in visited_children
        ]
        power = float(getattr(self.config, "mcts_wasserstein_power", 1.0))
        node.wasserstein_value_mean = self._wasserstein_power_mean(
            q_means, weights, power
        )
        node.wasserstein_value_std = max(
            self._wasserstein_power_mean(q_stds, weights, power),
            float(getattr(self.config, "mcts_wasserstein_min_std", 1e-6)),
        )

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            value_score = 0

        score = prior_score + value_score
        if self._use_wasserstein_mcts() and child.visit_count > 0:
            selection = str(
                getattr(self.config, "mcts_wasserstein_selection", "optimistic")
            ).strip().lower()
            if selection == "thompson":
                sampled_value = numpy.random.normal(
                    child.value(), max(child.value_std(), 1e-6)
                )
                sampled_value = sampled_value if len(self.config.players) == 1 else -sampled_value
                score += min_max_stats.normalize(
                    child.reward + self.config.discount * sampled_value
                ) - value_score
            else:
                uncertainty_coef = float(
                    getattr(self.config, "mcts_wasserstein_uncertainty_coef", 0.0)
                )
                if uncertainty_coef:
                    score += (
                        uncertainty_coef
                        * child.value_std()
                        * math.sqrt(math.log(parent.visit_count + 1))
                    )
        return score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                backed_up_value = value
                node.value_sum += backed_up_value
                node.value_sq_sum += backed_up_value * backed_up_value
                node.visit_count += 1
                if self._use_wasserstein_mcts():
                    self._refresh_wasserstein_value(node)
                min_max_stats.update(node.reward + self.config.discount * node.value())

                if node.is_chance_outcome:
                    value = node.reward + value
                else:
                    value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                backed_up_value = value if node.to_play == to_play else -value
                node.value_sum += backed_up_value
                node.value_sq_sum += backed_up_value * backed_up_value
                node.visit_count += 1
                if self._use_wasserstein_mcts():
                    self._refresh_wasserstein_value(node)
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                reward = -node.reward if node.to_play == to_play else node.reward
                if node.is_chance_outcome:
                    value = reward + value
                else:
                    value = reward + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.is_chance = False
        self.is_chance_outcome = False
        self.value_sum = 0
        self.value_sq_sum = 0
        self.wasserstein_value_mean = None
        self.wasserstein_value_std = None
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.wasserstein_value_mean is not None:
            return self.wasserstein_value_mean
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def value_std(self):
        if self.wasserstein_value_std is not None:
            return self.wasserstein_value_std
        if self.visit_count <= 1:
            return 1.0
        mean = self.value_sum / self.visit_count
        variance = self.value_sq_sum / self.visit_count - mean * mean
        return math.sqrt(max(variance, 1e-6))

    def expand(
        self,
        actions,
        to_play,
        reward,
        policy_logits,
        hidden_state,
        child_is_chance=False,
    ):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            child = Node(p)
            child.is_chance = child_is_chance
            child.to_play = to_play if child_is_chance else -1
            self.children[action] = child

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = numpy.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            numpy.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = numpy.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
