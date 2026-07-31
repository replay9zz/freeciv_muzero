import unittest

from freeciv_sim.evaluation.outcome import (
    GameOutcome,
    game_outcome,
    normalized_score_margin,
    optuna_objective,
)


class OutcomeTest(unittest.TestCase):
    def test_explicit_winner_overrides_score(self):
        result = game_outcome(
            {0: (5.0, True, "agent"), 1: (100.0, False, "ai")},
            0,
        )
        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.win_point, 1.0)
        self.assertEqual(result.decided_by, "winner")

    def test_strongest_opponent_score_is_used(self):
        result = game_outcome(
            {0: (20.0, None, "agent"), 1: (10.0, None, "ai1"), 2: (30.0, None, "ai2")},
            0,
        )
        self.assertEqual(result.opponent_score, 30.0)
        self.assertLess(result.value, 0.0)
        self.assertEqual(result.win_point, 0.0)

    def test_missing_score_is_draw_without_handcrafted_signal(self):
        result = game_outcome({0: (None, None, "agent")}, 0)
        self.assertEqual(result, GameOutcome(0.0, 0.5, None, None, "unavailable"))

    def test_margin_is_bounded(self):
        self.assertGreater(normalized_score_margin(100.0, 0.0), 0.0)
        self.assertLessEqual(normalized_score_margin(100.0, 0.0), 1.0)

    def test_objective_prefers_win_rate_over_margin(self):
        better_win_rate = [
            *[GameOutcome(-0.99, 1.0, 0.0, 100.0, "winner") for _ in range(6)],
            *[GameOutcome(-0.99, 0.0, 0.0, 100.0, "winner") for _ in range(4)],
        ]
        better_margin = [
            *[GameOutcome(0.99, 1.0, 100.0, 0.0, "score") for _ in range(5)],
            *[GameOutcome(0.99, 0.0, 100.0, 0.0, "score") for _ in range(5)],
        ]
        self.assertGreater(optuna_objective(better_win_rate), optuna_objective(better_margin))


if __name__ == "__main__":
    unittest.main()
