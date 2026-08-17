import unittest

from freeciv_sim.state.config import MapConfig
from freeciv_sim.state.multihead_state import MultiheadState
from freeciv_sim.state.providers import RandomMapProvider


class SimulatorEconomyTest(unittest.TestCase):
    def make_state(self, *, opponent_count: int = 1) -> MultiheadState:
        config = MapConfig(
            map_w=32,
            map_h=32,
            opponent_count=opponent_count,
        )
        return MultiheadState(
            config,
            RandomMapProvider(32, 32, p_open=1.0),
            max_units=24,
            max_cities=16,
        )

    def test_rate_action_updates_economy_and_encoding(self):
        state = self.make_state()
        before = state.encode(1)
        rate_idx = state.RATE_PRESETS.index((0, 40, 60))
        action = state.ECON_OFFSET + state.ECON_RATE_OFFSET + rate_idx

        self.assertEqual(1, state.valid_moves(1)[action])
        state.step(1, action)

        self.assertEqual((0, 40, 60), (
            state.tax_rates[1],
            state.luxury_rates[1],
            state.science_rates[1],
        ))
        self.assertEqual(before.shape, state.encode(1).shape)
        rate_start = state.ECON_OFFSET + state.ECON_RATE_OFFSET
        rate_end = state.ECON_OFFSET + state.ECON_PASS_OFFSET
        self.assertFalse(any(state.valid_moves(1)[rate_start:rate_end]))

        state.step(1, state.PASS_ACTION)
        self.assertTrue(any(state.valid_moves(1)[rate_start:rate_end]))

    def test_multiple_nations_are_aggregated_as_opponent(self):
        state = self.make_state(opponent_count=3)
        alive = [unit for unit in state.units[-1] if unit.alive]
        self.assertEqual(12, len(alive))
        self.assertGreaterEqual(len({(unit.x, unit.y) for unit in alive}), 3)


if __name__ == "__main__":
    unittest.main()
