import socket
import unittest

from freeciv_sim.evaluation.live_status import (
    detect_live_terminal,
    terminal_status_for_exception,
)
from freeciv_sim.remote.lua_queries import (
    parse_live_game_snapshot,
    parse_player_game_status,
)


class LiveStatusTest(unittest.TestCase):
    def test_explicit_winner_is_terminal(self):
        statuses = parse_player_game_status("0|12|true|true|agent;1|9|false|false|ai")
        result = detect_live_terminal(statuses)
        self.assertTrue(result.terminal)
        self.assertEqual(result.winner_ids, (0,))
        self.assertEqual(result.reason, "winner")

    def test_no_alive_players_is_terminal_draw(self):
        statuses = parse_player_game_status("0|12|false|false|agent;1|9|false|false|ai")
        result = detect_live_terminal(statuses)
        self.assertTrue(result.terminal)
        self.assertEqual(result.winner_ids, ())
        self.assertEqual(result.reason, "draw")

    def test_client_over_is_terminal_draw_with_alive_players(self):
        statuses = parse_player_game_status("0|12|false|true|agent;1|12|false|true|ai")
        result = detect_live_terminal(statuses, "over")
        self.assertTrue(result.terminal)
        self.assertEqual(result.reason, "draw")

    def test_client_disconnect_is_terminal(self):
        result = detect_live_terminal({}, "disconnected")
        self.assertTrue(result.terminal)
        self.assertEqual(result.reason, "disconnect")

    def test_running_empty_snapshot_is_not_terminal(self):
        self.assertFalse(detect_live_terminal({}, "running").terminal)

    def test_partial_alive_data_does_not_guess_terminal(self):
        statuses = parse_player_game_status("0|nil|nil|nil|agent;1|bad|false|false|ai")
        self.assertFalse(detect_live_terminal(statuses).terminal)

    def test_malformed_entries_are_skipped(self):
        statuses = parse_player_game_status("broken;x|1|true|true|bad;2|3|false|true|ok")
        self.assertEqual(statuses, {2: (3.0, False, True, "ok")})

    def test_snapshot_parser_separates_client_state(self):
        snapshot = parse_live_game_snapshot(
            "@state|over;0|12|true|true|agent;broken"
        )
        self.assertEqual(snapshot.client_state, "over")
        self.assertEqual(snapshot.players, {0: (12.0, True, True, "agent")})

    def test_snapshot_parser_supports_client_without_state_api(self):
        snapshot = parse_live_game_snapshot("@state|nil;0|12|false|true|agent")
        self.assertIsNone(snapshot.client_state)
        self.assertEqual(snapshot.players[0][2], True)

    def test_transport_failures_are_classified(self):
        self.assertEqual(terminal_status_for_exception(socket.timeout()).reason, "timeout")
        self.assertEqual(
            terminal_status_for_exception(ConnectionResetError()).reason,
            "disconnect",
        )
        self.assertEqual(terminal_status_for_exception(ValueError()).reason, "error")


if __name__ == "__main__":
    unittest.main()
