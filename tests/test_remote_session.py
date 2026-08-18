import unittest

from freeciv_sim.remote.lua_client import EvalResult
from freeciv_sim.remote.session import discover_client_player_id


class FakeClient:
    def __init__(self, value: str):
        self.value = value
        self.lua = ""

    def eval(self, lua: str) -> EvalResult:
        self.lua = lua
        return EvalResult(lines=[], returns=[self.value], errors=[])


class RemoteSessionTest(unittest.TestCase):
    def test_discover_client_player_id_uses_supported_player_binding(self):
        client = FakeClient("__PLAYER__ 4")

        self.assertEqual(discover_client_player_id(client), 4)
        self.assertIn("controlling_gui", client.lua)

    def test_discover_client_player_id_returns_none_for_observer(self):
        client = FakeClient("__PLAYER__ -1")

        self.assertIsNone(discover_client_player_id(client))


if __name__ == "__main__":
    unittest.main()
