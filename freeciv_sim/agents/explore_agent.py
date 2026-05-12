from __future__ import annotations

from dataclasses import dataclass

import numpy


@dataclass
class ExploreAgent:
    role: str = "explore"

    def legal_actions(self, game) -> list[int]:
        return list(game.legal_actions_by_agent().get(self.role, []))

    def preferred_actions(self, game) -> list[int]:
        legal = self.legal_actions(game)
        state = getattr(game, "_last_state", None)
        if not legal or state is None:
            return legal
        valid = numpy.zeros(state.ACTION_SIZE, dtype=numpy.int8)
        for action in legal:
            if 0 <= action < len(valid):
                valid[action] = 1
        forced = getattr(game, "_force_settler_city_actions", lambda _valid: None)(valid)
        if forced:
            return [action for action in forced if action in legal]
        return legal
