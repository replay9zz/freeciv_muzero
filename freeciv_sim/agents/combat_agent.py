from __future__ import annotations

from dataclasses import dataclass

import numpy


@dataclass
class CombatAgent:
    role: str = "combat"

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
        preferred_mask = getattr(game, "_prefer_attack_actions", lambda _valid: _valid)(valid)
        preferred = [
            action
            for action in legal
            if 0 <= action < len(preferred_mask) and preferred_mask[action]
        ]
        return preferred or legal
