from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProductionAgent:
    role: str = "production"

    def legal_actions(self, game) -> list[int]:
        return list(game.legal_actions_by_agent().get(self.role, []))

    def preferred_actions(self, game) -> list[int]:
        legal = self.legal_actions(game)
        if not legal:
            return []
        desired_unit = getattr(game, "_select_production_unit", lambda: None)()
        if not desired_unit or getattr(game, "_last_state", None) is None:
            return legal
        state = game._last_state
        econ_offset = state.MOVE_SIZE + state.ATTACK_SIZE
        prod_start = econ_offset + state.ECON_PRODUCTION_OFFSET
        preferred = []
        for action in legal:
            if action < prod_start:
                continue
            rel = action - prod_start
            item_idx = rel % state.PRODUCTION_ITEM_COUNT
            kind, name = state.PRODUCTION_ITEM_NAMES[item_idx]
            if kind == "unit" and name == desired_unit:
                preferred.append(action)
        return preferred or legal
