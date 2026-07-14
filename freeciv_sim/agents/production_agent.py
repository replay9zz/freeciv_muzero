from __future__ import annotations

from dataclasses import dataclass

from freeciv_sim.evaluation import production_asset_value


@dataclass
class ProductionAgent:
    role: str = "production"

    def legal_actions(self, game) -> list[int]:
        return list(game.legal_actions_by_agent().get(self.role, []))

    def preferred_actions(self, game) -> list[int]:
        legal = self.legal_actions(game)
        if not legal:
            return []
        if getattr(game, "_last_state", None) is None:
            return legal
        state = game._last_state
        econ_offset = state.ECON_OFFSET
        prod_start = econ_offset + state.ECON_PRODUCTION_OFFSET
        if len(state.cities[1]) < state.max_cities:
            live_settlers = sum(
                1 for unit in state.units[1] if unit.alive and unit.can_build_city
            )
            if live_settlers == 0:
                settler_actions = []
                for action in legal:
                    if action < prod_start:
                        continue
                    rel = action - prod_start
                    item_idx = rel % state.PRODUCTION_ITEM_COUNT
                    kind, name = state.PRODUCTION_ITEM_NAMES[item_idx]
                    if kind == "unit" and name == "Settlers":
                        settler_actions.append(action)
                if settler_actions:
                    return settler_actions
        valued = []
        for action in legal:
            if action < prod_start:
                continue
            rel = action - prod_start
            item_idx = rel % state.PRODUCTION_ITEM_COUNT
            kind, name = state.PRODUCTION_ITEM_NAMES[item_idx]
            valued.append((production_asset_value(state, 1, kind, name), action))
        if valued:
            best_value = max(value for value, _action in valued)
            if best_value > 0.0:
                return [
                    action
                    for value, action in valued
                    if value >= best_value * 0.95
                ]

        desired_unit = getattr(game, "_select_production_unit", lambda: None)()
        if not desired_unit:
            return legal
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
