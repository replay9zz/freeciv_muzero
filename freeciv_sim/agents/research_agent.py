from __future__ import annotations

from dataclasses import dataclass

from freeciv_sim.evaluation import research_completion_value
from freeciv_sim.rules.research import TECH_PREREQS, pick_next_priority_tech


@dataclass
class ResearchAgent:
    role: str = "research"

    def legal_actions(self, game) -> list[int]:
        return list(game.legal_actions_by_agent().get(self.role, []))

    def preferred_actions(self, game) -> list[int]:
        legal = self.legal_actions(game)
        state = getattr(game, "_last_state", None)
        if not legal or state is None:
            return legal
        econ_offset = state.MOVE_SIZE + state.ATTACK_SIZE
        valued = []
        for action in legal:
            tech_idx = action - econ_offset
            if 0 <= tech_idx < len(state.RESEARCH_TECHS):
                tech = state.RESEARCH_TECHS[tech_idx]
                valued.append((research_completion_value(state, 1, tech), action))
        if valued:
            best_value = max(value for value, _action in valued)
            if best_value > 0.0:
                return [
                    action
                    for value, action in valued
                    if value >= best_value * 0.95
                ]
        flags = state.research_done.get(1, {})
        tech_name = pick_next_priority_tech(flags, TECH_PREREQS, state.RESEARCH_TECHS)
        if not tech_name:
            return legal
        try:
            tech_idx = state.RESEARCH_TECHS.index(tech_name)
        except ValueError:
            return legal
        action_idx = econ_offset + tech_idx
        if action_idx in legal:
            return [action_idx]
        return legal
