from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Protocol


ROLE_ORDER = ("production", "research", "explore", "combat")


class RoleAgent(Protocol):
    role: str

    def legal_actions(self, game) -> list[int]:
        ...

    def preferred_actions(self, game) -> list[int]:
        ...


@dataclass
class RoleAgentRouter:
    agents: Dict[str, RoleAgent]

    def legal_actions(self, game) -> Dict[str, list[int]]:
        by_agent = game.legal_actions_by_agent()
        return {
            role: list(by_agent.get(role, []))
            for role in ROLE_ORDER
            if role in self.agents
        }

    def preferred_actions(self, game) -> Dict[str, list[int]]:
        return {
            role: list(self.agents[role].preferred_actions(game))
            for role in ROLE_ORDER
            if role in self.agents
        }

    def unit_roles(self, game) -> Dict[int, str]:
        return dict(game.unit_agent_roles())

    def city_roles(self, game) -> Dict[int, str]:
        city_slots = getattr(game, "city_slots", [])
        return {slot_idx: "production" for slot_idx in range(len(city_slots))}

    def active_roles(self, game) -> list[str]:
        legal = self.legal_actions(game)
        return [role for role in ROLE_ORDER if legal.get(role)]


def build_default_role_agents(agents: Iterable[RoleAgent]) -> RoleAgentRouter:
    return RoleAgentRouter({agent.role: agent for agent in agents})
