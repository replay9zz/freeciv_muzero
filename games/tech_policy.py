from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence


RESEARCH_PRIORITY_CHAIN: tuple[str, ...] = (
    "Warrior Code",
    "Bronze Working",
    "Monarchy",
    "Gunpowder",
    "Democracy",
    "Robotics",
)


def pick_next_goal_tech(
    goal: str,
    researched: Dict[str, bool],
    prereqs: Dict[str, List[str]],
    available: Iterable[str],
) -> Optional[str]:
    """
    Pick the next missing tech on the path to goal, respecting prereqs.
    """
    available_set = set(available)
    if goal not in available_set or researched.get(goal, False):
        return None
    seen: set[str] = set()

    def walk(tech: str) -> Optional[str]:
        if tech in seen:
            return None
        seen.add(tech)
        if researched.get(tech, False):
            return None
        for req in prereqs.get(tech, []):
            if req not in available_set:
                continue
            if researched.get(req, False):
                continue
            candidate = walk(req)
            if candidate:
                return candidate
        return tech

    return walk(goal)


def pick_next_priority_tech(
    researched: Dict[str, bool],
    prereqs: Dict[str, List[str]],
    available: Iterable[str],
    chain: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    Pick the next tech from the priority chain.
    """
    priority = chain or RESEARCH_PRIORITY_CHAIN
    for tech in priority:
        if researched.get(tech, False):
            continue
        candidate = pick_next_goal_tech(tech, researched, prereqs, available)
        if candidate:
            return candidate
    return None
