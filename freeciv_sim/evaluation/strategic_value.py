from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..rules.research import TECH_PREREQS
from ..state.multihead_state import (
    BUILDING_SPECS,
    BUILDING_TECHS,
    GREAT_WONDER_NAMES,
    UNIT_SPECS,
    UNIT_TECHS,
    MultiheadState,
    Player,
)


@dataclass(frozen=True)
class StrategicValueWeights:
    city: float = 10.0
    population: float = 2.0
    land: float = 0.2
    military: float = 0.08
    research: float = 0.8
    future_research: float = 0.25
    production: float = 0.6
    exploration: float = 1.0
    safety: float = 3.0
    settler_need: float = 5.0
    redundancy: float = 0.18
    upkeep_pressure: float = 0.4


@dataclass(frozen=True)
class StrategicBreakdown:
    total: float
    cities: float
    population: float
    land: float
    military: float
    research: float
    production: float
    exploration: float
    safety: float


def strategic_potential(
    state: Optional[MultiheadState],
    player: Player = 1,
    weights: StrategicValueWeights = StrategicValueWeights(),
) -> StrategicBreakdown:
    if state is None:
        return StrategicBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    cities = weights.city * len(state.cities[player])
    population = weights.population * sum(city.size for city in state.cities[player])
    land = weights.land * _land_area(state, player)
    military = weights.military * _military_strength(state, player)
    research = weights.research * _completed_research_value(state, player)
    production = weights.production * _production_pipeline_value(state, player, weights)
    exploration = weights.exploration * _exploration_value(state, player)
    safety = weights.safety * _safety_value(state, player)
    total = (
        cities
        + population
        + land
        + military
        + research
        + production
        + exploration
        + safety
    )
    return StrategicBreakdown(
        total=total,
        cities=cities,
        population=population,
        land=land,
        military=military,
        research=research,
        production=production,
        exploration=exploration,
        safety=safety,
    )


def potential_shaping_reward(
    before: Optional[MultiheadState],
    after: Optional[MultiheadState],
    player: Player = 1,
    discount: float = 1.0,
    weights: StrategicValueWeights = StrategicValueWeights(),
) -> float:
    before_value = strategic_potential(before, player, weights).total
    after_value = strategic_potential(after, player, weights).total
    return discount * after_value - before_value


def research_completion_value(
    state: MultiheadState,
    player: Player,
    tech: str,
    weights: StrategicValueWeights = StrategicValueWeights(),
) -> float:
    if state.research_done[player].get(tech, False):
        return 0.0

    unlocks = 0.0
    for unit_name, prereqs in UNIT_TECHS.items():
        if _unlocked_by_finishing(state, player, tech, prereqs):
            unlocks += unit_asset_value(state, player, unit_name, weights)
    for building_name, prereqs in BUILDING_TECHS.items():
        if _unlocked_by_finishing(state, player, tech, prereqs):
            unlocks += building_asset_value(state, player, building_name, weights)

    future = 0.0
    for later_tech, prereqs in TECH_PREREQS.items():
        if state.research_done[player].get(later_tech, False):
            continue
        if tech in prereqs and _unlocked_by_finishing(state, player, tech, prereqs):
            future += weights.future_research

    return max(0.0, unlocks + future)


def production_asset_value(
    state: MultiheadState,
    player: Player,
    kind: str,
    name: str,
    weights: StrategicValueWeights = StrategicValueWeights(),
) -> float:
    if kind == "unit":
        return unit_asset_value(state, player, name, weights)
    if kind == "building":
        return building_asset_value(state, player, name, weights)
    return 0.0


def unit_asset_value(
    state: MultiheadState,
    player: Player,
    unit_name: str,
    weights: StrategicValueWeights = StrategicValueWeights(),
) -> float:
    spec = UNIT_SPECS.get(unit_name)
    if spec is None:
        return 0.0

    base = (
        spec.atk * 1.3
        + spec.df * 1.2
        + spec.hp * 0.08
        + spec.firepower * 0.8
        + spec.moves * 0.6
    )
    value = base

    need = _strategic_need(state, player)
    if spec.can_build_city:
        value += weights.settler_need * need["settler"]
    if spec.df >= spec.atk:
        value += (
            weights.safety
            * need["defense"]
            * (spec.df / max(1.0, spec.df + spec.atk))
        )
    if spec.atk > 0:
        value += (
            2.0
            * need["offense"]
            * (spec.atk / max(1.0, spec.df + spec.atk))
        )
    if spec.moves > 1:
        value += 1.5 * need["explore"]

    same_units = sum(
        1
        for unit in state.units[player]
        if unit.alive and unit.unit_type == unit_name
    )
    value -= weights.redundancy * same_units * base
    value -= weights.upkeep_pressure * _upkeep_pressure(state, player)
    return max(0.0, value)


def building_asset_value(
    state: MultiheadState,
    player: Player,
    building_name: str,
    weights: StrategicValueWeights = StrategicValueWeights(),
) -> float:
    spec = BUILDING_SPECS.get(building_name)
    if spec is None:
        return 0.0

    lower = building_name.lower()
    value = max(0.4, spec.cost / 80.0)
    need = _strategic_need(state, player)
    if building_name in GREAT_WONDER_NAMES:
        value += 2.0
    if "wall" in lower or "barracks" in lower:
        value += weights.safety * need["defense"]
    if any(word in lower for word in ("library", "university", "market", "bank")):
        value += 1.5 + len(state.cities[player]) * 0.4
    if any(word in lower for word in ("granary", "aqueduct", "sewer")):
        value += 1.0 + sum(city.size for city in state.cities[player]) * 0.1
    return max(0.0, value)


def _completed_research_value(state: MultiheadState, player: Player) -> float:
    total = 0.0
    for tech, done in state.research_done[player].items():
        if done:
            total += 1.0 + 0.2 * len(TECH_PREREQS.get(tech, ()))
    target = state.research_target.get(player)
    if target:
        cost = max(1.0, state._research_cost(player, target))
        progress = min(1.0, state.research_progress.get(player, 0.0) / cost)
        total += progress * research_completion_value(state, player, target)
    return total


def _production_pipeline_value(
    state: MultiheadState,
    player: Player,
    weights: StrategicValueWeights,
) -> float:
    total = 0.0
    for city in state.cities[player]:
        items = []
        if city.production_kind and city.production_target:
            items.append((city.production_kind, city.production_target, 1.0))
        items.extend(
            (kind, name, 0.45 ** (idx + 1))
            for idx, (kind, name) in enumerate(city.production_queue)
        )
        for kind, name, queue_discount in items:
            asset_value = production_asset_value(state, player, kind, name, weights)
            if kind == "unit":
                cost = max(
                    1.0,
                    float(UNIT_SPECS.get(name).cost if name in UNIT_SPECS else 1.0),
                )
            else:
                cost = max(
                    1.0,
                    float(
                        BUILDING_SPECS.get(name).cost
                        if name in BUILDING_SPECS
                        else 1.0
                    ),
                )
            progress = min(1.0, city.production_progress / cost)
            total += queue_discount * asset_value * (0.25 + 0.75 * progress)
    return total


def _military_strength(state: MultiheadState, player: Player) -> float:
    total = 0.0
    for unit in state.units[player]:
        if not unit.alive:
            continue
        total += (
            unit.atk * 1.3
            + unit.df * 1.2
            + unit.hp * 0.08
            + unit.firepower * 0.8
        )
    return total


def _safety_value(state: MultiheadState, player: Player) -> float:
    if not state.cities[player]:
        return 0.0
    garrisoned = 0
    exposed = 0
    for city in state.cities[player]:
        defenders = [
            unit
            for unit in state.units[player]
            if unit.alive and unit.x == city.x and unit.y == city.y
        ]
        if defenders:
            garrisoned += min(2, len(defenders))
        else:
            exposed += 1
    threat = _enemy_pressure(state, player)
    return garrisoned - exposed * (1.0 + threat)


def _exploration_value(state: MultiheadState, player: Player) -> float:
    visited = state.visited.get(player)
    if visited is None:
        return 0.0
    total_tiles = max(1, state.cfg.map_w * state.cfg.map_h)
    return float(visited.sum()) / float(total_tiles)


def _strategic_need(state: MultiheadState, player: Player) -> dict[str, float]:
    cities = len(state.cities[player])
    alive_units = [unit for unit in state.units[player] if unit.alive]
    settlers = sum(1 for unit in alive_units if unit.can_build_city)
    max_cities = max(1, state.max_cities)
    city_room = max(0.0, (max_cities - cities) / max_cities)
    defense_need = _defense_need(state, player)
    unexplored = 1.0 - _exploration_value(state, player)
    offense = min(
        1.0,
        _enemy_pressure(state, player) + 0.25 * len(state.cities[-player]),
    )
    return {
        "settler": city_room * (1.0 if settlers == 0 else 0.35),
        "defense": defense_need,
        "explore": max(0.0, unexplored),
        "offense": max(0.0, offense),
    }


def _defense_need(state: MultiheadState, player: Player) -> float:
    if not state.cities[player]:
        return 0.0
    desired = max(1, int(getattr(state.cfg, "city_unit_min", 1)))
    missing = 0
    for city in state.cities[player]:
        garrison = sum(
            1
            for unit in state.units[player]
            if unit.alive and unit.x == city.x and unit.y == city.y
        )
        missing += max(0, desired - garrison)
    return min(1.0, missing / max(1.0, desired * len(state.cities[player])))


def _enemy_pressure(state: MultiheadState, player: Player) -> float:
    if not state.cities[player]:
        return 0.0
    pressure = 0.0
    for city in state.cities[player]:
        for unit in state.units[-player]:
            if not unit.alive:
                continue
            dist = abs(unit.x - city.x) + abs(unit.y - city.y)
            if dist <= 3:
                pressure += (4 - dist) / 4.0
    return min(1.0, pressure / max(1, len(state.cities[player])))


def _upkeep_pressure(state: MultiheadState, player: Player) -> float:
    alive = sum(1 for unit in state.units[player] if unit.alive)
    free_per_city = int(getattr(state.cfg, "city_unit_free", 0))
    free_units = max(0, free_per_city * len(state.cities[player]))
    if alive <= free_units:
        return 0.0
    return (alive - free_units) / max(1.0, alive)


def _land_area(state: MultiheadState, player: Player) -> int:
    try:
        return int(state._land_area(player))
    except Exception:
        return len({(city.x, city.y) for city in state.cities[player]})


def _unlocked_by_finishing(
    state: MultiheadState,
    player: Player,
    tech: str,
    prereqs: list[str],
) -> bool:
    if tech not in prereqs:
        return False
    for req in prereqs:
        if req == tech:
            continue
        if not state.research_done[player].get(req, False):
            return False
    return True
