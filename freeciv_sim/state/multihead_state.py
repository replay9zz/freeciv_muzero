from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .movement import FreecivMovement
from .config import MapConfig
from .providers import BaseProvider, GroundTruth
from ..rules.research import (
    RESEARCH_TECHS,
    TARGET_TECH_NAME,
    TECH_COSTS,
    TECH_PREREQS,
    TECH_COST_STYLE,
    TECH_COST_FACTOR,
    BASE_TECH_COST,
    MIN_TECH_COST,
    build_tech_costs,
)
from ..rules.ruleset_loader import load_civ2civ3_ruleset

Player = int  # 1 or -1
Coord = Tuple[int, int]


@dataclass
class UnitSpec:
    name: str
    unit_class: str
    atk: int
    df: int
    hp: int
    firepower: int
    moves: int
    cost: int
    can_build_city: bool = False
    obsolete_by: Optional[str] = None


@dataclass
class BuildingSpec:
    name: str
    cost: int
    req_techs: List[str] = field(default_factory=list)
    req_buildings: List[str] = field(default_factory=list)
    genus: str = ""
    flags: List[str] = field(default_factory=list)


@dataclass
class City:
    x: int
    y: int
    size: int = 1
    food_storage: float = 0.0
    production_kind: Optional[str] = None  # "unit" or "building"
    production_target: Optional[str] = None
    production_progress: float = 0.0
    production_cost: int = 0
    production_per_turn: int = 0
    production_turns: int = 0
    production_queue: list[tuple[str, str]] = field(default_factory=list)
    has_city_walls: bool = False
    buildings: set[str] = field(default_factory=set)


@dataclass
class MHUnit:
    x: int
    y: int
    hp: int
    atk: int
    df: int
    firepower: int
    unit_type: str
    alive: bool = True
    can_build_city: bool = False
    home_city: Optional[int] = None
    moves_left: int = 0
    last_move_turn: int = -1


_RULESET = load_civ2civ3_ruleset()

PRODUCTION_UNIT_NAMES: Tuple[str, ...] = tuple(rule.name for rule in _RULESET.units)
PRODUCTION_BUILDING_NAMES: Tuple[str, ...] = tuple(
    rule.name for rule in _RULESET.buildings
)
UNIT_TYPE_NAMES: Tuple[str, ...] = PRODUCTION_UNIT_NAMES
UNIT_TYPE_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(UNIT_TYPE_NAMES)
}
PRODUCTION_ITEM_NAMES: Tuple[Tuple[str, str], ...] = tuple(
    [("unit", name) for name in PRODUCTION_UNIT_NAMES]
    + [("building", name) for name in PRODUCTION_BUILDING_NAMES]
)
PRODUCTION_ITEM_INDEX: Dict[Tuple[str, str], int] = {
    item: idx for idx, item in enumerate(PRODUCTION_ITEM_NAMES)
}

UNIT_SPECS: Dict[str, UnitSpec] = {}
UNIT_TECHS: Dict[str, List[str]] = {}
UNIT_OBSOLETE_BY: Dict[str, Optional[str]] = {}
SEA_UNIT_CLASSES = {"sea", "trireme"}
for rule in _RULESET.units:
    can_build = "Cities" in rule.flags
    UNIT_SPECS[rule.name] = UnitSpec(
        name=rule.name,
        unit_class=rule.unit_class,
        atk=rule.attack,
        df=rule.defense,
        hp=rule.hp,
        firepower=rule.firepower,
        moves=rule.moves,
        cost=rule.cost,
        can_build_city=can_build,
        obsolete_by=rule.obsolete_by,
    )
    UNIT_TECHS[rule.name] = list(rule.req_techs)
    UNIT_OBSOLETE_BY[rule.name] = rule.obsolete_by

TERRAIN_SPECS = {rule.name.lower(): rule for rule in _RULESET.terrains}
_MAX_UNIT_ATTACK = max((rule.attack for rule in _RULESET.units), default=1)
_MAX_UNIT_DEFENSE = max((rule.defense for rule in _RULESET.units), default=1)
_MAX_UNIT_HP = max((rule.hp for rule in _RULESET.units), default=1)
_MAX_UNIT_FIREPOWER = max((rule.firepower for rule in _RULESET.units), default=1)
_MAX_UNIT_MOVES = max((rule.moves for rule in _RULESET.units), default=1)
_MAX_UNIT_COST = max((rule.cost for rule in _RULESET.units), default=1)
_MAX_TERRAIN_FOOD = max((rule.food for rule in _RULESET.terrains), default=1)
_MAX_TERRAIN_SHIELD = max((rule.shield for rule in _RULESET.terrains), default=1)
_MAX_TERRAIN_TRADE = max((rule.trade for rule in _RULESET.terrains), default=1)
_MAX_TERRAIN_MOVE = max((rule.movement_cost for rule in _RULESET.terrains), default=1)

UNIT_GENERATION_CHAINS: Tuple[Tuple[str, ...], ...] = (
    (
        "Warriors",
        "Phalanx",
        "Pikemen",
        "Musketeers",
        "Riflemen",
        "Alpine Troops",
        "Mech. Inf.",
    ),
    (
        "Warriors",
        "Legion",
        "Archers",
        "Musketeers",
        "Riflemen",
        "Alpine Troops",
        "Mech. Inf.",
    ),
    ("Catapult", "Cannon", "Artillery", "Howitzer"),
    ("Horsemen", "Chariot", "Knights", "Dragoons", "Cavalry", "Armor"),
    (
        "Horsemen",
        "Elephants",
        "Crusaders",
        "Knights",
        "Dragoons",
        "Cavalry",
        "Armor",
    ),
)

BUILDING_SPECS: Dict[str, BuildingSpec] = {}
BUILDING_TECHS: Dict[str, List[str]] = {}
BUILDING_REQ_BUILDINGS: Dict[str, List[str]] = {}
GREAT_WONDER_NAMES: set[str] = set()
for rule in _RULESET.buildings:
    BUILDING_SPECS[rule.name] = BuildingSpec(
        name=rule.name,
        cost=rule.cost,
        req_techs=list(rule.req_techs),
        req_buildings=list(rule.req_buildings),
        genus=rule.genus,
        flags=list(rule.flags),
    )
    BUILDING_TECHS[rule.name] = list(rule.req_techs)
    BUILDING_REQ_BUILDINGS[rule.name] = list(rule.req_buildings)
    if rule.genus == "GreatWonder":
        GREAT_WONDER_NAMES.add(rule.name)


@dataclass
class MultiheadState:
    """
    Prototype multi-unit state for multi-head training:
    - Multiple units per side (fixed slots, default 4)
    - Move or attack per unit
    - Research actions preserved to keep tech dimensions aligned
    """
    cfg: MapConfig
    provider: BaseProvider
    max_units: int = 6
    max_cities: int = 16
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    gt: GroundTruth | None = None
    movement: FreecivMovement | None = None
    units: Dict[Player, List[MHUnit]] = field(default_factory=lambda: {1: [], -1: []})
    cities: Dict[Player, List[City]] = field(default_factory=lambda: {1: [], -1: []})
    research_done: Dict[Player, Dict[str, bool]] = field(default_factory=lambda: {1: {}, -1: {}})
    research_target: Dict[Player, Optional[str]] = field(
        default_factory=lambda: {1: None, -1: None}
    )
    research_progress: Dict[Player, float] = field(
        default_factory=lambda: {1: 0.0, -1: 0.0}
    )
    visited: Dict[Player, np.ndarray] = field(default_factory=lambda: {1: None, -1: None})
    turn: int = 0
    num_actions: int = 0
    actions_this_turn: int = 0
    max_actions_per_turn: int = 0
    acted_unit_slots: Dict[Player, set[int]] = field(
        default_factory=lambda: {1: set(), -1: set()}
    )
    acted_production_cities: Dict[Player, set[int]] = field(
        default_factory=lambda: {1: set(), -1: set()}
    )
    kills: Dict[Player, int] = field(default_factory=lambda: {1: 0, -1: 0})
    units_built: Dict[Player, int] = field(default_factory=lambda: {1: 0, -1: 0})
    future_techs: Dict[Player, int] = field(default_factory=lambda: {1: 0, -1: 0})
    scores: Dict[Player, float] = field(default_factory=lambda: {1: 0.0, -1: 0.0})
    winner: Optional[Player] = None
    terminal_reason: Optional[str] = None
    tech_costs: Dict[str, float] = field(default_factory=lambda: dict(TECH_COSTS))

    # Action layout:
    # Unit blocks: movement, directional attack, named unit activities.
    # After unit actions: research, build city, production, pass.
    RESEARCH_TECHS: Tuple[str, ...] = RESEARCH_TECHS
    PRODUCTION_UNIT_NAMES: Tuple[str, ...] = PRODUCTION_UNIT_NAMES
    PRODUCTION_BUILDING_NAMES: Tuple[str, ...] = PRODUCTION_BUILDING_NAMES
    PRODUCTION_ITEM_NAMES: Tuple[Tuple[str, str], ...] = PRODUCTION_ITEM_NAMES
    AGENT_ROLES: Tuple[str, ...] = ("explore", "combat", "research", "production")
    MOVE_DIRECTIONS = 6
    HOLD_DIR = 6
    MOVE_PER_UNIT = MOVE_DIRECTIONS + 1
    ATTACK_PER_UNIT = 6
    UNIT_ACTIVITY_NAMES: Tuple[str, ...] = (
        "fortify",
        "sentry",
        "road",
        "irrigate",
        "mine",
        "cultivate",
        "plant",
        "transform",
        "clean",
        "pillage",
        "fortress",
        "airbase",
    )
    UNIT_ACTIVITY_PER_UNIT = len(UNIT_ACTIVITY_NAMES)

    def __post_init__(self) -> None:
        self.movement = FreecivMovement(self.cfg.map_w, self.cfg.map_h)
        # Head sizes
        self.MOVE_SIZE = self.max_units * self.MOVE_PER_UNIT
        self.ATTACK_SIZE = self.max_units * self.ATTACK_PER_UNIT
        self.UNIT_ACTIVITY_SIZE = self.max_units * self.UNIT_ACTIVITY_PER_UNIT
        self.UNIT_ACTIVITY_OFFSET = self.MOVE_SIZE + self.ATTACK_SIZE
        self.ECON_OFFSET = self.UNIT_ACTIVITY_OFFSET + self.UNIT_ACTIVITY_SIZE
        # Econ head contains research actions, build-city actions (per unit slot),
        # production actions (per city slot), and a pass/turn-end action.
        self.ECON_RESEARCH_OFFSET = 0
        self.ECON_BUILD_CITY_OFFSET = len(self.RESEARCH_TECHS)
        self.ECON_PRODUCTION_OFFSET = self.ECON_BUILD_CITY_OFFSET + self.max_units
        self.PRODUCTION_ITEM_COUNT = len(PRODUCTION_ITEM_NAMES)
        self.PRODUCTION_UNIT_COUNT = len(PRODUCTION_UNIT_NAMES)
        self.ECON_PASS_OFFSET = (
            self.ECON_PRODUCTION_OFFSET + self.max_cities * self.PRODUCTION_ITEM_COUNT
        )
        self.ECON_SIZE = self.ECON_PASS_OFFSET + 1
        self.ACTION_SIZE = self.ECON_OFFSET + self.ECON_SIZE
        self.PASS_ACTION = self.ACTION_SIZE - 1  # last index in econ head
        # Allow multiple actions within the same logical turn; cap to avoid stalling.
        if self.max_actions_per_turn <= 0:
            self.max_actions_per_turn = max(1, self.max_units * 2 + self.max_cities)
        if self.gt is None:
            self.reset()
        if not self.research_done.get(1):
            self.research_done = self._init_research_status()
        self._refresh_tech_costs()

    def _init_research_status(self) -> Dict[Player, Dict[str, bool]]:
        base = {tech: False for tech in self.RESEARCH_TECHS}
        return {1: dict(base), -1: dict(base)}

    def reset(self) -> None:
        self.gt = self.provider.resample()
        self.turn = 0
        self.num_actions = 0
        self.actions_this_turn = 0
        self.winner = None
        self.terminal_reason = None
        self.units = {1: [], -1: []}
        self.cities = {1: [], -1: []}
        self.research_done = self._init_research_status()
        self.research_target = {1: None, -1: None}
        self.research_progress = {1: 0.0, -1: 0.0}
        self.visited = {
            1: np.zeros((self.cfg.map_h, self.cfg.map_w), dtype=bool),
            -1: np.zeros((self.cfg.map_h, self.cfg.map_w), dtype=bool),
        }
        self.kills = {1: 0, -1: 0}
        self.units_built = {1: 0, -1: 0}
        self.future_techs = {1: 0, -1: 0}
        self.scores = {1: 0.0, -1: 0.0}
        self.acted_unit_slots = {1: set(), -1: set()}
        self._refresh_tech_costs()
        self._spawn_units()

    def _refresh_tech_costs(self) -> None:
        style = getattr(self.cfg, "tech_cost_style", TECH_COST_STYLE)
        base_cost = getattr(self.cfg, "base_tech_cost", BASE_TECH_COST)
        min_cost = getattr(self.cfg, "min_tech_cost", MIN_TECH_COST)
        cost_factor = getattr(self.cfg, "tech_cost_factor", TECH_COST_FACTOR)
        self.tech_costs = build_tech_costs(
            TECH_PREREQS,
            style=style,
            base_cost=base_cost,
            min_cost=min_cost,
            cost_factor=cost_factor,
        )

    def _spawn_units(self) -> None:
        assert self.gt is not None
        starts_me = [(0, 0)]
        starts_opp = [(self.cfg.map_w - 1, self.cfg.map_h - 1)]
        spawn_units = ("Settlers", "Workers", "Explorer", "Diplomat")
        missing = [name for name in spawn_units if name not in UNIT_SPECS]
        if missing:
            raise RuntimeError(
                "Missing unit specs for start units: " + ", ".join(missing)
            )
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        mx, my = starts_me[0]
        ox, oy = starts_opp[0]
        for idx, name in enumerate(spawn_units):
            dx, dy = offsets[idx] if idx < len(offsets) else (0, 0)
            spec = UNIT_SPECS[name]
            ux, uy = self._find_spawn_near(mx + dx, my + dy)
            self.units[1].append(
                MHUnit(
                    ux,
                    uy,
                    spec.hp,
                    spec.atk,
                    spec.df,
                    spec.firepower,
                    spec.name,
                    True,
                    spec.can_build_city,
                    None,
                    spec.moves,
                )
            )
            vx, vy = self._find_spawn_near(ox - dx, oy - dy)
            self.units[-1].append(
                MHUnit(
                    vx,
                    vy,
                    spec.hp,
                    spec.atk,
                    spec.df,
                    spec.firepower,
                    spec.name,
                    True,
                    spec.can_build_city,
                    None,
                    spec.moves,
                )
            )
        self._ensure_unit_slots()
        for player in (1, -1):
            for u in self.units[player]:
                if u.alive:
                    self.visited[player][u.y, u.x] = True

    def _ensure_unit_slots(self) -> None:
        for player in (1, -1):
            while len(self.units[player]) < self.max_units:
                self.units[player].append(
                    MHUnit(
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        "None",
                        False,
                        False,
                        None,
                        0,
                    )
                )

    def _find_spawn_near(self, sx: int, sy: int) -> Coord:
        assert self.gt is not None
        sx = int(np.clip(sx, 0, self.cfg.map_w - 1))
        sy = int(np.clip(sy, 0, self.cfg.map_h - 1))
        if self.gt.au_map[sy, sx] == 'A':
            return sx, sy
        frontier = [(sx, sy)]
        seen = {(sx, sy)}
        while frontier:
            x, y = frontier.pop(0)
            if 0 <= x < self.cfg.map_w and 0 <= y < self.cfg.map_h and self.gt.au_map[y, x] == 'A':
                return x, y
            for nx, ny in self.movement.get_native_neighbors(x, y):
                if nx is None:
                    continue
                if (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                frontier.append((nx, ny))
        return sx, sy

    def _city_spacing_ok(self, player: Player, x: int, y: int) -> bool:
        min_dist = getattr(self.cfg, "city_min_distance", 0)
        claim_radius = max(0, int(getattr(self.cfg, "land_claim_radius", 0)))
        if min_dist <= 0 and claim_radius <= 0:
            return True
        if claim_radius > 0:
            candidate_claim = self._city_claim_tiles(x, y, claim_radius)
            for p in (1, -1):
                for city in self.cities[p]:
                    if candidate_claim & self._city_claim_tiles(
                        city.x, city.y, claim_radius
                    ):
                        return False
        if min_dist <= 0:
            return True
        frontier = deque()
        frontier.append((x, y, 0))
        seen = {(x, y)}
        while frontier:
            cx, cy, dist = frontier.popleft()
            if dist >= min_dist:
                continue
            if self._city_at(cx, cy, 1) is not None:
                return False
            if self._city_at(cx, cy, -1) is not None:
                return False
            for nx, ny in self.movement.get_native_neighbors(cx, cy):
                if nx is None or ny is None:
                    continue
                if (nx, ny) in seen:
                    continue
                if self.gt and self.gt.au_map[ny, nx] != "A":
                    continue
                seen.add((nx, ny))
                frontier.append((nx, ny, dist + 1))
        return True

    def duplicate(self) -> "MultiheadState":
        new = MultiheadState.__new__(MultiheadState)
        new.cfg = self.cfg
        new.provider = self.provider
        new.max_units = self.max_units
        new.max_cities = self.max_cities
        new.rng = self.rng
        new.movement = self.movement
        new.gt = self.gt.copy() if self.gt else None
        new.units = {
            p: [
                MHUnit(
                    u.x,
                    u.y,
                    u.hp,
                    u.atk,
                    u.df,
                    u.firepower,
                    u.unit_type,
                    u.alive,
                    u.can_build_city,
                    u.home_city,
                    u.moves_left,
                    u.last_move_turn,
                )
                for u in lst
            ]
            for p, lst in self.units.items()
        }
        new.cities = {
            p: [
                City(
                    x=c.x,
                    y=c.y,
                    size=c.size,
                    food_storage=c.food_storage,
                    production_kind=c.production_kind,
                    production_target=c.production_target,
                    production_progress=c.production_progress,
                    production_cost=c.production_cost,
                    production_per_turn=c.production_per_turn,
                    production_turns=c.production_turns,
                    production_queue=list(c.production_queue),
                    has_city_walls=c.has_city_walls,
                    buildings=set(c.buildings),
                )
                for c in lst
            ]
            for p, lst in self.cities.items()
        }
        new.research_done = {p: dict(flags) for p, flags in self.research_done.items()}
        new.research_target = {
            p: target for p, target in self.research_target.items()
        }
        new.research_progress = {
            p: float(val) for p, val in self.research_progress.items()
        }
        new.visited = {
            1: self.visited[1].copy(),
            -1: self.visited[-1].copy(),
        }
        new.turn = self.turn
        new.num_actions = self.num_actions
        new.actions_this_turn = self.actions_this_turn
        new.max_actions_per_turn = self.max_actions_per_turn
        new.acted_unit_slots = {
            1: set(self.acted_unit_slots.get(1, set())),
            -1: set(self.acted_unit_slots.get(-1, set())),
        }
        new.acted_production_cities = {
            1: set(self.acted_production_cities.get(1, set())),
            -1: set(self.acted_production_cities.get(-1, set())),
        }
        new.kills = dict(self.kills)
        new.units_built = dict(self.units_built)
        new.future_techs = dict(self.future_techs)
        new.scores = dict(self.scores)
        new.winner = self.winner
        new.terminal_reason = self.terminal_reason
        new.tech_costs = dict(self.tech_costs)
        new.RESEARCH_TECHS = self.RESEARCH_TECHS
        new.PRODUCTION_UNIT_NAMES = self.PRODUCTION_UNIT_NAMES
        new.PRODUCTION_BUILDING_NAMES = self.PRODUCTION_BUILDING_NAMES
        new.PRODUCTION_ITEM_NAMES = self.PRODUCTION_ITEM_NAMES
        new.MOVE_PER_UNIT = self.MOVE_PER_UNIT
        new.ATTACK_PER_UNIT = self.ATTACK_PER_UNIT
        new.UNIT_ACTIVITY_NAMES = self.UNIT_ACTIVITY_NAMES
        new.UNIT_ACTIVITY_PER_UNIT = self.UNIT_ACTIVITY_PER_UNIT
        new.MOVE_SIZE = self.MOVE_SIZE
        new.ATTACK_SIZE = self.ATTACK_SIZE
        new.UNIT_ACTIVITY_SIZE = self.UNIT_ACTIVITY_SIZE
        new.UNIT_ACTIVITY_OFFSET = self.UNIT_ACTIVITY_OFFSET
        new.ECON_OFFSET = self.ECON_OFFSET
        new.ECON_SIZE = self.ECON_SIZE
        new.ECON_RESEARCH_OFFSET = self.ECON_RESEARCH_OFFSET
        new.ECON_BUILD_CITY_OFFSET = self.ECON_BUILD_CITY_OFFSET
        new.ECON_PRODUCTION_OFFSET = self.ECON_PRODUCTION_OFFSET
        new.ECON_PASS_OFFSET = self.ECON_PASS_OFFSET
        new.PRODUCTION_ITEM_COUNT = self.PRODUCTION_ITEM_COUNT
        new.PRODUCTION_UNIT_COUNT = self.PRODUCTION_UNIT_COUNT
        new.ACTION_SIZE = self.ACTION_SIZE
        new.PASS_ACTION = self.PASS_ACTION
        return new

    def _unit_max_moves(self, unit: Optional[MHUnit]) -> int:
        if unit is None or not unit.alive:
            return 0
        spec = UNIT_SPECS.get(unit.unit_type)
        if spec is not None:
            return max(0, int(spec.moves))
        return 1

    def valid_moves(self, player: Player) -> np.ndarray:
        moves = np.zeros(self.ACTION_SIZE, dtype=np.int8)
        if self.winner is not None:
            moves[self.PASS_ACTION] = 1
            return moves
        # move head
        acted_slots = self.acted_unit_slots.get(player, set())
        auto_workers = getattr(self.cfg, "auto_worker_units", False)
        for idx in range(self.max_units):
            move_base = idx * self.MOVE_PER_UNIT
            atk_base = self.MOVE_SIZE + idx * self.ATTACK_PER_UNIT
            u = self.units[player][idx] if idx < len(self.units[player]) else None
            if u is None or not u.alive or idx in acted_slots:
                continue
            if auto_workers and self._unit_is_worker_like(u.unit_type):
                continue
            moves[move_base + self.HOLD_DIR] = 1
            if u.moves_left <= 0:
                continue
            neighbors = self.movement.get_native_neighbors(u.x, u.y)
            for dir_idx, (nx, ny) in enumerate(neighbors):
                if dir_idx >= self.MOVE_DIRECTIONS:
                    break
                if nx is None or ny is None:
                    continue
                if self.gt and 0 <= ny < self.cfg.map_h and 0 <= nx < self.cfg.map_w and self.gt.au_map[ny, nx] == 'A':
                    # Freeciv permits friendly stacking inside cities.
                    own_city = self._city_at(nx, ny, player) is not None
                    if self._unit_at(nx, ny, player) is None or own_city:
                        moves[move_base + dir_idx] = 1
                    # attack only if an enemy occupies the target
                    if u.atk > 0 and (
                        self._unit_at(nx, ny, -player) is not None
                        or self._city_at(nx, ny, -player) is not None
                    ):
                        moves[atk_base + dir_idx] = 1
        # research actions (select current target)
        offset = self.ECON_OFFSET
        current_target = self.research_target.get(player)
        if current_target and self.research_done[player].get(current_target, False):
            current_target = None
        if current_target is None:
            for idx, tech in enumerate(self.RESEARCH_TECHS):
                if self.research_done[player].get(tech, False):
                    continue
                prereqs = TECH_PREREQS.get(tech, [])
                if any(
                    not self.research_done[player].get(req, False) for req in prereqs
                ):
                    continue
                moves[offset + idx] = 1
        # build city actions (per unit slot)
        build_offset = offset + self.ECON_BUILD_CITY_OFFSET
        if len(self.cities[player]) < self.max_cities:
            for idx in range(self.max_units):
                u = self.units[player][idx] if idx < len(self.units[player]) else None
                if (
                    u is None
                    or not u.alive
                    or not u.can_build_city
                    or u.moves_left <= 0
                    or idx in acted_slots
                ):
                    continue
                if self._city_at(u.x, u.y, player) is not None:
                    continue
                if not self._city_spacing_ok(player, u.x, u.y):
                    continue
                moves[build_offset + idx] = 1
        # production actions (per city slot)
        prod_offset = offset + self.ECON_PRODUCTION_OFFSET
        acted_prod = self.acted_production_cities.get(player, set())
        for city_idx in range(min(len(self.cities[player]), self.max_cities)):
            if city_idx in acted_prod:
                continue
            city = self.cities[player][city_idx]
            for item_idx, (kind, name) in enumerate(PRODUCTION_ITEM_NAMES):
                if kind == "unit":
                    if not self._unit_unlocked(player, name):
                        continue
                else:
                    if (
                        city.production_kind == "building"
                        and city.production_target == name
                    ):
                        continue
                    if any(
                        queued_kind == "building" and queued_name == name
                        for queued_kind, queued_name in city.production_queue
                    ):
                        continue
                    if not self._building_unlocked(player, city, name):
                        continue
                moves[
                    prod_offset + city_idx * self.PRODUCTION_ITEM_COUNT + item_idx
                ] = 1
        # pass always valid
        moves[self.PASS_ACTION] = 1
        return moves

    def _unit_is_explorer_like(self, unit_name: str) -> bool:
        label = (unit_name or "").lower()
        return any(tag in label for tag in ("explorer", "diplomat", "caravan"))

    def unit_agent_role(self, unit: Optional[MHUnit]) -> str:
        if unit is None:
            return "combat"
        if (
            unit.can_build_city
            or self._unit_is_worker_like(unit.unit_type)
            or self._unit_is_explorer_like(unit.unit_type)
        ):
            return "explore"
        return "combat"

    def slot_agent_role(self, player: Player, slot_idx: int) -> str:
        if slot_idx < 0 or slot_idx >= len(self.units[player]):
            return "combat"
        return self.unit_agent_role(self.units[player][slot_idx])

    def action_agent_role(self, player: Player, action: int) -> Optional[str]:
        if action < 0 or action >= self.ACTION_SIZE or action == self.PASS_ACTION:
            return None
        if action < self.MOVE_SIZE:
            unit_idx = action // self.MOVE_PER_UNIT
            return self.slot_agent_role(player, unit_idx)
        if action < self.MOVE_SIZE + self.ATTACK_SIZE:
            return "combat"

        if action < self.ECON_OFFSET:
            rel = action - self.UNIT_ACTIVITY_OFFSET
            unit_idx = rel // self.UNIT_ACTIVITY_PER_UNIT
            activity_idx = rel % self.UNIT_ACTIVITY_PER_UNIT
            activity_name = self.UNIT_ACTIVITY_NAMES[activity_idx]
            if activity_name in ("fortify", "sentry", "pillage"):
                return "combat"
            return self.slot_agent_role(player, unit_idx)

        econ_idx = action - self.ECON_OFFSET
        if 0 <= econ_idx < len(self.RESEARCH_TECHS):
            return "research"
        if self.ECON_BUILD_CITY_OFFSET <= econ_idx < self.ECON_PRODUCTION_OFFSET:
            return "explore"
        if self.ECON_PRODUCTION_OFFSET <= econ_idx < self.ECON_PASS_OFFSET:
            return "production"
        return None

    def agent_action_masks(
        self,
        player: Player,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        base_mask = (
            self.valid_moves(player)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=np.int8).copy()
        )
        masks = {
            role: np.zeros(self.ACTION_SIZE, dtype=np.int8) for role in self.AGENT_ROLES
        }
        for action in np.flatnonzero(base_mask):
            role = self.action_agent_role(player, int(action))
            if role is None:
                continue
            masks[role][int(action)] = 1
        return masks

    def agent_legal_actions(
        self,
        player: Player,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, List[int]]:
        masks = self.agent_action_masks(player, valid_mask=valid_mask)
        return {
            role: [idx for idx, allowed in enumerate(mask) if allowed]
            for role, mask in masks.items()
        }

    def step(self, player: Player, action: int) -> None:
        if self.winner is not None:
            return
        if action == self.PASS_ACTION or action < 0 or action >= self.ACTION_SIZE:
            self.num_actions += 1
            self._advance_turn()
            self._check_action_limit()
            return

        if action < self.MOVE_SIZE:
            unit_idx = action // self.MOVE_PER_UNIT
            dir_idx = action % self.MOVE_PER_UNIT
            if unit_idx < len(self.units[player]) and self.units[player][unit_idx].alive:
                if dir_idx != self.HOLD_DIR:
                    self._handle_unit_action(player, unit_idx, dir_idx, is_attack=False)
                self.units[player][unit_idx].moves_left = 0
                self.acted_unit_slots.setdefault(player, set()).add(unit_idx)
        elif action < self.MOVE_SIZE + self.ATTACK_SIZE:
            rel = action - self.MOVE_SIZE
            unit_idx = rel // self.ATTACK_PER_UNIT
            dir_idx = rel % self.ATTACK_PER_UNIT
            if unit_idx < len(self.units[player]) and self.units[player][unit_idx].alive:
                self._handle_unit_action(player, unit_idx, dir_idx, is_attack=True)
                self.units[player][unit_idx].moves_left = 0
                self.acted_unit_slots.setdefault(player, set()).add(unit_idx)
        elif action < self.ECON_OFFSET:
            rel = action - self.UNIT_ACTIVITY_OFFSET
            unit_idx = rel // self.UNIT_ACTIVITY_PER_UNIT
            if unit_idx < len(self.units[player]) and self.units[player][unit_idx].alive:
                self.units[player][unit_idx].moves_left = 0
                self.acted_unit_slots.setdefault(player, set()).add(unit_idx)
        else:
            econ_idx = action - self.ECON_OFFSET
            # research
            if 0 <= econ_idx < len(self.RESEARCH_TECHS):
                tech = self.RESEARCH_TECHS[econ_idx]
                if (
                    not self.research_done[player].get(tech, False)
                    and self.research_target.get(player) != tech
                ):
                    prereqs = TECH_PREREQS.get(tech, [])
                    if all(
                        self.research_done[player].get(req, False) for req in prereqs
                    ):
                        self.research_target[player] = tech
                # small bonus hooks can be added later
            # build city (per unit slot)
            elif self.ECON_BUILD_CITY_OFFSET <= econ_idx < self.ECON_PRODUCTION_OFFSET:
                unit_idx = econ_idx - self.ECON_BUILD_CITY_OFFSET
                if unit_idx < len(self.units[player]) and len(self.cities[player]) < self.max_cities:
                    u = self.units[player][unit_idx]
                    if (
                        u.alive
                        and u.can_build_city
                        and u.moves_left > 0
                        and self._city_at(u.x, u.y, player) is None
                        and self._city_spacing_ok(player, u.x, u.y)
                    ):
                        u.moves_left = 0
                        u.alive = False
                        self._add_city(player, u.x, u.y)
                        self.scores[player] += self.cfg.build_city_reward
                        self.acted_unit_slots.setdefault(player, set()).add(unit_idx)
            # production selection
            elif self.ECON_PRODUCTION_OFFSET <= econ_idx < self.ECON_PASS_OFFSET:
                rel = econ_idx - self.ECON_PRODUCTION_OFFSET
                city_slot = rel // self.PRODUCTION_ITEM_COUNT
                item_idx = rel % self.PRODUCTION_ITEM_COUNT
                if city_slot < len(self.cities[player]):
                    city = self.cities[player][city_slot]
                    kind, name = PRODUCTION_ITEM_NAMES[item_idx]
                    queue_limit = getattr(self.cfg, "production_queue_max", 0)
                    queue_add = max(1, int(getattr(self.cfg, "production_queue_add", 1)))
                    add_count = 1 if kind == "building" else queue_add
                    if kind == "unit":
                        if self._unit_unlocked(player, name):
                            if (
                                city.production_kind != "unit"
                                or city.production_target != name
                            ):
                                city.production_kind = "unit"
                                city.production_target = name
                                city.production_progress = 0.0
                    elif city.production_target:
                        if (
                            city.production_kind == "building"
                            and city.production_target == name
                        ):
                            pass
                        elif any(
                            queued_kind == "building" and queued_name == name
                            for queued_kind, queued_name in city.production_queue
                        ):
                            pass
                        else:
                            if queue_limit <= 0 or len(city.production_queue) < queue_limit:
                                city.production_queue.append((kind, name))
                    else:
                        if self._building_unlocked(player, city, name):
                            city.production_kind = "building"
                            city.production_target = name
                            city.production_progress = 0.0
                            for _ in range(add_count - 1):
                                if queue_limit > 0 and len(city.production_queue) >= queue_limit:
                                    break
                                city.production_queue.append((kind, name))
                    self.acted_production_cities.setdefault(player, set()).add(city_slot)
        self._resolve_terminal()
        # Stay in the same turn unless we exceed the per-turn action cap.
        self.actions_this_turn += 1
        self.num_actions += 1
        if self.actions_this_turn >= self.max_actions_per_turn:
            self._advance_turn()
        self._check_action_limit()

    def _handle_unit_action(
        self, player: Player, unit_idx: int, dir_idx: int, is_attack: bool
    ) -> None:
        if unit_idx >= len(self.units[player]):
            return
        u = self.units[player][unit_idx]
        if not u.alive:
            return
        neighbors = self.movement.get_native_neighbors(u.x, u.y)
        nx, ny = neighbors[dir_idx]
        if nx is None or ny is None or not self.gt or self.gt.au_map[ny, nx] != 'A':
            return
        if is_attack:
            if u.atk <= 0:
                return
            attack_reward = getattr(self.cfg, "attack_reward", 0.0)
            enemy = self._unit_at(nx, ny, -player)
            if enemy:
                if attack_reward:
                    self.scores[player] += attack_reward
                    attack_reward = 0.0
                self._attack(player, u, enemy, -player)
                if enemy.alive:
                    return
                if not u.alive:
                    return
                city_idx = self._city_at_index(nx, ny, -player)
                if city_idx is not None:
                    if attack_reward:
                        self.scores[player] += attack_reward
                        attack_reward = 0.0
                    self._attack_city(player, u, -player, city_idx)
            else:
                city_idx = self._city_at_index(nx, ny, -player)
                if city_idx is not None:
                    if attack_reward:
                        self.scores[player] += attack_reward
                        attack_reward = 0.0
                    self._attack_city(player, u, -player, city_idx)
        else:
            own_city = self._city_at(nx, ny, player) is not None
            if self._unit_at(nx, ny, player) is None or own_city:
                u.x, u.y = nx, ny
                u.last_move_turn = self.turn
                if not self.visited[player][ny, nx]:
                    self.visited[player][ny, nx] = True
                    self.scores[player] += self.cfg.move_reward

    def _attack(
        self,
        player: Player,
        attacker: MHUnit,
        defender: MHUnit,
        defender_player: Player,
    ) -> None:
        if attacker.atk <= 0:
            return
        atk = max(1, attacker.atk)
        df = max(1, defender.df)
        city = self._city_at(defender.x, defender.y, defender_player)
        if city is not None:
            df *= max(0.1, getattr(self.cfg, "city_defense_multiplier", 1.0))
            if city.has_city_walls:
                df *= max(
                    0.1, getattr(self.cfg, "city_walls_defense_multiplier", 1.0)
                )
        fatigue_mult = getattr(self.cfg, "move_fatigue_defense_multiplier", 1.0)
        if (
            fatigue_mult != 1.0
            and self.turn > 0
            and defender.last_move_turn == self.turn - 1
        ):
            df = max(0.1, df * fatigue_mult)
        p_hit = atk / float(atk + df)
        while attacker.hp > 0 and defender.hp > 0:
            if self.rng.random() < p_hit:
                defender.hp -= max(1, attacker.firepower)
            else:
                attacker.hp -= max(1, defender.firepower)

        if defender.hp <= 0:
            defender.alive = False
            self.kills[player] += 1
            self.scores[player] += self.cfg.elimination_bonus
            self.scores[-player] -= self.cfg.elimination_bonus
        if attacker.hp <= 0:
            attacker.alive = False

    def _attack_city(
        self,
        player: Player,
        attacker: MHUnit,
        defender: Player,
        city_idx: int,
    ) -> None:
        if attacker.atk <= 0:
            return
        if city_idx < 0 or city_idx >= len(self.cities[defender]):
            return
        city = self.cities[defender][city_idx]
        if self._unit_at(city.x, city.y, defender) is not None:
            return
        self._remove_city(defender, city_idx)
        self.scores[player] += self.cfg.city_capture_reward
        self.scores[-player] -= self.cfg.city_capture_reward

    def _unit_at(self, x: int, y: int, player: Player) -> Optional[MHUnit]:
        for u in self.units[player]:
            if u.alive and u.x == x and u.y == y:
                return u
        return None

    def _city_at(self, x: int, y: int, player: Player) -> Optional[City]:
        for city in self.cities[player]:
            if city.x == x and city.y == y:
                return city
        return None

    def _city_at_index(self, x: int, y: int, player: Player) -> Optional[int]:
        for idx, city in enumerate(self.cities[player]):
            if city.x == x and city.y == y:
                return idx
        return None

    def _unit_unlocked(self, player: Player, unit_name: str) -> bool:
        techs = UNIT_TECHS.get(unit_name, [])
        if not techs:
            return True
        return all(self.research_done[player].get(tech, False) for tech in techs)

    def _unit_strength_score(self, spec: UnitSpec) -> int:
        return (
            spec.atk * 100
            + spec.df * 90
            + spec.hp * 10
            + spec.firepower * 50
            + spec.moves * 20
        )

    def _unit_is_obsolete_exempt(self, unit_name: str) -> bool:
        label = (unit_name or "").lower()
        return any(tag in label for tag in ("settler", "worker", "engineer", "migrant"))

    def _unit_is_worker_like(self, unit_name: str) -> bool:
        label = (unit_name or "").lower()
        return any(tag in label for tag in ("worker", "engineer", "migrant"))

    def _unit_is_sea(self, unit_name: str) -> bool:
        spec = UNIT_SPECS.get(unit_name)
        if spec is None:
            return False
        return spec.unit_class.lower() in SEA_UNIT_CLASSES

    def _unit_is_excluded(self, unit_name: str) -> bool:
        if not getattr(self.cfg, "allow_sea_units", True) and self._unit_is_sea(unit_name):
            return True
        label = (unit_name or "").lower()
        return any(tag in label for tag in ("diplomat", "explorer"))

    def _player_has_worker_like(self, player: Player) -> bool:
        return any(
            u.alive and self._unit_is_worker_like(u.unit_type) for u in self.units[player]
        )

    def _best_unlocked_in_chain(
        self, player: Player, chain: Tuple[str, ...]
    ) -> Optional[str]:
        best = None
        for name in chain:
            if name in UNIT_SPECS and self._unit_unlocked(player, name):
                if self._unit_is_excluded(name):
                    continue
                best = name
        return best

    def _unit_best_upgrade(self, player: Player, unit_name: str) -> str:
        candidates: list[str] = []
        for chain in UNIT_GENERATION_CHAINS:
            if unit_name not in chain:
                continue
            best = self._best_unlocked_in_chain(player, chain)
            if best:
                candidates.append(best)
        if not candidates:
            return unit_name

        def score(name: str) -> int:
            spec = UNIT_SPECS.get(name)
            if spec is None:
                return -1
            return self._unit_strength_score(spec)

        return max(candidates, key=score)

    def _unit_obsolete(self, player: Player, unit_name: str) -> bool:
        if self._unit_is_obsolete_exempt(unit_name):
            return False
        if self._unit_best_upgrade(player, unit_name) != unit_name:
            return True
        obsolete_by = UNIT_OBSOLETE_BY.get(unit_name)
        if not obsolete_by:
            return False
        return self._unit_unlocked(player, obsolete_by)

    def _upgrade_unit_name(self, player: Player, unit_name: str) -> str:
        upgraded = self._unit_best_upgrade(player, unit_name)
        if upgraded != unit_name:
            return upgraded
        seen = {unit_name}
        while True:
            obsolete_by = UNIT_OBSOLETE_BY.get(unit_name)
            if not obsolete_by or obsolete_by in seen:
                break
            if not self._unit_unlocked(player, obsolete_by):
                break
            unit_name = obsolete_by
            seen.add(unit_name)
        return unit_name

    def _refresh_production_queue(self, player: Player) -> None:
        for city in self.cities[player]:
            updated: list[tuple[str, str]] = []
            changed = False
            for kind, name in city.production_queue:
                if kind == "unit":
                    upgraded = self._upgrade_unit_name(player, name)
                    if upgraded != name:
                        name = upgraded
                        changed = True
                updated.append((kind, name))
            if changed:
                city.production_queue = updated
            current_kind = city.production_kind
            current_name = city.production_target
            if current_kind is None and current_name is None and city.production_queue:
                current_kind, current_name = city.production_queue[0]
            if current_kind != "unit" or not current_name:
                continue
            upgraded = self._upgrade_unit_name(player, current_name)
            if upgraded == current_name:
                continue
            existing_units = {
                name for kind, name in city.production_queue if kind == "unit"
            }
            if upgraded in existing_units:
                continue
            queue_limit = getattr(self.cfg, "production_queue_max", 0)
            if queue_limit > 0 and len(city.production_queue) >= queue_limit:
                continue
            city.production_queue.append(("unit", upgraded))

    def _building_is_palace(self, building_name: str) -> bool:
        return "palace" in (building_name or "").lower()

    def _building_allowed_by_wonder_policy(self, building_name: str) -> bool:
        if building_name not in GREAT_WONDER_NAMES:
            return True
        allowlist = getattr(self.cfg, "wonder_production_allowlist", ())
        if allowlist:
            return building_name in allowlist
        blocklist = getattr(self.cfg, "wonder_production_blocklist", ())
        return building_name not in blocklist

    def _building_unlocked(
        self, player: Player, city: City, building_name: str
    ) -> bool:
        if self._building_is_palace(building_name):
            return False
        if not self._building_allowed_by_wonder_policy(building_name):
            return False
        if building_name in city.buildings:
            return False
        if (building_name or "").lower() == "aqueduct":
            if not self.research_done[player].get("Construction", False):
                return False
        lowered = (building_name or "").lower()
        if lowered.startswith("aqueduct") and ("river" in lowered or "lake" in lowered):
            return False
        techs = BUILDING_TECHS.get(building_name, [])
        if techs and not all(
            self.research_done[player].get(tech, False) for tech in techs
        ):
            return False
        req_buildings = BUILDING_REQ_BUILDINGS.get(building_name, [])
        if req_buildings and not all(req in city.buildings for req in req_buildings):
            return False
        return True

    def _city_unit_count(self, player: Player, city_idx: int) -> int:
        count = sum(
            1
            for u in self.units[player]
            if u.alive and u.home_city == city_idx
        )
        if city_idx < len(self.cities[player]):
            city = self.cities[player][city_idx]
            tile_count = sum(
                1
                for u in self.units[player]
                if u.alive and u.home_city is None and u.x == city.x and u.y == city.y
            )
            if tile_count > count:
                count = tile_count
        return count

    def _add_city(
        self, player: Player, x: int, y: int, *, has_city_walls: bool = False
    ) -> None:
        if len(self.cities[player]) >= self.max_cities:
            return
        city = City(x=x, y=y, has_city_walls=has_city_walls)
        if has_city_walls:
            city.buildings.add("City Walls")
        self.cities[player].append(city)

    def _remove_city(self, player: Player, city_idx: int) -> None:
        if city_idx < 0 or city_idx >= len(self.cities[player]):
            return
        del self.cities[player][city_idx]
        for u in self.units[player]:
            if not u.alive or u.home_city is None:
                continue
            if u.home_city == city_idx:
                u.home_city = None
            elif u.home_city > city_idx:
                u.home_city -= 1

    def _place_unit(
        self, player: Player, unit: MHUnit, city_idx: Optional[int]
    ) -> bool:
        for slot in self.units[player]:
            if not slot.alive:
                slot.x = unit.x
                slot.y = unit.y
                slot.hp = unit.hp
                slot.atk = unit.atk
                slot.df = unit.df
                slot.firepower = unit.firepower
                slot.unit_type = unit.unit_type
                slot.alive = True
                slot.can_build_city = unit.can_build_city
                slot.home_city = city_idx
                slot.moves_left = unit.moves_left
                slot.last_move_turn = unit.last_move_turn
                return True
        if len(self.units[player]) < self.max_units:
            unit.home_city = city_idx
            self.units[player].append(unit)
            return True
        return False

    def _spawn_from_city(self, player: Player, city_idx: int, unit_name: str) -> bool:
        if city_idx >= len(self.cities[player]):
            return False
        if self._city_unit_count(player, city_idx) >= self.cfg.city_unit_cap:
            return False
        spec = UNIT_SPECS.get(unit_name)
        if spec is None:
            return False
        city = self.cities[player][city_idx]
        candidates = [(city.x, city.y)] + [
            (nx, ny)
            for nx, ny in self.movement.get_native_neighbors(city.x, city.y)
            if nx is not None and ny is not None
        ]
        for cx, cy in candidates:
            if (
                self._unit_at(cx, cy, player) is None
                and self.gt
                and 0 <= cy < self.cfg.map_h
                and 0 <= cx < self.cfg.map_w
                and self.gt.au_map[cy, cx] == 'A'
            ):
                unit = MHUnit(
                    cx,
                    cy,
                    spec.hp,
                    spec.atk,
                    spec.df,
                    spec.firepower,
                    spec.name,
                    True,
                    spec.can_build_city,
                    city_idx,
                    spec.moves,
                )
                if self._place_unit(player, unit, city_idx):
                    self.units_built[player] = self.units_built.get(player, 0) + 1
                    return True
                return False
        return False

    def _apply_city_economy(self) -> None:
        bulbs_by_player = {1: 0.0, -1: 0.0}
        for player in (1, -1):
            for city_idx, city in enumerate(self.cities[player]):
                size = max(1, city.size)
                if city.production_target is None and city.production_queue:
                    kind, name = city.production_queue.pop(0)
                    city.production_kind = kind
                    city.production_target = name
                unit_count = self._city_unit_count(player, city_idx)
                gold_start = getattr(self.cfg, "unit_upkeep_gold_start", 0)
                gold_cost = getattr(self.cfg, "unit_upkeep_gold_cost", 0)
                food_start = getattr(self.cfg, "unit_upkeep_food_start", 0)
                food_cost = getattr(self.cfg, "unit_upkeep_food_cost", 0)
                gold_upkeep = 0
                food_upkeep = 0
                if gold_start > 0 and gold_cost > 0 and unit_count >= gold_start:
                    gold_upkeep = (unit_count - gold_start + 1) * gold_cost
                if food_start > 0 and food_cost > 0 and unit_count >= food_start:
                    food_upkeep = (unit_count - food_start + 1) * food_cost
                total_food = self.cfg.city_food + self.cfg.grass_food * size - food_upkeep
                total_shields = (
                    self.cfg.city_shield + self.cfg.grass_shield * size
                ) * getattr(self.cfg, "production_rate_multiplier", 1.0)
                total_trade = self.cfg.city_trade + self.cfg.grass_trade * size - gold_upkeep
                if total_trade < 0:
                    total_trade = 0.0
                science_rate = getattr(self.cfg, "tax_science_rate", 0.0)
                bulbs_by_player[player] += (
                    total_trade
                    * science_rate
                    * getattr(self.cfg, "research_rate_multiplier", 1.0)
                )

                food_surplus = total_food - self.cfg.food_consumption * size
                if food_surplus > 0:
                    city.food_storage += food_surplus
                    if city.food_storage >= self.cfg.food_growth:
                        city.food_storage -= self.cfg.food_growth
                        city.size += 1
                else:
                    city.food_storage = max(0.0, city.food_storage + food_surplus)

                city.production_progress += total_shields
                if city.production_target and city.production_kind == "unit":
                    spec = UNIT_SPECS.get(city.production_target)
                    if spec and city.production_progress >= spec.cost:
                        if self._spawn_from_city(
                            player, city_idx, city.production_target
                        ):
                            city.production_progress -= spec.cost
                            if city.production_target == "Settlers":
                                pop_cost = int(getattr(self.cfg, "settler_population_cost", 1))
                                city.size = max(1, city.size - pop_cost)
                            if city.production_queue:
                                kind, name = city.production_queue.pop(0)
                                city.production_kind = kind
                                city.production_target = name
                elif city.production_target and city.production_kind == "building":
                    spec = BUILDING_SPECS.get(city.production_target)
                    if (
                        spec
                        and city.production_progress >= spec.cost
                        and city.production_target not in city.buildings
                    ):
                        city.buildings.add(city.production_target)
                        if city.production_target == "City Walls":
                            city.has_city_walls = True
                        city.production_progress -= spec.cost
                        if city.production_queue:
                            kind, name = city.production_queue.pop(0)
                            city.production_kind = kind
                            city.production_target = name
                        else:
                            city.production_kind = None
                            city.production_target = None

                # Research bulbs are added from trade; completion handled below.
        self._apply_research_bulbs(bulbs_by_player)

    def _apply_research_bulbs(self, bulbs_by_player: Dict[Player, float]) -> None:
        for player, bulbs in bulbs_by_player.items():
            if bulbs != 0:
                self.research_progress[player] = (
                    self.research_progress.get(player, 0.0) + bulbs
                )
            target = self.research_target.get(player)
            if not target:
                continue
            if self.research_done[player].get(target, False):
                self.research_target[player] = None
                continue
            cost = self._research_cost(player, target)
            if self.research_progress[player] >= cost:
                self.research_progress[player] -= cost
                self.research_done[player][target] = True
                reward = self.cfg.research_reward_map.get(
                    target, self.cfg.research_reward
                )
                self.scores[player] += reward
                self._refresh_production_queue(player)
                self.research_target[player] = None

    @staticmethod
    def _normalize_tech_cost_style(style: str) -> str:
        return ''.join(ch for ch in (style or "").lower() if ch.isalnum())

    @staticmethod
    def _round_cost(value: float) -> int:
        if value <= 0:
            return 0
        return int(value + 0.5)

    def _techs_researched(self, player: Player) -> int:
        return sum(1 for done in self.research_done[player].values() if done)

    def _research_cost(self, player: Player, target: str) -> float:
        style = self._normalize_tech_cost_style(getattr(self.cfg, "tech_cost_style", ""))
        if style in {"civ1civ2", "civiii"}:
            techs = max(1, self._techs_researched(player))
            base = getattr(self.cfg, "base_tech_cost", 10.0) * techs
            factor = getattr(self.cfg, "tech_cost_factor", 1.0)
            sciencebox = getattr(self.cfg, "sciencebox", 100)
            cost = base * factor * (sciencebox / 100.0)
            min_cost = getattr(self.cfg, "min_tech_cost", 0.0)
            if min_cost > 0 and cost < min_cost:
                cost = min_cost
            return self._round_cost(cost)
        cost = TECH_COSTS.get(target, 0.0)
        if self.tech_costs:
            cost = self.tech_costs.get(target, cost)
        if cost <= 0:
            cost = getattr(self.cfg, "base_tech_cost", 10.0)
        sciencebox = getattr(self.cfg, "sciencebox", 100)
        return self._round_cost(cost * (sciencebox / 100.0))

    def _resolve_terminal(self) -> None:
        alive_me = any(u.alive for u in self.units[1]) or bool(self.cities[1])
        alive_opp = any(u.alive for u in self.units[-1]) or bool(self.cities[-1])
        if alive_me and not alive_opp:
            self.winner = 1
            self.terminal_reason = "eliminate_opp"
        elif alive_opp and not alive_me:
            self.winner = -1
            self.terminal_reason = "eliminate_me"
        elif not alive_me and not alive_opp:
            self.winner = 0
            self.terminal_reason = "mutual_destruction"

    def _check_action_limit(self) -> None:
        if self.winner is None and self.num_actions >= self.cfg.max_num_actions:
            self.winner = 0
            self.terminal_reason = "max_num_actions"

    def _advance_turn(self) -> None:
        self._apply_city_economy()
        self.turn += 1
        self.actions_this_turn = 0
        self.acted_unit_slots = {1: set(), -1: set()}
        self.acted_production_cities = {1: set(), -1: set()}
        self.acted_production_cities = {1: set(), -1: set()}
        for units in self.units.values():
            for unit in units:
                if unit.alive:
                    unit.moves_left = self._unit_max_moves(unit)

    def _alive_count(self, player: Player) -> int:
        return sum(1 for u in self.units[player] if u.alive)

    def _hp_sum(self, player: Player) -> int:
        return sum(u.hp for u in self.units[player] if u.alive)

    def _tile_is_land(self, x: int, y: int) -> bool:
        if self.gt is None:
            return False
        if x < 0 or y < 0 or x >= self.cfg.map_w or y >= self.cfg.map_h:
            return False
        return self.gt.au_map[y, x] == "A"

    def _city_claim_tiles(self, x: int, y: int, radius: int) -> set[tuple[int, int]]:
        if self.movement is None or radius < 0:
            return set()
        if not self._tile_is_land(x, y):
            return set()
        if radius == 0:
            return {(x, y)}
        claimed: set[tuple[int, int]] = set()
        frontier = deque([(x, y, 0)])
        seen = {(x, y)}
        while frontier:
            cx, cy, dist = frontier.popleft()
            if self._tile_is_land(cx, cy):
                claimed.add((cx, cy))
            if dist >= radius:
                continue
            for nx, ny in self.movement.get_native_neighbors(cx, cy):
                if nx is None or ny is None:
                    continue
                if (nx, ny) in seen:
                    continue
                if not self._tile_is_land(nx, ny):
                    continue
                seen.add((nx, ny))
                frontier.append((nx, ny, dist + 1))
        return claimed

    def _land_area(self, player: Player) -> int:
        radius = max(0, int(getattr(self.cfg, "land_claim_radius", 0)))
        if radius <= 0:
            claimed = set()
            for city in self.cities[player]:
                claimed.add((city.x, city.y))
            return len(claimed)
        claimed: set[tuple[int, int]] = set()
        for city in self.cities[player]:
            claimed.update(self._city_claim_tiles(city.x, city.y, radius))
        return len(claimed)

    def muzero_score(self, player: Player) -> float:
        """Observable proxy for Freeciv's server-side civilization score.

        Culture and spaceship state are not exposed in MultiheadState, so the
        authoritative score must still come from Player:score_game().
        """
        citizens = sum(city.size for city in self.cities[player])
        techs = sum(1 for done in self.research_done[player].values() if done)
        future = self.future_techs.get(player, 0)
        wonders = sum(
            1
            for city in self.cities[player]
            for building in city.buildings
            if building in GREAT_WONDER_NAMES
        )
        units_built = self.units_built.get(player, 0)
        kills = self.kills.get(player, 0)
        return float(
            citizens
            + techs * 2
            + future * 5
            + wonders * 5
            + units_built // 10
            + kills // 3
        )

    def civilization_score(self, player: Player) -> float:
        """Backward-compatible alias for the MuZero proxy score."""
        return self.muzero_score(player)

    def heuristic_score(self, player: Player) -> int:
        """
        Heuristic tiebreak when no explicit winner:
        1) kill diff
        2) alive unit count diff
        3) hp sum diff
        returns +1 if player leads, -1 if behind, 0 if equal.
        """
        opp = -player
        kd = self.kills[player] - self.kills[opp]
        if kd != 0:
            return 1 if kd > 0 else -1
        ad = self._alive_count(player) - self._alive_count(opp)
        if ad != 0:
            return 1 if ad > 0 else -1
        hd = self._hp_sum(player) - self._hp_sum(opp)
        if hd != 0:
            return 1 if hd > 0 else -1
        cd = len(self.cities[player]) - len(self.cities[opp])
        if cd != 0:
            return 1 if cd > 0 else -1
        return 0

    # ---------- encodings ----------
    def encode(self, perspective: Player) -> np.ndarray:
        assert self.gt is not None
        me = perspective
        opp = -perspective
        channels: List[np.ndarray] = []
        channels.append((self.gt.au_map == 'A').astype(np.float32))
        channels.append((self.gt.au_map == 'U').astype(np.float32))
        revealed = (self.gt.au_map != "U").astype(np.float32)
        visited = self.visited.get(perspective)
        channels.append(revealed)
        channels.append(
            visited.astype(np.float32)
            if visited is not None
            else np.zeros_like(channels[0])
        )
        terrain_food = np.zeros_like(channels[0])
        terrain_shield = np.zeros_like(channels[0])
        terrain_trade = np.zeros_like(channels[0])
        terrain_move_cost = np.zeros_like(channels[0])
        terrain_defense = np.zeros_like(channels[0])
        if self.gt.terrain_map is not None:
            for terrain_name, spec in TERRAIN_SPECS.items():
                mask = np.char.lower(self.gt.terrain_map.astype(str)) == terrain_name
                terrain_food[mask] = spec.food / max(1, _MAX_TERRAIN_FOOD)
                terrain_shield[mask] = spec.shield / max(1, _MAX_TERRAIN_SHIELD)
                terrain_trade[mask] = spec.trade / max(1, _MAX_TERRAIN_TRADE)
                terrain_move_cost[mask] = spec.movement_cost / max(1, _MAX_TERRAIN_MOVE)
                terrain_defense[mask] = spec.defense_bonus / 100.0
        channels.extend(
            [terrain_food, terrain_shield, terrain_trade, terrain_move_cost, terrain_defense]
        )
        unit_me = np.zeros_like(channels[0])
        unit_opp = np.zeros_like(channels[0])
        hp_me = np.zeros_like(channels[0])
        hp_opp = np.zeros_like(channels[0])
        moves_left_me = np.zeros_like(channels[0])
        moves_left_opp = np.zeros_like(channels[0])
        fatigue_me = np.zeros_like(channels[0])
        fatigue_opp = np.zeros_like(channels[0])
        city_me = np.zeros_like(channels[0])
        city_opp = np.zeros_like(channels[0])
        city_size_me = np.zeros_like(channels[0])
        city_size_opp = np.zeros_like(channels[0])
        city_walls_me = np.zeros_like(channels[0])
        city_walls_opp = np.zeros_like(channels[0])
        production_progress_me = np.zeros_like(channels[0])
        production_progress_opp = np.zeros_like(channels[0])
        production_queue_len_me = np.zeros_like(channels[0])
        production_queue_len_opp = np.zeros_like(channels[0])
        production_current_me = [
            np.zeros_like(channels[0]) for _ in PRODUCTION_ITEM_NAMES
        ]
        production_current_opp = [
            np.zeros_like(channels[0]) for _ in PRODUCTION_ITEM_NAMES
        ]
        production_next_me = [
            np.zeros_like(channels[0]) for _ in PRODUCTION_ITEM_NAMES
        ]
        production_next_opp = [
            np.zeros_like(channels[0]) for _ in PRODUCTION_ITEM_NAMES
        ]
        unit_type_me = [np.zeros_like(channels[0]) for _ in UNIT_TYPE_NAMES]
        unit_type_opp = [np.zeros_like(channels[0]) for _ in UNIT_TYPE_NAMES]
        unit_rule_me = [np.zeros_like(channels[0]) for _ in range(8)]
        unit_rule_opp = [np.zeros_like(channels[0]) for _ in range(8)]
        unit_reachable_me = [np.zeros_like(channels[0]) for _ in range(self.max_units)]
        unit_city_site_reachable_me = [
            np.zeros_like(channels[0]) for _ in range(self.max_units)
        ]
        fatigue_turn = self.turn - 1
        for unit_slot, u in enumerate(self.units[me][: self.max_units]):
            if not u.alive:
                continue
            unit_me[u.y, u.x] = 1.0
            hp_me[u.y, u.x] = u.hp / 20.0
            max_moves = max(1, self._unit_max_moves(u))
            moves_left_me[u.y, u.x] = min(1.0, float(u.moves_left) / float(max_moves))
            unit_type_idx = UNIT_TYPE_INDEX.get(u.unit_type)
            if unit_type_idx is not None:
                unit_type_me[unit_type_idx][u.y, u.x] = 1.0
            self._encode_unit_rule(u, unit_rule_me)
            self._encode_unit_reachable(
                me,
                u,
                unit_reachable_me[unit_slot],
                unit_city_site_reachable_me[unit_slot],
            )
            if self.turn > 0 and u.last_move_turn == fatigue_turn:
                fatigue_me[u.y, u.x] = 1.0
        for u in self.units[opp]:
            if not u.alive:
                continue
            unit_opp[u.y, u.x] = 1.0
            hp_opp[u.y, u.x] = u.hp / 20.0
            max_moves = max(1, self._unit_max_moves(u))
            moves_left_opp[u.y, u.x] = min(1.0, float(u.moves_left) / float(max_moves))
            unit_type_idx = UNIT_TYPE_INDEX.get(u.unit_type)
            if unit_type_idx is not None:
                unit_type_opp[unit_type_idx][u.y, u.x] = 1.0
            self._encode_unit_rule(u, unit_rule_opp)
            if self.turn > 0 and u.last_move_turn == fatigue_turn:
                fatigue_opp[u.y, u.x] = 1.0
        for c in self.cities[me]:
            city_me[c.y, c.x] = 1.0
            city_size_me[c.y, c.x] = min(
                1.0, float(c.size) / max(1.0, self.cfg.city_size_norm)
            )
            if c.has_city_walls:
                city_walls_me[c.y, c.x] = 1.0
            self._encode_city_production(
                c,
                production_progress_me,
                production_queue_len_me,
                production_current_me,
                production_next_me,
            )
        for c in self.cities[opp]:
            city_opp[c.y, c.x] = 1.0
            city_size_opp[c.y, c.x] = min(
                1.0, float(c.size) / max(1.0, self.cfg.city_size_norm)
            )
            if c.has_city_walls:
                city_walls_opp[c.y, c.x] = 1.0
            self._encode_city_production(
                c,
                production_progress_opp,
                production_queue_len_opp,
                production_current_opp,
                production_next_opp,
            )
        channels.append(unit_me)
        channels.append(unit_opp)
        channels.append(hp_me)
        channels.append(hp_opp)
        channels.append(moves_left_me)
        channels.append(moves_left_opp)
        channels.append(fatigue_me)
        channels.append(fatigue_opp)
        channels.append(city_me)
        channels.append(city_opp)
        channels.append(city_size_me)
        channels.append(city_size_opp)
        channels.append(city_walls_me)
        channels.append(city_walls_opp)
        channels.append(production_progress_me)
        channels.append(production_progress_opp)
        channels.append(production_queue_len_me)
        channels.append(production_queue_len_opp)
        channels.extend(production_current_me)
        channels.extend(production_current_opp)
        channels.extend(production_next_me)
        channels.extend(production_next_opp)
        channels.extend(unit_type_me)
        channels.extend(unit_type_opp)
        channels.extend(unit_rule_me)
        channels.extend(unit_rule_opp)
        channels.extend(unit_reachable_me)
        channels.extend(unit_city_site_reachable_me)
        # research planes
        for tech in self.RESEARCH_TECHS:
            tme = np.full_like(unit_me, 1.0 if self.research_done[me].get(tech, False) else 0.0)
            topp = np.full_like(unit_me, 1.0 if self.research_done[opp].get(tech, False) else 0.0)
            channels.append(tme)
            channels.append(topp)
            prereqs = TECH_PREREQS.get(tech, [])
            available_me = not self.research_done[me].get(tech, False) and all(
                self.research_done[me].get(req, False) for req in prereqs
            )
            available_opp = not self.research_done[opp].get(tech, False) and all(
                self.research_done[opp].get(req, False) for req in prereqs
            )
            channels.append(np.full_like(unit_me, float(available_me)))
            channels.append(np.full_like(unit_me, float(available_opp)))
        turn_plane = np.full_like(
            unit_me,
            min(float(self.turn) / float(max(1, self.cfg.max_turns)), 1.0),
        )
        channels.append(turn_plane)
        progress_plane = np.full_like(
            unit_me,
            min(self.num_actions / max(1, self.cfg.max_num_actions), 1.0),
        )
        channels.append(progress_plane)
        return np.stack(channels, axis=0)

    def _encode_unit_rule(self, unit: MHUnit, planes: List[np.ndarray]) -> None:
        spec = UNIT_SPECS.get(unit.unit_type)
        if spec is None:
            return
        values = (
            spec.atk / max(1, _MAX_UNIT_ATTACK),
            spec.df / max(1, _MAX_UNIT_DEFENSE),
            spec.hp / max(1, _MAX_UNIT_HP),
            spec.firepower / max(1, _MAX_UNIT_FIREPOWER),
            spec.moves / max(1, _MAX_UNIT_MOVES),
            spec.cost / max(1, _MAX_UNIT_COST),
            float(spec.can_build_city),
            float(bool(spec.obsolete_by)),
        )
        for plane, value in zip(planes, values):
            plane[unit.y, unit.x] = value

    def _encode_unit_reachable(
        self,
        player: Player,
        unit: MHUnit,
        reachable_plane: np.ndarray,
        city_site_plane: np.ndarray,
    ) -> None:
        if not unit.alive:
            return
        max_steps = max(0, int(unit.moves_left))
        if max_steps <= 0:
            reachable_plane[unit.y, unit.x] = 1.0
            if unit.can_build_city and self._can_found_city_at(player, unit.x, unit.y):
                city_site_plane[unit.y, unit.x] = 1.0
            return

        seen = {(unit.x, unit.y)}
        frontier = [(unit.x, unit.y)]
        reachable_plane[unit.y, unit.x] = 1.0
        if unit.can_build_city and self._can_found_city_at(player, unit.x, unit.y):
            city_site_plane[unit.y, unit.x] = 1.0
        for _depth in range(max_steps):
            next_frontier: list[Coord] = []
            for x, y in frontier:
                for nx, ny in self.movement.get_native_neighbors(x, y):
                    if nx is None or ny is None or (nx, ny) in seen:
                        continue
                    if not (0 <= nx < self.cfg.map_w and 0 <= ny < self.cfg.map_h):
                        continue
                    if self.gt is not None and self.gt.au_map[ny, nx] != "A":
                        continue
                    own_city = self._city_at(nx, ny, player) is not None
                    if self._unit_at(nx, ny, player) is not None and not own_city:
                        continue
                    seen.add((nx, ny))
                    next_frontier.append((nx, ny))
                    reachable_plane[ny, nx] = 1.0
                    if unit.can_build_city and self._can_found_city_at(player, nx, ny):
                        city_site_plane[ny, nx] = 1.0
            frontier = next_frontier
            if not frontier:
                break

    def _can_found_city_at(self, player: Player, x: int, y: int) -> bool:
        if len(self.cities[player]) >= self.max_cities:
            return False
        if self._city_at(x, y, player) is not None:
            return False
        return self._city_spacing_ok(player, x, y)

    def _encode_city_production(
        self,
        city: City,
        progress_plane: np.ndarray,
        queue_len_plane: np.ndarray,
        current_planes: List[np.ndarray],
        next_planes: List[np.ndarray],
    ) -> None:
        queue_limit = max(1, int(getattr(self.cfg, "production_queue_max", 1)))
        queue_len_plane[city.y, city.x] = min(
            1.0, float(len(city.production_queue)) / float(queue_limit)
        )
        if city.production_kind and city.production_target:
            item = (city.production_kind, city.production_target)
            item_idx = PRODUCTION_ITEM_INDEX.get(item)
            if item_idx is not None:
                current_planes[item_idx][city.y, city.x] = 1.0
            cost = self._production_cost(city.production_kind, city.production_target)
            if cost > 0:
                progress_plane[city.y, city.x] = min(
                    1.0, float(city.production_progress) / float(cost)
                )
        if city.production_queue:
            next_idx = PRODUCTION_ITEM_INDEX.get(city.production_queue[0])
            if next_idx is not None:
                next_planes[next_idx][city.y, city.x] = 1.0

    @staticmethod
    def _production_cost(kind: str, name: str) -> int:
        if kind == "unit":
            spec = UNIT_SPECS.get(name)
        elif kind == "building":
            spec = BUILDING_SPECS.get(name)
        else:
            spec = None
        return int(getattr(spec, "cost", 0) or 0)

    def string(self) -> str:
        parts = [f"turn={self.turn}"]
        parts.append(f"acts={self.actions_this_turn}")
        for p in (1, -1):
            for idx, u in enumerate(self.units[p]):
                if u.alive:
                    parts.append(f"p{p}u{idx}:{u.x},{u.y},hp{u.hp}")
            for cidx, city in enumerate(self.cities[p]):
                parts.append(f"c{p}{cidx}:{city.x},{city.y},sz{city.size}")
            # Include research status to disambiguate states with identical unit positions.
            res_bits = ''.join('1' if self.research_done[p].get(tech, False) else '0' for tech in self.RESEARCH_TECHS)
            parts.append(f"r{p}:{res_bits}")
        if self.winner is not None:
            parts.append(f"winner={self.winner}")
        parts.append(f"kills:{self.kills.get(1,0)}/{self.kills.get(-1,0)}")
        return '|'.join(parts)
