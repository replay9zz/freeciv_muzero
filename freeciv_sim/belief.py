from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .movement import FreecivMovement

Coord = Tuple[int, int]


@dataclass
class OpponentBelief:
    visible_units: np.ndarray
    visible_cities: np.ndarray
    belief_units: np.ndarray
    threat: np.ndarray
    last_seen_age: np.ndarray
    territory: np.ndarray


@dataclass
class BeliefTracker:
    map_w: int
    map_h: int
    max_enemy_slots: int = 3
    slots: Dict[int, OpponentBelief] = field(default_factory=dict)
    movement: FreecivMovement | None = None

    def __post_init__(self) -> None:
        if self.movement is None:
            self.movement = FreecivMovement(self.map_w, self.map_h)

    def _zeros(self) -> np.ndarray:
        return np.zeros((self.map_h, self.map_w), dtype=np.float32)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.map_w and 0 <= y < self.map_h

    def ensure_slot(self, slot_id: int) -> OpponentBelief:
        if slot_id not in self.slots:
            self.slots[slot_id] = OpponentBelief(
                visible_units=self._zeros(),
                visible_cities=self._zeros(),
                belief_units=self._zeros(),
                threat=self._zeros(),
                last_seen_age=self._zeros(),
                territory=self._zeros(),
            )
        return self.slots[slot_id]

    def begin_turn(self) -> None:
        for mem in self.slots.values():
            self._clear_visible(mem)
            mem.last_seen_age += 1.0

    def begin_observation(self) -> None:
        for mem in self.slots.values():
            self._clear_visible(mem)

    def _clear_visible(self, mem: OpponentBelief) -> None:
        mem.visible_units.fill(0.0)
        mem.visible_cities.fill(0.0)

    def update_territory(
        self,
        slot_id: int,
        enemy_tiles: Iterable[Coord],
        neutral_tiles: Iterable[Coord],
        my_tiles: Iterable[Coord],
    ) -> None:
        mem = self.ensure_slot(slot_id)
        mem.territory.fill(0.0)
        for x, y in enemy_tiles:
            if self._in_bounds(x, y):
                mem.territory[y, x] = 1.0
        for x, y in neutral_tiles:
            if self._in_bounds(x, y):
                mem.territory[y, x] = 0.4
        for x, y in my_tiles:
            if self._in_bounds(x, y):
                mem.territory[y, x] = 0.1

    def observe_units(self, slot_id: int, coords: Iterable[Coord]) -> None:
        mem = self.ensure_slot(slot_id)
        for x, y in coords:
            if not self._in_bounds(x, y):
                continue
            mem.visible_units[y, x] = 1.0
            mem.belief_units[y, x] = 1.0
            mem.last_seen_age[y, x] = 0.0

    def observe_cities(self, slot_id: int, coords: Iterable[Coord]) -> None:
        mem = self.ensure_slot(slot_id)
        for x, y in coords:
            if not self._in_bounds(x, y):
                continue
            mem.visible_cities[y, x] = 1.0
            mem.last_seen_age[y, x] = 0.0

    def mask_visible_tiles(self, slot_id: int, coords: Iterable[Coord]) -> None:
        mem = self.ensure_slot(slot_id)
        for x, y in coords:
            if not self._in_bounds(x, y):
                continue
            if mem.visible_units[y, x] > 0.0 or mem.visible_cities[y, x] > 0.0:
                continue
            mem.belief_units[y, x] = 0.0

    def _diffuse_once(self, belief: np.ndarray) -> np.ndarray:
        assert self.movement is not None
        nxt = np.zeros_like(belief)

        for y in range(self.map_h):
            for x in range(self.map_w):
                mass = float(belief[y, x])
                if mass <= 0.0:
                    continue

                neighbors = [
                    (nx, ny)
                    for nx, ny in self.movement.get_native_neighbors(x, y)
                    if nx is not None and ny is not None
                ]
                share = mass / float(len(neighbors) + 1)
                nxt[y, x] += share
                for nx, ny in neighbors:
                    nxt[ny, nx] += share

        return np.clip(nxt, 0.0, 1.0)

    def diffuse_belief(self, slot_id: int, steps: int = 1) -> None:
        mem = self.ensure_slot(slot_id)
        cur = mem.belief_units.copy()
        for _ in range(max(1, steps)):
            cur = self._diffuse_once(cur)
        mem.belief_units = np.clip(cur * np.maximum(mem.territory, 0.1), 0.0, 1.0)

    def rebuild_threat(self, slot_id: int, my_border: np.ndarray) -> None:
        mem = self.ensure_slot(slot_id)
        mem.threat = np.clip(
            0.7 * mem.belief_units + 0.3 * my_border + 0.2 * mem.visible_cities,
            0.0,
            1.0,
        )

    def export_planes(self, slot_id: int) -> List[np.ndarray]:
        mem = self.ensure_slot(slot_id)
        age = np.clip(mem.last_seen_age / 20.0, 0.0, 1.0)
        return [
            mem.visible_units,
            mem.visible_cities,
            mem.belief_units,
            mem.threat,
            age,
            mem.territory,
        ]
