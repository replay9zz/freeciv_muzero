from __future__ import annotations

import argparse
import json
import pathlib
import sys
from bisect import bisect_right
from typing import Any

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from games.freeciv_remote import MuZeroConfig
from freeciv_sim.state.multihead_state import (
    MultiheadState,
    PRODUCTION_ITEM_INDEX,
    PRODUCTION_ITEM_NAMES,
)
from freeciv_sim.imitation.train_imitation import (
    _city_slot,
    _direction,
    _make_state,
    _unit_slot,
)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_index"] = idx
            rows.append(row)
    return rows


def _snapshots_by_turn(rows: list[dict[str, Any]]) -> tuple[list[int], dict[int, dict[str, Any]]]:
    by_turn: dict[int, dict[str, Any]] = {}
    for row in rows:
        snap = row.get("snapshot") or {}
        turn = int(row.get("turn") or snap.get("turn") or 0)
        # First snapshot in a turn is closest to phase start.
        by_turn.setdefault(turn, row)
    return sorted(by_turn), by_turn


def _snapshot_for_action(
    turns: list[int],
    by_turn: dict[int, dict[str, Any]],
    action_turn: int,
) -> dict[str, Any] | None:
    # ACTLOG actions are emitted during the current server turn.  The first
    # snapshot for that same turn is normally closest to the pre-action state.
    if int(action_turn) in by_turn:
        return by_turn[int(action_turn)]
    idx = bisect_right(turns, int(action_turn)) - 1
    if idx < 0:
        return None
    return by_turn[turns[idx]]


def _player_from_action(action: dict[str, Any]) -> int | None:
    for key in ("actor_player", "target_tile_owner", "target_player", "phase_player"):
        value = action.get(key)
        if value is not None:
            return int(value)
    return None


def _find_unit_on_tile(snapshot: dict[str, Any], player_id: int, x: int, y: int) -> int | None:
    units = sorted(
        [
            unit
            for unit in snapshot.get("units", [])
            if int(unit.get("owner", -9999)) == int(player_id)
            and int(unit.get("x", -1)) == int(x)
            and int(unit.get("y", -1)) == int(y)
        ],
        key=lambda unit: int(unit.get("id", 0)),
    )
    if not units:
        return None
    return int(units[0].get("id"))


def _action_index_from_actionlog(
    config: MuZeroConfig,
    snapshot_row: dict[str, Any],
    action: dict[str, Any],
) -> tuple[int, int] | None:
    snapshot = snapshot_row.get("snapshot") or {}
    name = str(action.get("action") or "")
    event = str(action.get("event") or "")
    player_id = _player_from_action(action)
    if player_id is None:
        return None

    move_size = config.max_units * MultiheadState.MOVE_PER_UNIT
    attack_size = config.max_units * MultiheadState.ATTACK_PER_UNIT
    activity_size = config.max_units * MultiheadState.UNIT_ACTIVITY_PER_UNIT
    econ_offset = move_size + attack_size + activity_size
    production_offset = (
        econ_offset + len(MultiheadState.RESEARCH_TECHS) + config.max_units
    )

    if name == "Unit Move":
        unit_id = action.get("actor_unit")
        if unit_id is None:
            return None
        slot = _unit_slot(snapshot, player_id, int(unit_id), config.max_units)
        if slot is None:
            return None
        units = {
            int(unit.get("id")): unit
            for unit in snapshot.get("units", [])
            if unit.get("id") is not None
        }
        unit = units.get(int(unit_id))
        if unit is None:
            return None
        state = _make_state(config, snapshot, player_id)
        direction = _direction(
            state,
            [int(unit.get("x", 0)), int(unit.get("y", 0))],
            [int(action.get("target_x", unit.get("x", 0))), int(action.get("target_y", unit.get("y", 0)))],
        )
        if direction is None:
            return None
        return player_id, slot * MultiheadState.MOVE_PER_UNIT + direction

    if name == "Found City":
        x = int(action.get("target_x", 0))
        y = int(action.get("target_y", 0))
        unit_id = action.get("actor_unit")
        if unit_id is None:
            unit_id = _find_unit_on_tile(snapshot, player_id, x, y)
        if unit_id is None:
            return None
        slot = _unit_slot(snapshot, player_id, int(unit_id), config.max_units)
        if slot is None:
            return None
        return player_id, econ_offset + len(MultiheadState.RESEARCH_TECHS) + slot

    if event in {
        "action_finished_unit_unit",
        "action_finished_unit_units",
        "action_finished_unit_city",
    }:
        unit_id = action.get("actor_unit")
        if unit_id is None:
            return None
        slot = _unit_slot(snapshot, player_id, int(unit_id), config.max_units)
        if slot is None:
            return None
        units = {
            int(unit.get("id")): unit
            for unit in snapshot.get("units", [])
            if unit.get("id") is not None
        }
        unit = units.get(int(unit_id))
        if unit is None:
            return None
        target_x = action.get("target_x")
        target_y = action.get("target_y")
        if target_x is None or target_y is None:
            return None
        state = _make_state(config, snapshot, player_id)
        direction = _direction(
            state,
            [int(unit.get("x", 0)), int(unit.get("y", 0))],
            [int(target_x), int(target_y)],
        )
        if direction is None:
            return None
        return player_id, move_size + slot * MultiheadState.ATTACK_PER_UNIT + direction

    if event == "unit_built":
        city_id = action.get("source_city")
        unit_name = action.get("actor_type")
        if city_id is None or unit_name is None:
            return None
        city_slot = _city_slot(snapshot, player_id, int(city_id), config.max_cities)
        item_idx = PRODUCTION_ITEM_INDEX.get(("unit", str(unit_name)))
        if city_slot is None or item_idx is None:
            return None
        return (
            player_id,
            production_offset
            + city_slot * len(PRODUCTION_ITEM_NAMES)
            + item_idx,
        )

    if event == "building_built":
        city_id = action.get("target_city")
        building_name = action.get("building")
        if city_id is None or building_name is None:
            return None
        city_slot = _city_slot(snapshot, player_id, int(city_id), config.max_cities)
        item_idx = PRODUCTION_ITEM_INDEX.get(("building", str(building_name)))
        if city_slot is None or item_idx is None:
            return None
        return (
            player_id,
            production_offset
            + city_slot * len(PRODUCTION_ITEM_NAMES)
            + item_idx,
        )

    return None


def build_samples(
    snapshots_path: pathlib.Path,
    actions_path: pathlib.Path,
    out_path: pathlib.Path,
    validate_legal: bool,
    actionlog_path: pathlib.Path | None = None,
) -> dict[str, int]:
    config = MuZeroConfig()
    snapshots = _read_jsonl(snapshots_path)
    actions = _read_jsonl(actions_path)
    if actionlog_path is not None:
        actions.extend(
            row
            for row in _read_jsonl(actionlog_path)
            if row.get("event") in {"unit_built", "building_built"}
        )
        actions.sort(key=lambda row: int(row.get("_source_line", 0)))
    turns, by_turn = _snapshots_by_turn(snapshots)
    stats = {
        "snapshots": len(snapshots),
        "actions": len(actions),
        "labeled": 0,
        "missing_snapshot": 0,
        "unmapped": 0,
        "illegal": 0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for action in actions:
            snapshot_row = _snapshot_for_action(turns, by_turn, int(action.get("turn") or 0))
            if snapshot_row is None:
                stats["missing_snapshot"] += 1
                continue
            labeled = _action_index_from_actionlog(config, snapshot_row, action)
            if labeled is None:
                stats["unmapped"] += 1
                continue
            player_id, action_index = labeled
            state = _make_state(config, snapshot_row.get("snapshot") or {}, player_id)
            if validate_legal and not state.valid_moves(1)[action_index]:
                stats["illegal"] += 1
                continue
            record = {
                "snapshot_ref": int(snapshot_row["_line_index"]),
                "turn": int(action.get("turn") or 0),
                "player": int(player_id),
                "event": action.get("event"),
                "action": action.get("action"),
                "action_index": int(action_index),
                "source_line": action.get("_source_line"),
            }
            fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stats["labeled"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Freeciv imitation samples from snapshots and ACTLOG JSONL.")
    parser.add_argument("--snapshots", required=True, type=pathlib.Path)
    parser.add_argument("--actionlog", type=pathlib.Path, help="All ACTLOG records; kept for pipeline metadata.")
    parser.add_argument("--actions", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--validate-legal", action="store_true")
    args = parser.parse_args()

    stats = build_samples(
        args.snapshots,
        args.actions,
        args.out,
        args.validate_legal,
        args.actionlog,
    )
    print(f"[imitation-dataset] out={args.out} stats={stats}", flush=True)


if __name__ == "__main__":
    main()
