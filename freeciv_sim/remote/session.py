from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .lua_client import LuaRemoteClient
from .lua_actions import auto_settler, set_player_research
from ..state.movement import FreecivMovement
from .lua_queries import (
    list_all_cities,
    list_all_units,
    list_city_walls,
    list_player_known_techs,
    list_visible_tiles_call,
    parse_position_result,
    parse_vision_tiles,
    player_knows_tech,
    simple_find_unit_pos,
)
from ..state.config import MapConfig
from ..state.multihead_state import MultiheadState
from ..rules.research import TARGET_TECH_NAME, TECH_PREREQS, pick_next_goal_tech, pick_next_priority_tech


def get_unit_rule_name(client: LuaRemoteClient, unit_id: int) -> Optional[str]:
    lua = (
        "return (function() "
        f"local target_id={int(unit_id)}; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local iter = pl.units_iterate and pl:units_iterate() or nil; "
        "    if iter then "
        "      while true do "
        "        local u = iter(); "
        "        if not u then break end; "
        "        if u.id == target_id then "
        "          local ok, ut = pcall(function() return u:utype() end); "
        "          if not ok or not ut then ut = u.utype end; "
        "          if not ut then return '__NONE__' end; "
        "          local ok2, nm = pcall(function() return ut:rule_name() end); "
        "          if ok2 and nm then return nm end; "
        "          if ut.rule_name then return ut.rule_name end; "
        "          local ok3, tn = pcall(function() return ut:name_translation() end); "
        "          if ok3 and tn then return tn end; "
        "          return '__NONE__' "
        "        end "
        "      end "
        "    end "
        "  end "
        "end "
        "return '__NONE__' "
        "end)()"
    )
    try:
        res = client.eval(lua)
        val = res.last_return() if res else None
        if isinstance(val, str) and val != "__NONE__":
            return val
    except Exception:
        return None
    return None


@dataclass
class Snapshot:
    au_map: np.ndarray
    enemy_map: np.ndarray
    visited: np.ndarray
    revealed: np.ndarray
    player_pos: Tuple[int, int]
    enemy_pos: Tuple[int, int]
    status_lookup: Dict[Tuple[int, int], Tuple[str, bool, bool, bool, bool]]
    terrain_map: Optional[np.ndarray] = None
    research_name: Optional[str] = None
    research_done: bool = False
    research_flags: Dict[str, bool] = field(default_factory=dict)


def chunked(seq: Iterable[Tuple[int, int]], size: int) -> Iterable[List[Tuple[int, int]]]:
    bucket: List[Tuple[int, int]] = []
    for item in seq:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def simple_knows_tech(client: LuaRemoteClient, player_id: int, tech_name: str) -> bool:
    return player_knows_tech(client, player_id, tech_name)


def discover_client_player_id(client: LuaRemoteClient) -> Optional[int]:
    lua = (
        "return (function() "
        "if client and client.player_id then "
        "  local ok,pid=pcall(client.player_id); "
        "  if ok and pid and pid >= 0 then return string.format('__PLAYER__ %d', pid) end "
        "end; "
        "for i=0,63 do "
        "  local pl=find.player and find.player(i); "
        "  if pl then "
        "    local ok,gui=pcall(function() return pl:controlling_gui() end); "
        "    gui=ok and tostring(gui or '') or ''; "
        "    if gui ~= '' and gui ~= 'None' then "
        "      return string.format('__PLAYER__ %d', pl.id or i) "
        "    end "
        "  end "
        "end; "
        "return '__PLAYER__ -1' "
        "end)()"
    )
    try:
        res = client.eval(lua)
        val = res.last_return() if res else None
    except Exception:
        return None
    if not isinstance(val, str) or "__PLAYER__" not in val:
        return None
    try:
        pid = int(float(val.split("__PLAYER__", 1)[1].strip()))
    except ValueError:
        return None
    return pid if pid >= 0 else None


def discover_controlled_units(
    client: LuaRemoteClient,
    player_hint: Optional[int],
) -> Tuple[List[int], Optional[int]]:
    try:
        units = list_all_units(client)
    except Exception as exc:
        raise RuntimeError("Failed to enumerate units via LuaRemote.") from exc

    client_player_id = discover_client_player_id(client)
    player_id = client_player_id if client_player_id is not None else player_hint
    if player_id is None:
        for _uid, _x, _y, unit_owner in units:
            if unit_owner >= 0:
                player_id = unit_owner
                break

    if player_id is None:
        return [], None

    controlled = [uid for uid, _x, _y, unit_owner in units if unit_owner == player_id]
    if (
        not controlled
        and player_hint is not None
        and client_player_id is not None
        and client_player_id != player_hint
    ):
        player_id = client_player_id
        controlled = [uid for uid, _x, _y, unit_owner in units if unit_owner == player_id]
    return controlled, player_id


def discover_player_cities(
    client: LuaRemoteClient,
    player_id: Optional[int],
) -> List[Tuple[int, int, int]]:
    if player_id is None:
        return []
    try:
        cities = list_all_cities(client)
    except Exception:
        return []
    owned: List[Tuple[int, int, int]] = []
    for cid, cx, cy, owner, _name in cities:
        if owner == player_id:
            owned.append((cid, cx, cy))
    return owned


def query_player_research(client: LuaRemoteClient, player_id: int) -> str:
    lua = (
        "return (function() "
        f"local pl = find.player and find.player({player_id}); "
        "if not pl then return '__NORESEARCH__' end; "
        "local ok, tech = pcall(function() return pl:researching() end); "
        "if not ok or tech == nil then return '__NORESEARCH__' end; "
        "if type(tech) == 'string' then return '__TECH__ '..tech end; "
        "local ok2, name = pcall(function() return tech:rule_name() end); "
        "if ok2 and name and name ~= '' then return '__TECH__ '..name end; "
        "return '__NORESEARCH__' "
        "end)()"
    )
    try:
        res = client.eval(lua)
        val = res.last_return() if res else None
        return val if isinstance(val, str) else "__NORESEARCH__"
    except Exception:
        return "__NORESEARCH__"


def set_research_to_target(
    client: LuaRemoteClient,
    player_id: Optional[int],
    research_flags: Optional[Dict[str, bool]] = None,
    tech_name: Optional[str] = None,
) -> bool:
    if player_id is None:
        return False
    if tech_name is None:
        flags = research_flags or {}
        tech_name = pick_next_priority_tech(
            flags, priority_chain=None, prereqs=TECH_PREREQS
        )
        if tech_name is None:
            tech_name = pick_next_goal_tech(TARGET_TECH_NAME, flags, prereqs=TECH_PREREQS)
        if tech_name is None:
            tech_name = TARGET_TECH_NAME
    try:
        ok = set_player_research(client, player_id, tech_name)
        actual = query_player_research(client, player_id)
        print(f"[research] request={tech_name} current={actual} ok={ok}")
        return ok
    except Exception:
        return False


def is_target_researched(
    client: LuaRemoteClient,
    player_id: Optional[int],
    tech_name: str = TARGET_TECH_NAME,
) -> bool:
    if player_id is None:
        return False
    try:
        return player_knows_tech(client, player_id, tech_name)
    except Exception:
        return False


def gather_snapshot(
    client: LuaRemoteClient,
    movement: FreecivMovement,
    cfg: MapConfig,
    unit_id: int,
    player_id: Optional[int],
    known_tiles: Dict[Tuple[int, int], str],
    known_terrains: Dict[Tuple[int, int], str],
    known_enemy: Dict[Tuple[int, int], bool],
    visited_tiles: Set[Tuple[int, int]],
) -> Tuple[Snapshot, Optional[int]]:
    known_enemy.clear()
    pos_result = client.eval(simple_find_unit_pos(unit_id))
    pos_info = parse_position_result(pos_result)
    if pos_info is None:
        raise RuntimeError("Controlled unit was not found in the current Freeciv session.")

    player_pos = (pos_info[0], pos_info[1])
    if player_id is None and pos_info[2] is not None and pos_info[2] >= 0:
        player_id = int(pos_info[2])
    research_name: Optional[str] = None
    research_done = False
    research_flags: Dict[str, bool] = {tech: False for tech in MultiheadState.RESEARCH_TECHS}
    if player_id is not None:
        try:
            research_name = query_player_research(client, player_id)
        except Exception:
            research_name = None
        try:
            research_done = is_target_researched(client, player_id)
        except Exception:
            research_done = False
        try:
            research_flags.update(
                list_player_known_techs(client, player_id, MultiheadState.RESEARCH_TECHS)
            )
        except Exception:
            for tech in MultiheadState.RESEARCH_TECHS:
                try:
                    research_flags[tech] = simple_knows_tech(client, player_id, tech)
                except Exception:
                    continue
        research_done = research_flags.get(TARGET_TECH_NAME, research_done)

    visible_tiles: Set[Tuple[int, int]] = set()
    status_lookup: Dict[Tuple[int, int], Tuple[str, bool, bool, bool, bool]] = {}

    if player_id is not None:
        try:
            tiles_result = client.eval(list_visible_tiles_call(player_id, unit_id))
            visible_tiles = set(parse_vision_tiles(tiles_result))
        except Exception:
            visible_tiles = set()

    visible_tiles.add(player_pos)

    coords_to_query: Set[Tuple[int, int]] = set(visible_tiles)
    coords_to_query.update(
        coord
        for coord in movement.get_native_neighbors(*player_pos)
        if coord[0] is not None and coord[1] is not None
    )

    for batch in chunked(coords_to_query, 48):
        try:
            batch_status = client.neighbors_status(unit_id, batch)
        except Exception:
            continue
        for entry in batch_status:
            if len(entry) < 7:
                continue
            nx, ny, au_char, enemy_flag, terrain, enemy_units, friendly_units = entry[:7]
            has_walls = False
            if len(entry) > 7:
                has_walls = bool(entry[7])
            status_lookup[(nx, ny)] = (
                au_char,
                bool(enemy_flag),
                bool(enemy_units),
                bool(friendly_units),
                has_walls,
            )
            if terrain:
                known_terrains[(nx, ny)] = str(terrain)

    for coord, status in status_lookup.items():
        if not status:
            continue
        au_char, enemy_flag, enemy_units, friendly_units = status[:4]
        if au_char:
            known_tiles[coord] = au_char
        if enemy_flag or enemy_units:
            known_enemy[coord] = True

    known_tiles[player_pos] = "A"

    au_grid = np.full((cfg.map_h, cfg.map_w), "U", dtype="<U1")
    enemy_grid = np.zeros((cfg.map_h, cfg.map_w), dtype=bool)
    visited_grid = np.zeros((cfg.map_h, cfg.map_w), dtype=bool)
    revealed_grid = np.zeros((cfg.map_h, cfg.map_w), dtype=bool)
    terrain_grid = np.full((cfg.map_h, cfg.map_w), "", dtype="<U64")

    for (nx, ny), au_char in known_tiles.items():
        if 0 <= ny < cfg.map_h and 0 <= nx < cfg.map_w:
            au_grid[ny, nx] = au_char
            revealed_grid[ny, nx] = True
    for (nx, ny), enemy_flag in known_enemy.items():
        if enemy_flag and 0 <= ny < cfg.map_h and 0 <= nx < cfg.map_w:
            enemy_grid[ny, nx] = True
    for (nx, ny), terrain in known_terrains.items():
        if 0 <= ny < cfg.map_h and 0 <= nx < cfg.map_w:
            terrain_grid[ny, nx] = terrain
    for (nx, ny) in visited_tiles:
        if 0 <= ny < cfg.map_h and 0 <= nx < cfg.map_w:
            visited_grid[ny, nx] = True

    px, py = player_pos
    if 0 <= py < cfg.map_h and 0 <= px < cfg.map_w:
        au_grid[py, px] = "A"
        revealed_grid[py, px] = True
        visited_grid[py, px] = True

    enemy_pos = player_pos
    for coord, flag in known_enemy.items():
        if flag:
            enemy_pos = coord
            break

    snapshot = Snapshot(
        au_map=au_grid,
        enemy_map=enemy_grid,
        visited=visited_grid,
        revealed=revealed_grid,
        player_pos=player_pos,
        enemy_pos=enemy_pos,
        status_lookup=status_lookup,
        terrain_map=terrain_grid,
        research_name=research_name,
        research_done=research_done,
        research_flags=research_flags,
    )
    return snapshot, player_id


def parse_dir_ids(raw: str) -> List[int]:
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if len(parts) != 6:
        raise ValueError("Expected 6 comma-separated direction ids for [N,NE,SE,S,SW,NW]")
    return [int(p) for p in parts]
