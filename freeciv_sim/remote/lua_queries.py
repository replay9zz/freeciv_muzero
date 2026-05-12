import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

from .lua_client import EvalResult, LuaRemoteClient
from ..state.movement import FreecivMovement


def simple_find_unit_pos(uid: int) -> str:
    return (
        "return (function() "
        f"local target_id={int(uid)}; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local iter = pl.units_iterate and pl:units_iterate() or nil; "
        "    if iter then "
        "      while true do "
        "        local u = iter(); "
        "        if not u then break end; "
        "        if u.id == target_id then "
        "          local t = u.tile; "
        "          if t then "
        "            return string.format('__POS__ %d %d %d', t.nat_x or -1, t.nat_y or -1, pl.id or (pl.player_num or -1)) "
        "          else "
        "            return '__NOPOS__' "
        "          end "
        "        end "
        "      end "
        "    end "
        "  end "
        "end "
        "return '__NOPOS__' "
        "end)()"
    )


def list_visible_tiles_call(player_id: int, unit_id: int) -> str:
    return (
        "return (function() "
        "if type(list_visible_tiles) ~= 'function' then return '__NO__' end; "
        f"return list_visible_tiles({int(player_id)}, {int(unit_id)}) "
        "end)()"
    )


def parse_vision_tiles(result: EvalResult) -> List[Tuple[int, int]]:
    s = result.last_return()
    tiles: List[Tuple[int, int]] = []
    if not s:
        return tiles
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",", 1)
        if len(parts) != 2:
            continue
        try:
            sx, sy = parts
            tiles.append((int(sx), int(sy)))
        except ValueError:
            continue
    return tiles


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "y", "yes"}
    return bool(value)


def _lua_quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def parse_position_result(result: EvalResult) -> Optional[Tuple[int, int, Optional[int]]]:
    candidates: List[str] = []
    try:
        ret = result.last_return()
        if ret:
            candidates.append(ret)
    except Exception:
        pass
    try:
        for r in getattr(result, "returns", []):
            if r not in candidates:
                candidates.append(r)
    except Exception:
        pass
    try:
        candidates.extend(getattr(result, "lines", []))
    except Exception:
        pass

    for s in candidates:
        if not isinstance(s, str):
            continue
        if "__POS__" in s or "**POS**" in s:
            match = re.search(
                r"(?:\*\*POS\*\*|__POS__)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(?:\s+(-?\d+(?:\.\d+)?))?",
                s,
            )
            if match:
                sx, sy, sowner = match.groups()
                try:
                    x_val = int(float(sx))
                    y_val = int(float(sy))
                except ValueError:
                    continue
                owner_val = None
                if sowner is not None:
                    try:
                        owner_val = int(float(sowner))
                    except ValueError:
                        owner_val = None
                return (x_val, y_val, owner_val)
        if s.strip() in {"__NOPOS__", "**NOPOS**"}:
            return None
    return None


def _escape_lua_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _build_tech_lookup_lua(tech_identifier: Union[str, int]) -> str:
    if isinstance(tech_identifier, str):
        safe_name = _escape_lua_string(tech_identifier)
        return (
            "local tech=nil; "
            f"if find.tech_type then tech = find.tech_type('{safe_name}') or tech end; "
            f"if find.tech then tech = tech or find.tech('{safe_name}') end; "
            "if not tech and find.tech_type_iterate then "
            "  local iter = find.tech_type_iterate(); "
            "  while true do "
            "    local t = iter(); "
            "    if not t then break end; "
            f"    if t.name and string.lower(t.name) == string.lower('{safe_name}') then tech = t; break end; "
            f"    if t.rule_name and string.lower(t:rule_name()) == string.lower('{safe_name}') then tech = t; break end; "
            "  end "
            "end "
        )
    tech_id = int(tech_identifier)
    return (
        "local tech=nil; "
        f"if find.tech_type then tech = find.tech_type({tech_id}) or tech end; "
        f"if find.tech then tech = tech or find.tech({tech_id}) end; "
        "if not tech and find.tech_type_iterate then "
        "  local iter = find.tech_type_iterate(); "
        "  while true do "
        "    local t = iter(); "
        "    if not t then break end; "
        "    if t.id and t.id == " + str(tech_id) + " then tech = t; break end; "
        "  end "
        "end "
    )


def player_knows_tech(lr: LuaRemoteClient, player_id: int, tech_identifier: Union[str, int]) -> bool:
    lookup = _build_tech_lookup_lua(tech_identifier)
    lua = (
        "return (function() "
        f"local pl = find.player and find.player({int(player_id)}); "
        f"{lookup}"
        "if not pl or not tech then return '__KNOWS__ 0' end; "
        "local known = false; "
        "if pl.knows_tech then known = pl:knows_tech(tech) end; "
        "return known and '__KNOWS__ 1' or '__KNOWS__ 0' "
        "end)()"
    )
    try:
        res = lr.eval(lua)
    except Exception:
        return False

    candidates: List[str] = []
    try:
        ret = res.last_return()
        if ret:
            candidates.append(ret)
    except Exception:
        pass
    try:
        candidates.extend(getattr(res, "lines", []))
    except Exception:
        pass
    try:
        candidates.extend(getattr(res, "returns", []))
    except Exception:
        pass

    return any(
        isinstance(val, str) and ("__KNOWS__ 1" in val or "**KNOWS** 1" in val)
        for val in candidates
    )


def set_player_research(lr: LuaRemoteClient, player_id: int, tech_identifier: Union[str, int]) -> bool:
    lookup = _build_tech_lookup_lua(tech_identifier)
    lua = (
        "return (function() "
        f"local target_pid = {int(player_id)}; "
        "local pl = find.player and find.player(target_pid); "
        f"{lookup}"
        "if not pl or not tech then return '__SETRESEARCH__ 0' end; "
        "local tech_name=nil; "
        "if tech.rule_name then "
        "  local ok,res=pcall(function() return tech:rule_name() end); "
        "  if ok and res and res ~= '' then tech_name=res end "
        "end "
        "if not tech_name and tech.name then tech_name=tostring(tech.name) end; "
        "if not tech_name or tech_name == '' then return '__SETRESEARCH__ 0' end; "
        "local ok,res=pcall(function() "
        "  if client and client.set_research then return client.set_research(target_pid, tech_name) end "
        "  return nil "
        "end); "
        "local success=false; "
        "if ok and res~=nil then success = (res==true) or (res==1) end; "
        "if success then return '__SETRESEARCH__ 1' end; "
        "local ok2,res2=pcall(function() "
        "  if client and client.set_research_goal then return client.set_research_goal(target_pid, tech_name) end "
        "  return nil "
        "end); "
        "local success2=false; "
        "if ok2 and res2~=nil then success2 = (res2==true) or (res2==1) end; "
        "if success2 then return '__SETRESEARCH__ 1' end; "
        "pl.researching = tech; "
        "return '__SETRESEARCH__ 1' "
        "end)()"
    )
    try:
        res = lr.eval(lua)
    except Exception:
        return False

    candidates: List[str] = []
    try:
        ret = res.last_return()
        if ret:
            candidates.append(ret)
    except Exception:
        pass
    try:
        candidates.extend(getattr(res, "lines", []))
    except Exception:
        pass
    try:
        candidates.extend(getattr(res, "returns", []))
    except Exception:
        pass

    return any(
        isinstance(val, str) and ("__SETRESEARCH__ 1" in val or "**SETRESEARCH** 1" in val)
        for val in candidates
    )


def set_government(lr: LuaRemoteClient, gov_identifier: Union[str, int]) -> bool:
    gov_value = str(gov_identifier)
    lua = (
        "return (function() "
        f"local gov='{_escape_lua_string(gov_value)}'; "
        "local ok,res=pcall(function() "
        "  if client and client.set_government then return client.set_government(gov) end "
        "  return nil "
        "end); "
        "local success=false; "
        "if ok and res~=nil then success = (res==true) or (res==1) end; "
        "if success then return '__SETGOV__ 1' end; "
        "return '__SETGOV__ 0' "
        "end)()"
    )
    try:
        res = lr.eval(lua)
    except Exception:
        return False

    candidates: List[str] = []
    try:
        ret = res.last_return()
        if ret:
            candidates.append(ret)
    except Exception:
        pass
    try:
        candidates.extend(getattr(res, "lines", []))
    except Exception:
        pass
    try:
        candidates.extend(getattr(res, "returns", []))
    except Exception:
        pass

    return any(
        isinstance(val, str) and ("__SETGOV__ 1" in val or "**SETGOV** 1" in val)
        for val in candidates
    )


def auto_settler(lr: LuaRemoteClient, unit_id: int) -> bool:
    lua = (
        "return (function() "
        f"local uid={int(unit_id)}; "
        "local ok,res=pcall(function() "
        "  if client and client.auto_settler then return client.auto_settler(uid) end "
        "  return nil "
        "end); "
        "local success=false; "
        "if ok and res~=nil then success = (res==true) or (res==1) end; "
        "if success then return '__AUTOSETTLER__ 1' end; "
        "return '__AUTOSETTLER__ 0' "
        "end)()"
    )
    try:
        res = lr.eval(lua)
    except Exception:
        return False

    candidates: List[str] = []
    try:
        ret = res.last_return()
        if ret:
            candidates.append(ret)
    except Exception:
        pass
    try:
        candidates.extend(getattr(res, "lines", []))
    except Exception:
        pass
    try:
        candidates.extend(getattr(res, "returns", []))
    except Exception:
        pass

    return any(
        isinstance(val, str) and ("__AUTOSETTLER__ 1" in val or "**AUTOSETTLER** 1" in val)
        for val in candidates
    )


def list_all_units(lr: LuaRemoteClient) -> List[Tuple[int, int, int, int]]:
    lua = (
        "local parts = {}; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local ir = pl.units_iterate and pl:units_iterate() or nil; "
        "    if ir then "
        "      while true do "
        "        local u = ir(); "
        "        if not u then break end; "
        "        local t = u.tile; "
        "        if t then "
        "          local owner_id = -1; "
        "          if pl.id then owner_id = pl.id elseif pl.player_num then owner_id = pl.player_num end; "
        "          parts[#parts + 1] = string.format('%d|%d|%d|%d', u.id, t.nat_x or -1, t.nat_y or -1, owner_id); "
        "        end "
        "      end "
        "    end "
        "  end "
        "end; "
        "return table.concat(parts, ';')"
    )
    result = lr.eval(lua)
    payload = result.last_return()
    units: List[Tuple[int, int, int, int]] = []
    if payload:
        for chunk in payload.split(";"):
            if not chunk:
                continue
            parts = chunk.split("|")
            if len(parts) < 4:
                continue
            try:
                uid = int(parts[0])
                nx = int(parts[1])
                ny = int(parts[2])
                owner = int(parts[3])
                units.append((uid, nx, ny, owner))
            except ValueError:
                continue
    return units


def list_all_unit_status(lr: LuaRemoteClient) -> List[Tuple[int, int, int, int, str, int, int]]:
    lua = (
        "local parts = {}; "
        "local you = nil; "
        "if client and client.conn then "
        "  if client.conn.playing then you = client.conn.playing "
        "  elseif client.conn.player then you = client.conn.player end "
        "end; "
        "local function can_see(u) "
        "  if not you then return true end; "
        "  if u.tile then "
        "    local t = u.tile; "
        "    if t.known_and_seen then "
        "      local ok, res = pcall(function() return t:known_and_seen(you) end); "
        "      if ok then return res and true or false end; "
        "    end "
        "  end "
        "  if u.can_be_seen_by then "
        "    local ok, res = pcall(function() return u:can_be_seen_by(you) end); "
        "    if ok then return res and true or false end; "
        "  end "
        "  return true "
        "end; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local ir = pl.units_iterate and pl:units_iterate() or nil; "
        "    if ir then "
        "      while true do "
        "        local u = ir(); "
        "        if not u then break end; "
        "        local t = u.tile; "
        "        if t and can_see(u) then "
        "          local owner_id = -1; "
        "          if pl.id then owner_id = pl.id elseif pl.player_num then owner_id = pl.player_num end; "
        "          local ut = nil; "
        "          local ok_ut, res_ut = pcall(function() return u:utype() end); "
        "          if ok_ut and res_ut then ut = res_ut else ut = u.utype end; "
        "          local uname = ''; "
        "          if ut then "
        "            local ok_rn, rn = pcall(function() return ut:rule_name() end); "
        "            if ok_rn and rn then uname = tostring(rn) end; "
        "            if uname == '' and ut.rule_name then uname = tostring(ut.rule_name) end; "
        "            if uname == '' and ut.name then uname = tostring(ut.name) end; "
        "            if uname == '' and ut.singular then uname = tostring(ut.singular) end; "
        "          end; "
        "          if uname == '' then uname = 'unknown' end; "
        "          uname = uname:gsub('[|;]', '/'); "
        "          local hp = u.hp or 0; "
        "          local moves = u.moves_left or 0; "
        "          parts[#parts + 1] = string.format('%d|%d|%d|%d|%s|%d|%d', u.id, t.nat_x or -1, t.nat_y or -1, owner_id, uname, hp, moves); "
        "        end "
        "      end "
        "    end "
        "  end "
        "end; "
        "return table.concat(parts, ';')"
    )
    result = lr.eval(lua)
    payload = result.last_return()
    units: List[Tuple[int, int, int, int, str, int, int]] = []
    if payload:
        for chunk in payload.split(";"):
            if not chunk:
                continue
            parts = chunk.split("|", 6)
            if len(parts) != 7:
                continue
            try:
                uid = int(parts[0])
                nx = int(parts[1])
                ny = int(parts[2])
                owner = int(parts[3])
                name = parts[4]
                hp = int(float(parts[5]))
                moves = int(float(parts[6]))
                units.append((uid, nx, ny, owner, name, hp, moves))
            except ValueError:
                continue
    return units


def list_all_cities(lr: LuaRemoteClient) -> List[Tuple[int, int, int, int, str]]:
    lua = (
        "local parts = {}; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local ir = pl.cities_iterate and pl:cities_iterate() or nil; "
        "    if ir then "
        "      while true do "
        "        local c = ir(); "
        "        if not c then break end; "
        "        local t = c.tile; "
        "        if t then "
        "          local owner_id = -1; "
        "          if pl.id then owner_id = pl.id elseif pl.player_num then owner_id = pl.player_num end; "
        "          local cname = c.name or ''; "
        "          cname = cname:gsub('|','/'); "
        "          parts[#parts + 1] = string.format('%d|%d|%d|%d|%s', c.id, t.nat_x or -1, t.nat_y or -1, owner_id, cname); "
        "        end "
        "      end "
        "    end "
        "  end "
        "end; "
        "return table.concat(parts, ';')"
    )
    result = lr.eval(lua)
    payload = result.last_return()
    cities: List[Tuple[int, int, int, int, str]] = []
    if payload:
        for entry in payload.split(";"):
            if not entry:
                continue
            parts = entry.split("|", 4)
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
                nx = int(parts[1])
                ny = int(parts[2])
                owner = int(parts[3])
                name = parts[4]
            except ValueError:
                continue
            cities.append((cid, nx, ny, owner, name))
    return cities


def list_player_scores(lr: LuaRemoteClient) -> Dict[int, Tuple[Optional[float], Optional[bool], str]]:
    lua = (
        "local parts = {}; "
        "local ps = game.list_players and game.list_players() or nil; "
        "local function emit(pl) "
        "  if not pl then return end; "
        "  local pid = pl.id or pl.player_num or -1; "
        "  local pname = pl.name or ''; "
        "  pname = tostring(pname):gsub('[|;]','/'); "
        "  local score=nil; "
        "  if pl.score_game then score = pl:score_game() end; "
        "  local win=nil; "
        "  if pl.is_winner then win = pl:is_winner() end; "
        "  parts[#parts + 1] = string.format('%d|%s|%s|%s', pid, tostring(score), tostring(win), pname); "
        "end; "
        "if type(ps) == 'table' then "
        "  for _, pl in ipairs(ps) do emit(pl) end "
        "else "
        "  for i=0,63 do "
        "    local pl = find.player and find.player(i); "
        "    if pl then emit(pl) end "
        "  end "
        "end; "
        "return table.concat(parts, ';')"
    )
    result = lr.eval(lua)
    payload = result.last_return()
    scores: Dict[int, Tuple[Optional[float], Optional[bool], str]] = {}
    if payload:
        for entry in payload.split(";"):
            if not entry:
                continue
            parts = entry.split("|", 3)
            if len(parts) != 4:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            raw_score = parts[1].strip().lower()
            raw_win = parts[2].strip().lower()
            name = parts[3]
            if raw_score in {"nil", "none", ""}:
                score_val = None
            else:
                try:
                    score_val = float(raw_score)
                except ValueError:
                    score_val = None
            if raw_win in {"true", "false"}:
                win_val = raw_win == "true"
            else:
                win_val = None
            scores[pid] = (score_val, win_val, name)
    return scores


def list_city_sizes(lr: LuaRemoteClient) -> Dict[int, int]:
    lua = (
        "local parts = {}; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local ir = pl.cities_iterate and pl:cities_iterate() or nil; "
        "    if ir then "
        "      while true do "
        "        local c = ir(); "
        "        if not c then break end; "
        "        local sz = c.size; "
        "        if sz == nil then "
        "          local ok, v = pcall(function() return c:size() end); "
        "          if ok then sz = v end "
        "        end; "
        "        if sz == nil then "
        "          local ok, v = pcall(function() return c:population() end); "
        "          if ok then sz = v end "
        "        end; "
        "        if sz == nil then sz = -1 end; "
        "        parts[#parts + 1] = string.format('%d|%d', c.id, sz); "
        "      end "
        "    end "
        "  end "
        "end; "
        "return table.concat(parts, ';')"
    )
    result = lr.eval(lua)
    payload = result.last_return()
    sizes: Dict[int, int] = {}
    if payload:
        for entry in payload.split(";"):
            if not entry:
                continue
            parts = entry.split("|", 1)
            if len(parts) != 2:
                continue
            try:
                cid = int(parts[0])
                size = int(parts[1])
            except ValueError:
                continue
            if size >= 0:
                sizes[cid] = size
    return sizes


def list_tile_owners(lr: LuaRemoteClient, map_w: int, map_h: int) -> Dict[Tuple[int, int], int]:
    lua = (
        "local parts = {}; "
        f"for x=0,{int(map_w) - 1} do "
        f"  for y=0,{int(map_h) - 1} do "
        "    local t = find.tile and find.tile(x, y); "
        "    if t then "
        "      local pl = t.owner; "
        "      if pl then "
        "        local owner_id = -1; "
        "        if pl.id then owner_id = pl.id elseif pl.player_num then owner_id = pl.player_num end; "
        "        if owner_id and owner_id >= 0 then "
        "          parts[#parts + 1] = string.format('%d|%d|%d', x, y, owner_id); "
        "        end "
        "      end "
        "    end "
        "  end "
        "end; "
        "return table.concat(parts, ';')"
    )
    result = lr.eval(lua)
    payload = result.last_return()
    owners: Dict[Tuple[int, int], int] = {}
    if payload:
        for entry in payload.split(";"):
            if not entry:
                continue
            parts = entry.split("|", 2)
            if len(parts) != 3:
                continue
            try:
                x = int(parts[0])
                y = int(parts[1])
                owner = int(parts[2])
            except ValueError:
                continue
            owners[(x, y)] = owner
    return owners


def list_city_walls(lr: LuaRemoteClient) -> Dict[int, bool]:
    lua = (
        "local parts = {}; "
        "local bt = nil; "
        "if find.building_type then bt = find.building_type('City Walls') end; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local ir = pl.cities_iterate and pl:cities_iterate() or nil; "
        "    if ir then "
        "      while true do "
        "        local c = ir(); "
        "        if not c then break end; "
        "        local has = 0; "
        "        if bt and c.has_building then "
        "          if c:has_building(bt) then has = 1 end; "
        "        end; "
        "        if c.id then "
        "          parts[#parts + 1] = string.format('%d|%d', c.id, has); "
        "        end "
        "      end "
        "    end "
        "  end "
        "end; "
        "return table.concat(parts, ';')"
    )
    result = lr.eval(lua)
    payload = result.last_return()
    walls: Dict[int, bool] = {}
    if payload:
        for entry in payload.split(";"):
            if not entry:
                continue
            parts = entry.split("|", 1)
            if len(parts) != 2:
                continue
            try:
                cid = int(parts[0])
                flag = int(parts[1])
            except ValueError:
                continue
            walls[cid] = bool(flag)
    return walls


def list_city_buildings(lr: LuaRemoteClient, building_names: Iterable[str]) -> Dict[int, set[str]]:
    names = [name for name in building_names if name]
    if not names:
        return {}
    lua_names = ",".join(_lua_quote(name) for name in names)
    lua = (
        "local parts = {}; "
        f"local names = {{{lua_names}}}; "
        "local types = {}; "
        "if find.building_type then "
        "  for i=1,#names do "
        "    types[i] = find.building_type(names[i]); "
        "  end "
        "end; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local ir = pl.cities_iterate and pl:cities_iterate() or nil; "
        "    if ir then "
        "      while true do "
        "        local c = ir(); "
        "        if not c then break end; "
        "        local built = {}; "
        "        if c.has_building then "
        "          for idx=1,#names do "
        "            local bt = types[idx]; "
        "            if bt then "
        "              local ok, res = pcall(function() return c:has_building(bt) end); "
        "              if ok and res then built[#built + 1] = names[idx] end; "
        "            end "
        "          end "
        "        end; "
        "        local bid = c.id or -1; "
        "        parts[#parts + 1] = string.format('%d|%s', bid, table.concat(built, ',')); "
        "      end "
        "    end "
        "  end "
        "end; "
        "return table.concat(parts, ';')"
    )
    result = lr.eval(lua)
    payload = result.last_return()
    out: Dict[int, set[str]] = {}
    if payload:
        for entry in payload.split(";"):
            if not entry:
                continue
            parts = entry.split("|", 1)
            if len(parts) != 2:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            names_raw = parts[1].strip()
            if names_raw:
                built = {name for name in names_raw.split(",") if name}
            else:
                built = set()
            out[cid] = built
    return out


def list_city_adjacent_water(lr: LuaRemoteClient) -> Dict[int, Dict[str, bool]]:
    lua = (
        "local parts = {}; "
        "local you = nil; "
        "if client and client.conn then "
        "  if client.conn.playing then you = client.conn.playing "
        "  elseif client.conn.player then you = client.conn.player end "
        "end; "
        "local you_id = -1; "
        "if you then if you.id then you_id = you.id elseif you.player_num then you_id = you.player_num end end; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local pid = -1; "
        "    if pl.id then pid = pl.id elseif pl.player_num then pid = pl.player_num end; "
        "    if you_id >= 0 and pid ~= you_id then goto skip_player end; "
        "    local ir = pl.cities_iterate and pl:cities_iterate() or nil; "
        "    if ir then "
        "      while true do "
        "        local c = ir(); "
        "        if not c then break end; "
        "        local has_river = 0; "
        "        local has_lake = 0; "
        "        local ct = c.tile; "
        "        if ct then "
        "          local idx = -1; "
        "          while true do "
        "            idx = methods_private.Tile.next_outward_index(ct, idx, 1); "
        "            if idx < 0 then break end; "
        "            local t = methods_private.Tile.tile_for_outward_index(ct, idx); "
        "            if t then "
        "              if t.has_extra then "
        "                local ok, res = pcall(function() return t:has_extra('River') end); "
        "                if ok and res then has_river = 1 end; "
        "              end; "
        "              local terr = t.terrain; "
        "              if terr then "
        "                local ok_tn, tn = pcall(function() return terr:rule_name() end); "
        "                if ok_tn and tn and string.lower(tn) == 'lake' then has_lake = 1 end; "
        "              end; "
        "            end; "
        "            if has_river == 1 and has_lake == 1 then break end; "
        "          end; "
        "        end; "
        "        local cid = c.id or -1; "
        "        parts[#parts + 1] = string.format('%d|%d|%d', cid, has_river, has_lake); "
        "      end "
        "    end "
        "  end "
        "  ::skip_player:: "
        "end; "
        "return table.concat(parts, ';')"
    )
    result = lr.eval(lua)
    payload = result.last_return()
    out: Dict[int, Dict[str, bool]] = {}
    if payload:
        for entry in payload.split(";"):
            if not entry:
                continue
            parts = entry.split("|")
            if len(parts) != 3:
                continue
            try:
                cid = int(parts[0])
                has_river = bool(int(parts[1]))
                has_lake = bool(int(parts[2]))
            except ValueError:
                continue
            out[cid] = {"river": has_river, "lake": has_lake}
    return out


def list_units_by_homecity(lr: LuaRemoteClient) -> Dict[int, Dict[str, int]]:
    lua = (
        "local parts = {}; "
        "local you = nil; "
        "if client and client.conn then "
        "  if client.conn.playing then you = client.conn.playing "
        "  elseif client.conn.player then you = client.conn.player end "
        "end; "
        "local you_id = -1; "
        "if you then if you.id then you_id = you.id elseif you.player_num then you_id = you.player_num end end; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local pid = -1; "
        "    if pl.id then pid = pl.id elseif pl.player_num then pid = pl.player_num end; "
        "    if you_id >= 0 and pid ~= you_id then goto skip_player end; "
        "    local ir = pl.units_iterate and pl:units_iterate() or nil; "
        "    if ir then "
        "      while true do "
        "        local u = ir(); "
        "        if not u then break end; "
        "        local ut = nil; "
        "        local ok_ut, res_ut = pcall(function() return u:utype() end); "
        "        if ok_ut and res_ut then ut = res_ut else ut = u.utype end; "
        "        local uname = ''; "
        "        if ut then "
        "          local ok_rn, rn = pcall(function() return ut:rule_name() end); "
        "          if ok_rn and rn then uname = tostring(rn) end; "
        "          if uname == '' and ut.rule_name then uname = tostring(ut.rule_name) end; "
        "          if uname == '' and ut.name then uname = tostring(ut.name) end; "
        "          if uname == '' and ut.singular then uname = tostring(ut.singular) end; "
        "        end; "
        "        if uname == '' then uname = 'unknown' end; "
        "        uname = uname:gsub('[|;]', '/'); "
        "        local home = -1; "
        "        if u.get_homecity then "
        "          local ok_h, hc = pcall(function() return u:get_homecity() end); "
        "          if ok_h and hc and hc.id then home = hc.id end; "
        "        elseif u.homecity then "
        "          home = u.homecity "
        "        end; "
        "        parts[#parts + 1] = string.format('%d|%s', home or -1, uname); "
        "      end "
        "    end "
        "  end "
        "  ::skip_player:: "
        "end; "
        "return table.concat(parts, ';')"
    )
    result = lr.eval(lua)
    payload = result.last_return()
    out: Dict[int, Dict[str, int]] = {}
    if payload:
        for entry in payload.split(";"):
            if not entry:
                continue
            parts = entry.split("|", 1)
            if len(parts) != 2:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            name = parts[1].strip()
            if not name:
                continue
            bucket = out.setdefault(cid, {})
            bucket[name] = bucket.get(name, 0) + 1
    return out
