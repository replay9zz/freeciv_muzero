from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from freeciv_sim.remote.lua_client import LuaRemoteClient
from freeciv_sim.remote.lua_queries import (
    list_all_cities,
    list_all_unit_status,
    list_city_scores,
    list_player_scores,
)


Unit = Dict[str, Any]
City = Dict[str, Any]


def _lua_quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _query_turn(client: LuaRemoteClient) -> int:
    lua = (
        "return (function() "
        "local candidates = {}; "
        "if game then "
        "  if game.current_turn then local ok,res=pcall(function() return game.current_turn() end); if ok then candidates[#candidates+1]=res end end; "
        "  if game.turn then "
        "    if type(game.turn) == 'function' then local ok,res=pcall(function() return game.turn() end); if ok then candidates[#candidates+1]=res end "
        "    else candidates[#candidates+1] = game.turn end "
        "  end "
        "  if game.info then candidates[#candidates+1] = game.info.turn end "
        "end; "
        "if server and server.game_info then candidates[#candidates+1] = server.game_info.turn end; "
        "for _,v in ipairs(candidates) do "
        "  local n = tonumber(v); if n then return tostring(math.floor(n)) end "
        "end; "
        "return '0' "
        "end)()"
    )
    try:
        value = client.eval(lua).last_return()
        return int(float(value or 0))
    except Exception:
        return 0


def _query_players(client: LuaRemoteClient) -> Dict[int, Dict[str, Any]]:
    scores = list_player_scores(client)
    out: Dict[int, Dict[str, Any]] = {}
    for pid, (score, winner, name) in scores.items():
        out[int(pid)] = {"id": int(pid), "name": name, "score": score, "winner": winner}
    return out


def _query_research(client: LuaRemoteClient) -> Dict[int, str]:
    lua = (
        "return (function() "
        "local parts = {}; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl then "
        "    local pid = pl.id or pl.player_num or i; "
        "    local tech = nil; "
        "    local ok,res = pcall(function() if pl.researching then return pl:researching() end return nil end); "
        "    if ok then tech = res end; "
        "    if not tech then local ok2,res2 = pcall(function() return pl.researching end); if ok2 then tech = res2 end end; "
        "    local name = ''; "
        "    if tech then "
        "      local ok3,res3 = pcall(function() return tech:rule_name() end); "
        "      if ok3 and res3 then name = tostring(res3) else name = tostring(tech) end "
        "    end; "
        "    name = name:gsub('[|;]', '/'); "
        "    parts[#parts+1] = string.format('%d|%s', pid, name); "
        "  end "
        "end; "
        "return table.concat(parts, ';') "
        "end)()"
    )
    try:
        payload = client.eval(lua).last_return()
    except Exception:
        return {}
    out: Dict[int, str] = {}
    if not payload:
        return out
    for chunk in payload.split(";"):
        if not chunk:
            continue
        raw_pid, sep, name = chunk.partition("|")
        if not sep:
            continue
        try:
            out[int(raw_pid)] = name
        except ValueError:
            continue
    return out


def _query_city_production(client: LuaRemoteClient) -> Dict[int, Dict[str, str]]:
    lua = (
        "return (function() "
        "local parts = {}; "
        "for i=0,63 do "
        "  local pl = find.player and find.player(i); "
        "  if pl and pl.cities_iterate then "
        "    local ir = pl:cities_iterate(); "
        "    if ir then while true do "
        "      local c = ir(); if not c then break end; "
        "      local kind = ''; local name = ''; "
        "      local target = nil; "
        "      local ok,res = pcall(function() if c.production then return c:production() end return nil end); "
        "      if ok then target = res end; "
        "      if not target then local ok2,res2 = pcall(function() return c.production end); if ok2 then target = res2 end end; "
        "      if target then "
        "        local ok3,res3 = pcall(function() return target:rule_name() end); "
        "        if ok3 and res3 then name = tostring(res3) else name = tostring(target) end; "
        "        local ok4,res4 = pcall(function() return target:type_name() end); "
        "        if ok4 and res4 then kind = tostring(res4) end "
        "      end; "
        "      name = name:gsub('[|;]', '/'); kind = kind:gsub('[|;]', '/'); "
        "      parts[#parts+1] = string.format('%d|%s|%s', c.id or -1, kind, name); "
        "    end end "
        "  end "
        "end; "
        "return table.concat(parts, ';') "
        "end)()"
    )
    try:
        payload = client.eval(lua).last_return()
    except Exception:
        return {}
    out: Dict[int, Dict[str, str]] = {}
    if not payload:
        return out
    for chunk in payload.split(";"):
        parts = chunk.split("|", 2)
        if len(parts) != 3:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        out[cid] = {"kind": parts[1], "name": parts[2]}
    return out


def _snapshot(client: LuaRemoteClient) -> Dict[str, Any]:
    units: List[Unit] = []
    for uid, x, y, owner, unit_type, hp, moves in list_all_unit_status(client):
        units.append(
            {
                "id": uid,
                "x": x,
                "y": y,
                "owner": owner,
                "type": unit_type,
                "hp": hp,
                "moves_left": moves,
            }
        )
    city_scores = {int(c["id"]): c for c in list_city_scores(client)}
    production = _query_city_production(client)
    cities: List[City] = []
    for cid, x, y, owner, name in list_all_cities(client):
        score_info = city_scores.get(cid, {})
        cities.append(
            {
                "id": cid,
                "x": x,
                "y": y,
                "owner": owner,
                "name": name,
                "size": score_info.get("size"),
                "score": score_info.get("score"),
                "production": production.get(cid, {}),
            }
        )
    return {
        "turn": _query_turn(client),
        "players": _query_players(client),
        "research": _query_research(client),
        "units": sorted(units, key=lambda item: item["id"]),
        "cities": sorted(cities, key=lambda item: item["id"]),
    }


def _by_id(items: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(item["id"]): item for item in items if item.get("id") is not None}


def _infer_events(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> List[Dict[str, Any]]:
    if prev is None:
        return []
    events: List[Dict[str, Any]] = []
    prev_units = _by_id(prev.get("units", []))
    cur_units = _by_id(cur.get("units", []))
    prev_cities = _by_id(prev.get("cities", []))
    cur_cities = _by_id(cur.get("cities", []))

    for uid, unit in cur_units.items():
        old = prev_units.get(uid)
        if old is None:
            events.append({"type": "unit_created", "unit": unit})
            continue
        if (old.get("x"), old.get("y")) != (unit.get("x"), unit.get("y")):
            events.append(
                {
                    "type": "unit_moved",
                    "unit_id": uid,
                    "owner": unit.get("owner"),
                    "unit_type": unit.get("type"),
                    "from": [old.get("x"), old.get("y")],
                    "to": [unit.get("x"), unit.get("y")],
                }
            )
    for uid, unit in prev_units.items():
        if uid not in cur_units:
            events.append({"type": "unit_removed", "unit": unit})

    for cid, city in cur_cities.items():
        old = prev_cities.get(cid)
        if old is None:
            owner = city.get("owner")
            x, y = city.get("x"), city.get("y")
            consumed = [
                unit
                for unit in prev_units.values()
                if unit.get("owner") == owner
                and unit.get("x") == x
                and unit.get("y") == y
                and unit.get("id") not in cur_units
            ]
            event = {"type": "city_created", "city": city}
            if consumed:
                event["inferred_action"] = "build_city"
                event["source_unit"] = consumed[0]
            events.append(event)
            continue
        if old.get("size") != city.get("size"):
            events.append(
                {
                    "type": "city_size_changed",
                    "city_id": cid,
                    "owner": city.get("owner"),
                    "from": old.get("size"),
                    "to": city.get("size"),
                }
            )
        if old.get("production") != city.get("production"):
            events.append(
                {
                    "type": "city_production_changed",
                    "city_id": cid,
                    "owner": city.get("owner"),
                    "from": old.get("production"),
                    "to": city.get("production"),
                }
            )
    for cid, city in prev_cities.items():
        if cid not in cur_cities:
            events.append({"type": "city_removed", "city": city})

    prev_research = prev.get("research", {})
    cur_research = cur.get("research", {})
    for pid, research in cur_research.items():
        old = prev_research.get(pid)
        if old != research:
            events.append(
                {
                    "type": "research_changed",
                    "player": int(pid),
                    "from": old,
                    "to": research,
                }
            )
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect built-in Freeciv AI trajectories via LuaRemote.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4451)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--poll-interval", type=float, default=0.5)
    ap.add_argument("--connect-timeout", type=float, default=30.0)
    ap.add_argument("--progress-interval", type=float, default=5.0)
    ap.add_argument("--stuck-warning-seconds", type=float, default=30.0)
    args = ap.parse_args()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = LuaRemoteClient(args.host, args.port, timeout=2.5)
    deadline = time.monotonic() + max(1.0, args.connect_timeout)
    last_exc: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            client.connect()
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5)
    else:
        raise SystemExit(f"LuaRemote connect failed: {last_exc}")

    prev: Optional[Dict[str, Any]] = None
    start_time = time.monotonic()
    last_progress = 0.0
    last_turn = -1
    last_turn_change = start_time
    warned_stuck = False
    try:
        with out_path.open("w", encoding="utf-8") as fp:
            while True:
                snap = _snapshot(client)
                now = time.monotonic()
                turn = int(snap.get("turn") or 0)
                if turn != last_turn:
                    last_turn = turn
                    last_turn_change = now
                    warned_stuck = False
                elif (
                    args.stuck_warning_seconds > 0
                    and not warned_stuck
                    and now - last_turn_change >= args.stuck_warning_seconds
                ):
                    print(
                        f"[trajectory] warning: turn stuck at {turn} for "
                        f"{_format_duration(now - last_turn_change)}; check /observe, /start, and timeout",
                        file=sys.stderr,
                        flush=True,
                    )
                    warned_stuck = True
                if (
                    args.progress_interval > 0
                    and (now - last_progress >= args.progress_interval or prev is None)
                ):
                    elapsed = max(0.0, now - start_time)
                    rate = turn / elapsed if elapsed > 0 else 0.0
                    remaining = max(0, int(args.max_turns) - turn)
                    eta = remaining / rate if rate > 0 else None
                    eta_text = "unknown" if eta is None else _format_duration(eta)
                    print(
                        f"[trajectory] turn={turn}/{args.max_turns} "
                        f"elapsed={_format_duration(elapsed)} eta={eta_text} out={out_path}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_progress = now
                record = {
                    "kind": "snapshot",
                    "time": time.time(),
                    "turn": turn,
                    "snapshot": snap,
                    "events": _infer_events(prev, snap),
                }
                fp.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                fp.flush()
                prev = snap
                if turn >= args.max_turns:
                    break
                time.sleep(max(0.05, args.poll_interval))
    finally:
        client.close()


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
