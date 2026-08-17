import os
import socket
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class EvalResult:
    """Container for a single LuaRemote evaluation."""

    lines: List[str]
    returns: List[str]
    errors: List[str]

    def last_return(self) -> Optional[str]:
        """Convenience helper returning the final captured return value (if any)."""
        return self.returns[-1] if self.returns else None

    def last_error(self) -> Optional[str]:
        """Return the last captured error line (if any)."""
        return self.errors[-1] if self.errors else None


@dataclass(frozen=True)
class CityProductionInfo:
    cost: int
    shield_stock: int
    shield_surplus: int
    turns: int


@dataclass(frozen=True)
class PlayerEconomy:
    tax: int
    luxury: int
    science: int
    max_rate: int
    gold: int


class LuaRemoteClient:
    """
    Minimal client for Freeciv GTK LuaRemote (ENABLE_LUAREMOTE), listening on 127.0.0.1:PORT.
    - Sends one-line Lua chunks
    - Captures console output until an end marker appears
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4444, timeout: float = 2.5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._remote_ready = False
        self._profile_enabled = os.getenv("FREECIV_LUA_PROFILE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._profile_interval = max(
            0, int(os.getenv("FREECIV_LUA_PROFILE_INTERVAL", "100") or "100")
        )
        self._profile_started = time.perf_counter()
        self._profile_durations: Dict[str, List[float]] = defaultdict(list)
        self._profile_last_reported_count = 0

    def connect(self) -> None:
        if self._sock is not None:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        try:
            _ = s.recv(4096)
        except socket.timeout:
            pass
        self._sock = s
        self._remote_ready = False

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            try:
                self._sock.sendall(b"quit\n")
            except Exception:
                pass
            self._sock.close()
        finally:
            self._sock = None
            self._remote_ready = False

        self.print_profile_summary(final=True)

    def eval(
        self,
        lua_code: str,
        end_marker: Optional[str] = "__END__",
        profile_label: Optional[str] = None,
    ) -> EvalResult:
        """Execute Lua and optionally record synchronous round-trip latency."""
        if not self._profile_enabled:
            return self._eval_impl(lua_code, end_marker)

        if profile_label is None:
            try:
                profile_label = sys._getframe(1).f_code.co_name
            except (AttributeError, ValueError):
                profile_label = "unknown"
        started = time.perf_counter()
        try:
            return self._eval_impl(lua_code, end_marker)
        finally:
            self._profile_durations[profile_label].append(time.perf_counter() - started)
            call_count = sum(len(values) for values in self._profile_durations.values())
            if self._profile_interval and call_count % self._profile_interval == 0:
                self.print_profile_summary()

    def _eval_impl(self, lua_code: str, end_marker: Optional[str]) -> EvalResult:
        """
        Execute a Lua one-liner in the client and capture console output until marker.
        Uses chat.base to guarantee mirrored output lines.
        Note: Freeciv may convert __ to ** in chat output, so we check for both patterns.
        """
        if self._sock is None:
            raise RuntimeError("LuaRemoteClient not connected")

        self._ensure_remote_ready()

        if os.getenv("FREECIV_LUA_LOG") == "all":
            safe_line = lua_code.replace("\n", "\\n")
            print(f"[LuaRemote] {safe_line}")

        payload = self._build_payload(lua_code, end_marker)
        self._sock.sendall(payload.encode("utf-8"))

        lines: List[str] = []
        returns: List[str] = []
        errors: List[str] = []
        buf = b""

        expected_markers = []
        if end_marker:
            expected_markers.append(end_marker)
            if "__" in end_marker:
                expected_markers.append(end_marker.replace("__", "**"))

        deadline = None
        if end_marker:
            wait_window = max(self.timeout, 1.0)
            deadline = time.monotonic() + wait_window * 4.0

        while True:
            try:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    s = line.decode("utf-8", errors="replace").strip()

                    if end_marker and s in expected_markers:
                        return EvalResult(lines=lines, returns=returns, errors=errors)

                    if s:
                        indicator, payload_text = self._parse_indicator_line(s)
                        if indicator == "RET":
                            returns.append(payload_text)
                        elif indicator == "ERR":
                            errors.append(payload_text)

                    lines.append(s)
            except socket.timeout:
                if not end_marker:
                    return EvalResult(lines=lines, returns=returns, errors=errors)
                if deadline is not None and time.monotonic() >= deadline:
                    return EvalResult(lines=lines, returns=returns, errors=errors)
                continue

    @staticmethod
    def _percentile(values: List[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
        return ordered[index]

    def print_profile_summary(self, final: bool = False) -> None:
        """Print aggregate LuaRemote RPC timings when profiling is enabled."""
        if not self._profile_enabled:
            return
        all_values = [
            duration
            for values in self._profile_durations.values()
            for duration in values
        ]
        count = len(all_values)
        if not count or (final and count == self._profile_last_reported_count):
            return
        total = sum(all_values)
        wall = max(0.0, time.perf_counter() - self._profile_started)
        mean = total / count
        print(
            "[LuaRemoteProfile] "
            f"{'final' if final else 'progress'} calls={count} "
            f"rpc_total_s={total:.3f} wall_s={wall:.3f} "
            f"rpc_wall_ratio={total / wall if wall else 0.0:.3f} "
            f"mean_ms={mean * 1000:.2f} "
            f"p50_ms={self._percentile(all_values, 0.50) * 1000:.2f} "
            f"p95_ms={self._percentile(all_values, 0.95) * 1000:.2f} "
            f"max_ms={max(all_values) * 1000:.2f}",
            file=sys.stderr,
            flush=True,
        )
        for label, values in sorted(
            self._profile_durations.items(),
            key=lambda item: sum(item[1]),
            reverse=True,
        )[:20]:
            label_total = sum(values)
            print(
                "[LuaRemoteProfile] "
                f"label={label} calls={len(values)} total_s={label_total:.3f} "
                f"mean_ms={label_total / len(values) * 1000:.2f} "
                f"p95_ms={self._percentile(values, 0.95) * 1000:.2f} "
                f"max_ms={max(values) * 1000:.2f}",
                file=sys.stderr,
                flush=True,
            )
        self._profile_last_reported_count = count

    @staticmethod
    def _quote_lua_string(text: str) -> str:
        return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"

    @staticmethod
    def _lua_quote_chunk(code: str) -> str:
        def escape_char(ch: str) -> str:
            if ch == "\\":
                return "\\\\"
            if ch == '"':
                return "\\\""
            if ch == "\n":
                return "\\n"
            if ch == "\r":
                return "\\r"
            if ch == "\t":
                return "\\t"
            if ord(ch) < 32 or ord(ch) == 127:
                return f"\\{ord(ch):03d}"
            return ch

        escaped = "".join(escape_char(ch) for ch in code)
        return '"' + escaped + '"'

    def _ensure_remote_ready(self) -> None:
        if self._remote_ready:
            return

        setup = (
            "if not __lr_exec then "
            "local function __lr_emit(s) if not s then return end if chat and chat.base then chat.base(s) end if log and log.normal then log.normal(s) end end "
            "local function __lr_ts(v) if v==nil then return 'nil' end local ok,res=pcall(tostring,v) if ok then return res else return '<tostring error>' end end "
            "local function __lr_pack(...) return {n=select('#', ...), ...} end "
            "function __lr_exec(chunk, marker) "
            "local f,err=load(chunk,'LuaRemoteChunk','t'); "
            "if not f then __lr_emit('__ERR__ '..__lr_ts(err)); if marker then __lr_emit(marker) end; return end "
            "local ok,res=pcall(function() return __lr_pack(f()) end); "
            "if not ok then __lr_emit('__ERR__ '..__lr_ts(res)); if marker then __lr_emit(marker) end; return end "
            "local n=res.n or #res; if n>0 then for i=1,n do __lr_emit(string.format('__RET__[%d] %s', i, __lr_ts(res[i]))) end end "
            "if marker then __lr_emit(marker) end "
            "end "
            "end"
        )

        self._sock.sendall((setup + "\n").encode("utf-8"))
        self._remote_ready = True

    @staticmethod
    def _parse_indicator_line(text: str) -> Tuple[Optional[str], str]:
        stripped = text.strip()
        for prefix in ("__RET__", "**RET**"):
            if stripped.startswith(prefix):
                payload = stripped[len(prefix):].strip()
                if payload.startswith("["):
                    closing = payload.find("]")
                    if closing != -1:
                        payload = payload[closing + 1 :].strip()
                return "RET", payload
        for prefix in ("__ERR__", "**ERR**"):
            if stripped.startswith(prefix):
                return "ERR", stripped[len(prefix):].strip()
        return None, text

    def _build_payload(self, lua_code: str, end_marker: Optional[str]) -> str:
        chunk = self._lua_quote_chunk(lua_code)
        if end_marker:
            marker = self._quote_lua_string(end_marker)
            return f"__lr_exec({chunk}, {marker})\n"
        return f"__lr_exec({chunk}, nil)\n"

    def _log_action(self, label: str, lua_code: str) -> None:
        if os.getenv("FREECIV_LUA_LOG") == "actions":
            safe_line = lua_code.replace("\n", "\\n")
            print(f"[LuaRemote:{label}] {safe_line}")

    def list_units(self, include_type: bool = False) -> List[Tuple]:
        if include_type:
            entry_fmt = (
                "local ut = nil; "
                "local ok_ut, res_ut = pcall(function() return u:utype() end); "
                "if ok_ut and res_ut then ut = res_ut else ut = u.utype end; "
                "local uname = ''; "
                "if ut then "
                "  local ok_rn, rn = pcall(function() return ut:rule_name() end); "
                "  if ok_rn and rn then uname = tostring(rn) end; "
                "  if uname == '' and ut.rule_name then uname = tostring(ut.rule_name) end; "
                "  if uname == '' and ut.name then uname = tostring(ut.name) end; "
                "  if uname == '' and ut.singular then uname = tostring(ut.singular) end; "
                "end; "
                "if uname == '' then uname = 'unknown' end; "
                "uname = uname:gsub('[|;]', '/'); "
                "parts[#parts + 1] = string.format('%d|%d|%d|%s|%s', u.id or -1, t.nat_x or -1, t.nat_y or -1, tostring(pl.name or ''), uname); "
            )
        else:
            entry_fmt = (
                "parts[#parts + 1] = string.format('%d|%d|%d|%s', u.id or -1, t.nat_x or -1, t.nat_y or -1, tostring(pl.name or '')); "
            )
        lua = (
            "return (function() "
            "local parts = {}; "
            "for i=0,63 do "
            "  local pl = find.player and find.player(i); "
            "  if pl and pl.units_iterate then "
            "    local it = pl:units_iterate(); "
            "    if it then "
            "      while true do "
            "        local u = it(); "
            "        if not u then break end; "
            "        local t = u.tile; "
            "        if t then "
            f"          {entry_fmt}"
            "        end "
            "      end "
            "    end "
            "  end "
            "end; "
            "return table.concat(parts, ';') "
            "end)()"
        )
        result = self.eval(lua)
        out: List[Tuple] = []
        payload = result.last_return()
        if not payload:
            return out
        for chunk in payload.split(";"):
            if not chunk:
                continue
            parts = chunk.split("|")
            expected_parts = 5 if include_type else 4
            if len(parts) < expected_parts:
                continue
            try:
                uid = int(float(parts[0]))
                nx = int(float(parts[1]))
                ny = int(float(parts[2]))
                pname = parts[3]
                if include_type:
                    out.append((uid, nx, ny, pname, parts[4]))
                else:
                    out.append((uid, nx, ny, pname))
            except Exception:
                continue
        return out

    def get_unit_pos(self, unit_id: int) -> Optional[Tuple[int, int]]:
        for uid, nx, ny, _ in self.list_units():
            if uid == unit_id:
                return (nx, ny)
        return None

    def end_turn(self) -> bool:
        lua = (
            "do local ok=false; "
            "if client and client.end_turn then ok=pcall(function() client.end_turn() end) end; "
            "if (not ok) and end_turn then ok=pcall(end_turn) end; "
            "local msg = ok and '__OK__ turn_end' or '__ERR__ turn_end'; log.normal(msg); chat.base(msg) end"
        )
        self._log_action("end_turn", lua)
        result = self.eval(lua)
        return any(s.strip() in ["__OK__ turn_end", "**OK** turn_end"] for s in result.lines)

    def move_dir(self, unit_id: int, dir_str: str) -> bool:
        lua = (
            f"do local d=game.direction('{dir_str}'); "
            f"local id = d and direction.id(d) or nil; "
            f"if id then client.move_dir({unit_id}, id); chat.base('__OK__ move'); log.normal('__OK__ move'); else chat.base('__ERR__ dir'); log.normal('__ERR__ dir'); end end"
        )
        self._log_action("move_dir", lua)
        result = self.eval(lua)
        return any(x.strip() in ["__OK__ move", "**OK** move"] for x in result.lines)

    def action_id(self, rule_name: str) -> Optional[int]:
        safe_name = self._quote_lua_string(rule_name)
        result = self.eval(
            "return (function() "
            f"if not client or not client.action_id then return -1 end; "
            f"return client.action_id({safe_name}) end)()"
        )
        try:
            value = int(result.last_return() or -1)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    def action_target_kind(self, action_id: int) -> Optional[int]:
        result = self.eval(
            "return (function() "
            "if not client or not client.action_target_kind then return -1 end; "
            f"return client.action_target_kind({int(action_id)}) end)()"
        )
        try:
            value = int(result.last_return() or -1)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    def can_action(
        self,
        action_id: int,
        actor_id: int,
        target_id: int,
        sub_target_id: int = -1,
    ) -> bool:
        lua = (
            "return (function() "
            "if not client or not client.can_action then return false end; "
            f"return client.can_action({int(action_id)},{int(actor_id)},"
            f"{int(target_id)},{int(sub_target_id)}) end)()"
        )
        result = self.eval(lua)
        return (result.last_return() or "").strip().lower() == "true"

    def do_action(
        self,
        action_id: int,
        actor_id: int,
        target_id: int,
        sub_target_id: int = -1,
        name: str = "",
    ) -> bool:
        safe_name = self._quote_lua_string(name)
        lua = (
            "do local ok=false; local res=false; "
            "if client and client.do_action then "
            f"ok,res=pcall(function() return client.do_action({int(action_id)},"
            f"{int(actor_id)},{int(target_id)},{int(sub_target_id)},{safe_name}) end) "
            "end; local msg=(ok and res) and '__OK__ action' or '__ERR__ action'; "
            "log.normal(msg); chat.base(msg) end"
        )
        self._log_action("do_action", lua)
        result = self.eval(lua)
        return any(
            line.strip() in ("__OK__ action", "**OK** action")
            for line in result.lines
        )

    def player_economy(self) -> Optional[PlayerEconomy]:
        lua = (
            "return (function() "
            "if not client or not client.tax_rate or not client.luxury_rate "
            "or not client.science_rate or not client.max_rate or not client.gold "
            "then return '__NO_ECONOMY__' end; "
            "return string.format('%d|%d|%d|%d|%d',client.tax_rate(),"
            "client.luxury_rate(),client.science_rate(),client.max_rate(),"
            "client.gold()) end)()"
        )
        result = self.eval(lua)
        payload = result.last_return()
        if not payload or payload.startswith("__") or payload.startswith("**"):
            return None
        try:
            values = [int(value) for value in payload.split("|")]
        except ValueError:
            return None
        if len(values) != 5:
            return None
        return PlayerEconomy(*values)

    def set_rates(self, tax: int, luxury: int, science: int) -> bool:
        lua = (
            "do local ok=false; local res=false; "
            "if client and client.set_rates then "
            f"ok,res=pcall(function() return client.set_rates({int(tax)},"
            f"{int(luxury)},{int(science)}) end) end; "
            "local msg=(ok and res) and '__OK__ rates' or '__ERR__ rates'; "
            "log.normal(msg); chat.base(msg) end"
        )
        self._log_action("set_rates", lua)
        result = self.eval(lua)
        return any(
            line.strip() in ("__OK__ rates", "**OK** rates")
            for line in result.lines
        )

    def player_buildable_names(
        self, player_id: int, kind: str, names: List[str]
    ) -> set[str]:
        """Return directly buildable unit/building names in one Lua round trip."""
        if kind not in ("unit", "building") or not names:
            return set()
        values = ",".join(self._quote_lua_string(name) for name in names)
        finder = "unit_type" if kind == "unit" else "building_type"
        lua = (
            "return (function() "
            f"local pl=find.player and find.player({int(player_id)}); "
            "if not pl or not pl.can_build_direct then return '' end; "
            f"local names={{{values}}}; local out={{}}; "
            "for i,name in ipairs(names) do "
            f"local obj=find.{finder} and find.{finder}(name); "
            "if obj then local ok,res=pcall(function() return pl:can_build_direct(obj) end); "
            "if ok and res then table.insert(out,tostring(i)) end end end; "
            "return table.concat(out,',') end)()"
        )
        result = self.eval(lua)
        payload = result.last_return() or ""
        indices: set[int] = set()
        for raw in payload.split(","):
            try:
                indices.add(int(raw) - 1)
            except ValueError:
                continue
        return {name for index, name in enumerate(names) if index in indices}

    def city_buy(self, city_id: int) -> bool:
        return self._boolean_client_action("city_buy", int(city_id))

    def city_sell(self, city_id: int, building_name: str) -> bool:
        return self._boolean_client_action(
            "city_sell", int(city_id), building_name
        )

    def city_toggle_worker(self, city_id: int, city_x: int, city_y: int) -> bool:
        return self._boolean_client_action(
            "city_toggle_worker", int(city_id), int(city_x), int(city_y)
        )

    def city_change_specialist(
        self, city_id: int, from_name: str, to_name: str
    ) -> bool:
        return self._boolean_client_action(
            "city_change_specialist", int(city_id), from_name, to_name
        )

    def city_rename(self, city_id: int, name: str) -> bool:
        return self._boolean_client_action("city_rename", int(city_id), name)

    def _boolean_client_action(self, method: str, *args: object) -> bool:
        safe_method = "".join(ch for ch in method if ch.isalnum() or ch == "_")
        if safe_method != method:
            raise ValueError("invalid client method")
        encoded = []
        for arg in args:
            if isinstance(arg, str):
                encoded.append(self._quote_lua_string(arg))
            else:
                encoded.append(str(int(arg)))
        lua = (
            "do local ok=false; local res=false; "
            f"if client and client.{safe_method} then "
            f"ok,res=pcall(function() return client.{safe_method}("
            + ",".join(encoded)
            + ") end) end; "
            f"local msg=(ok and res) and '__OK__ {safe_method}' or '__ERR__ {safe_method}'; "
            "log.normal(msg); chat.base(msg) end"
        )
        self._log_action(safe_method, lua)
        result = self.eval(lua)
        return any(
            line.strip() in (f"__OK__ {safe_method}", f"**OK** {safe_method}")
            for line in result.lines
        )

    def move_dir_id(self, unit_id: int, dir_id: int) -> bool:
        lua = (
            f"do local ok=false; "
            f"if client and client.move_dir then ok=pcall(function() client.move_dir({int(unit_id)}, {int(dir_id)}) end) end; "
            f"local msg = ok and '__OK__ move' or '__ERR__ move'; log.normal(msg); chat.base(msg) end"
        )
        self._log_action("move_dir_id", lua)
        result = self.eval(lua)
        return any(x.strip() in ["__OK__ move", "**OK** move"] for x in result.lines)

    def unit_activity(self, unit_id: int, activity_name: str) -> bool:
        safe_name = self._quote_lua_string(activity_name)
        lua = (
            "do local ok=false; local res=false; "
            f"local uid={int(unit_id)}; local activity={safe_name}; "
            "if client and client.unit_activity then "
            "  ok,res=pcall(function() return client.unit_activity(uid, activity) end) "
            "end; "
            "local msg = (ok and res) and '__OK__ unit_activity' or '__ERR__ unit_activity'; "
            "log.normal(msg); chat.base(msg) end"
        )
        self._log_action("unit_activity", lua)
        result = self.eval(lua)
        return any(
            x.strip() in ["__OK__ unit_activity", "**OK** unit_activity"]
            for x in result.lines
        )

    def unit_activity_masks(
        self,
        unit_ids: List[int],
        activity_names: List[str],
    ) -> Dict[int, Tuple[int, int]]:
        if not unit_ids or not activity_names:
            return {}
        ids = ",".join(str(int(unit_id)) for unit_id in unit_ids)
        names = ",".join(self._quote_lua_string(name) for name in activity_names)
        lua = (
            "return (function() "
            f"local ids={{{ids}}}; local activities={{{names}}}; local parts={{}}; "
            "for _,uid in ipairs(ids) do "
            "  local mask=0; "
            "  for idx,activity in ipairs(activities) do "
            "    local ok,res=pcall(function() "
            "      if client and client.can_unit_activity then "
            "        return client.can_unit_activity(uid, activity) "
            "      end; return false "
            "    end); "
            "    if ok and res then mask=mask + 2^(idx-1) end "
            "  end; "
            "  local current=-1; "
            "  local current_ok,current_res=pcall(function() "
            "    if client and client.unit_activity_id then "
            "      return client.unit_activity_id(uid) "
            "    end; return -1 "
            "  end); "
            "  if current_ok and current_res then current=current_res end; "
            "  table.insert(parts, string.format('%d:%d:%d', uid, mask, current)) "
            "end; "
            "return '__UNITACT__ '..table.concat(parts, ',') "
            "end)()"
        )
        self._log_action("unit_activity_masks", lua)
        try:
            result = self.eval(lua)
            value = result.last_return()
        except Exception:
            return {}
        if not isinstance(value, str) or "__UNITACT__" not in value:
            return {}
        masks: Dict[int, Tuple[int, int]] = {}
        payload = value.split("__UNITACT__", 1)[1].strip()
        for item in payload.split(","):
            try:
                unit_id, mask, current = item.split(":", 2)
                masks[int(unit_id)] = (int(mask), int(current))
            except (TypeError, ValueError):
                continue
        return masks

    def build_city(self, unit_id: int) -> bool:
        lua = (
            "do local ok=false; "
            f"local uid={int(unit_id)}; "
            "if client and client.build_city then ok=pcall(function() client.build_city(uid) end) end; "
            "local msg = ok and '__OK__ build_city' or '__ERR__ build_city'; "
            "log.normal(msg); chat.base(msg) end"
        )
        self._log_action("build_city", lua)
        result = self.eval(lua)
        return any(x.strip() in ["__OK__ build_city", "**OK** build_city"] for x in result.lines)

    def found_city(self, unit_id: int, name: str) -> bool:
        safe_name = self._quote_lua_string(name or "")
        lua = (
            "do local ok=false; "
            f"local uid={int(unit_id)}; local cname={safe_name}; "
            "if client and client.found_city then "
            "  ok=pcall(function() client.found_city(uid, cname) end) "
            "end; "
            "local msg = ok and '__OK__ found_city' or '__ERR__ found_city'; "
            "log.normal(msg); chat.base(msg) end"
        )
        self._log_action("found_city", lua)
        result = self.eval(lua)
        return any(x.strip() in ["__OK__ found_city", "**OK** found_city"] for x in result.lines)

    def set_city_production(self, city_id: int, kind: str, rule_name: str) -> bool:
        safe_kind = self._quote_lua_string(kind or "UnitType")
        safe_name = self._quote_lua_string(rule_name or "")
        lua = (
            "do local ok=false; local res=false; "
            f"local cid={int(city_id)}; "
            f"local kind={safe_kind}; "
            f"local uname={safe_name}; "
            "if client and client.set_city_production then "
            "  ok,res=pcall(function() return client.set_city_production(cid, kind, uname) end) "
            "end; "
            "local msg = (ok and res) and '__OK__ city_prod' or '__ERR__ city_prod'; "
            "log.normal(msg); chat.base(msg) end"
        )
        self._log_action("set_city_production", lua)
        result = self.eval(lua)
        return any(x.strip() in ["__OK__ city_prod", "**OK** city_prod"] for x in result.lines)

    def queue_city_production(
        self, city_id: int, kind: str, rule_name: str, position: int = -1
    ) -> bool:
        safe_kind = self._quote_lua_string(kind or "UnitType")
        safe_name = self._quote_lua_string(rule_name or "")
        lua = (
            "do local ok=false; local res=false; "
            f"local cid={int(city_id)}; "
            f"local kind={safe_kind}; "
            f"local uname={safe_name}; "
            f"local pos={int(position)}; "
            "if client and client.queue_city_production then "
            "  ok,res=pcall(function() return client.queue_city_production(cid, kind, uname, pos) end) "
            "end; "
            "local msg = (ok and res) and '__OK__ city_queue' or '__ERR__ city_queue'; "
            "log.normal(msg); chat.base(msg) end"
        )
        self._log_action("queue_city_production", lua)
        result = self.eval(lua)
        return any(x.strip() in ["__OK__ city_queue", "**OK** city_queue"] for x in result.lines)

    def city_production_info(
        self,
        city_id: int,
        kind: str,
        rule_name: str,
        *,
        include_shield_stock: bool = True,
    ) -> Optional[CityProductionInfo]:
        """Return Freeciv's exact current-state production estimate."""
        safe_kind = self._quote_lua_string(kind or "UnitType")
        safe_name = self._quote_lua_string(rule_name or "")
        include_stock = "true" if include_shield_stock else "false"
        lua = (
            "return (function() "
            f"local cid={int(city_id)}; local kind={safe_kind}; local uname={safe_name}; "
            "if not client or not client.city_production_cost "
            "or not client.city_production_turns or not client.city_shield_stock "
            "or not client.city_shield_surplus then return '__NO_CITY_PROD_INFO__' end; "
            "local ok1,cost=pcall(function() return client.city_production_cost(cid,kind,uname) end); "
            "local ok2,stock=pcall(function() return client.city_shield_stock(cid) end); "
            "local ok3,surplus=pcall(function() return client.city_shield_surplus(cid) end); "
            f"local ok4,turns=pcall(function() return client.city_production_turns(cid,kind,uname,{include_stock}) end); "
            "if not (ok1 and ok2 and ok3 and ok4) then return '__ERR_CITY_PROD_INFO__' end; "
            "return string.format('%d|%d|%d|%d',cost,stock,surplus,turns) end)()"
        )
        result = self.eval(lua)
        payload = result.last_return()
        if not payload or payload.startswith("__") or payload.startswith("**"):
            return None
        parts = payload.split("|")
        if len(parts) != 4:
            return None
        try:
            cost, stock, surplus, turns = (int(value) for value in parts)
        except ValueError:
            return None
        if cost < 0 or stock < 0 or turns < 0:
            return None
        return CityProductionInfo(cost, stock, surplus, turns)

    def attack_dir_id(self, unit_id: int, dir_id: int) -> bool:
        lua = (
            "do local ok=false; "
            f"local uid={int(unit_id)}; local id={int(dir_id)}; "
            "if client and client.attack_dir then ok=pcall(function() client.attack_dir(uid, id) end) end; "
            "local msg = ok and '__OK__ attack' or '__ERR__ attack'; log.normal(msg); chat.base(msg) end"
        )
        self._log_action("attack_dir_id", lua)
        result = self.eval(lua)
        return any(s.strip() in ["__OK__ attack", "**OK** attack"] for s in result.lines)

    def attack_dir(self, unit_id: int, dir_str: str) -> bool:
        lua = (
            "do local ok=false; "
            f"local uid={int(unit_id)}; local d=game.direction('{dir_str}'); "
            "local id = d and direction.id(d) or nil; "
            "if id and client and client.attack_dir then ok=pcall(function() client.attack_dir(uid, id) end) end; "
            "local msg = ok and '__OK__ attack' or '__ERR__ attack'; log.normal(msg); chat.base(msg) end"
        )
        self._log_action("attack_dir", lua)
        result = self.eval(lua)
        return any(s.strip() in ["__OK__ attack", "**OK** attack"] for s in result.lines)

    def attack_target(self, unit_id: int, nat_x: int, nat_y: int) -> bool:
        lua = (
            "do local ok=false; "
            f"local uid={int(unit_id)}; local nx={int(nat_x)}; local ny={int(nat_y)}; "
            "if client and client.attack then ok=pcall(function() client.attack(uid, nx, ny) end) end; "
            "local msg = ok and '__OK__ attack' or '__ERR__ attack'; log.normal(msg); chat.base(msg) end"
        )
        self._log_action("attack_target", lua)
        result = self.eval(lua)
        return any(s.strip() in ["__OK__ attack", "**OK** attack"] for s in result.lines)

    def conquer_city(self, unit_id: int, city_id: int) -> bool:
        lua = (
            "do local ok=false; "
            f"local uid={int(unit_id)}; local cid={int(city_id)}; "
            "if client and client.conquer_city then "
            "  ok=pcall(function() client.conquer_city(uid, cid) end) "
            "end; "
            "local msg = ok and '__OK__ conquer_city' or '__ERR__ conquer_city'; "
            "log.normal(msg); chat.base(msg) end"
        )
        self._log_action("conquer_city", lua)
        result = self.eval(lua)
        return any(s.strip() in ["__OK__ conquer_city", "**OK** conquer_city"] for s in result.lines)

    def neighbors_status(
        self, unit_id: int, coords: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, str, bool, str, bool, bool, bool]]:
        if not coords:
            return []

        flat = " ".join(f"{x},{y}" for x, y in coords)
        lua = (
            f"local coords='{flat}'; "
            "local cw = nil; "
            "if find and find.building_type then "
            "  local ok, res = pcall(find.building_type, 'City Walls'); "
            "  if ok then cw = res end "
            "end; "
            "local function find_unit_by_id(id) "
            "  for i=0,63 do "
            "    local pl = find.player and find.player(i); "
            "    if pl then "
            "      local ir = pl.units_iterate and pl:units_iterate() or nil; "
            "      if ir then "
            "        while true do "
            "          local u = ir(); "
            "          if not u then break end; "
            "          if u.id == id then return u, pl end "
            "        end "
            "      end "
            "    end "
            "  end "
            "  return nil, nil "
            "end; "
            f"local unit, owner = find_unit_by_id({int(unit_id)}); "
            "if not unit then return '__ERR__ unit' end; "
            "local utype = unit.utype; "
            "local parts = {}; "
            "for tok in string.gmatch(coords, '([^ ]+)') do "
            "  local sx, sy = string.match(tok, '([^,]+),([^,]+)'); "
            "  local nx = tonumber(sx) or -1; "
            "  local ny = tonumber(sy) or -1; "
            "  local tile = nil; "
            "  if find and find.tile then "
            "    local ok, res = pcall(find.tile, nx, ny); "
            "    if ok then tile = res end "
            "  end; "
            "  local terrain = ''; "
            "  if tile and tile.terrain and tile.terrain.name_translation then "
            "    local ok, name = pcall(function() return tile.terrain:name_translation() end); "
            "    if ok and name then terrain = name end "
            "  end; "
            "  local can = true; "
            "  if not tile then can = false end; "
            "  if can and terrain == 'Inaccessible' then can = false end; "
            "  if can and utype and utype.can_exist_at_tile then "
            "    local ok, res = pcall(function() return utype:can_exist_at_tile(tile) end); "
            "    if ok then can = res and true or false end "
            "  elseif can and unit.can_exist_at_tile then "
            "    local ok, res = pcall(function() return unit:can_exist_at_tile(tile) end); "
            "    if ok then can = res and true or false end "
            "  elseif can and Unit_Type and Unit_Type.can_exist_at_tile then "
            "    local ok, res = pcall(Unit_Type.can_exist_at_tile, Unit_Type, utype, tile); "
            "    if ok then can = res and true or false end "
            "  end; "
            "  local enemy = 0; "
            "  local enemy_from_units = 0; "
            "  local friendly_unit = 0; "
            "  local walls = 0; "
            "  if tile and tile.units_iterate then "
            "    local itr = tile:units_iterate(); "
            "    if itr then "
            "      while true do "
            "        local ou = itr(); "
            "        if not ou then break end; "
            "        if owner and ou.owner == owner then friendly_unit = 1 end; "
            "        if owner and ou.owner and ou.owner ~= owner then enemy_from_units = 1 end; "
            "      end "
            "    end "
            "  end; "
            "  if tile and tile.city then "
            "    local okc, city = pcall(tile.city, tile); "
            "    if okc and city and cw and city.has_building then "
            "      local okh, hasw = pcall(city.has_building, city, cw); "
            "      if okh and hasw then walls = 1 end "
            "    end "
            "  end; "
            "  if tile and owner and tile.enemy_tile then "
            "    local ok, res = pcall(function() return tile:enemy_tile(owner) end); "
            "    if ok and res then enemy = 1 end "
            "  end; "
            "  if enemy_from_units == 1 then enemy = 1 end; "
            "  parts[#parts + 1] = string.format('%d|%d|%d|%d|%s|%d|%d|%d', nx, ny, can and 1 or 0, enemy, terrain or '', enemy_from_units, friendly_unit, walls); "
            "end; "
            "return table.concat(parts, ';')"
        )

        result = self.eval(lua)
        payload = result.last_return()
        out: List[Tuple[int, int, str, bool, str, bool, bool, bool]] = []

        if payload and payload != "__ERR__ unit":
            for chunk in payload.split(";"):
                if not chunk:
                    continue
                parts = chunk.split("|")
                if len(parts) < 7:
                    continue
                try:
                    nx = int(parts[0])
                    ny = int(parts[1])
                    au = "A" if parts[2] == "1" else "U"
                    enemy = parts[3] == "1"
                    terrain = parts[4]
                    enemy_from_units = parts[5] == "1"
                    friendly_unit = parts[6] == "1"
                    walls = False
                    if len(parts) > 7:
                        walls = parts[7] == "1"
                    out.append((nx, ny, au, enemy, terrain, enemy_from_units, friendly_unit, walls))
                except ValueError:
                    continue

        return out
