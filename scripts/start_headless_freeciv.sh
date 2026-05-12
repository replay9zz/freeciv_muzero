#!/usr/bin/env bash
set -euo pipefail

scenario_file=""
if [[ $# -gt 0 ]]; then
  scenario_file="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Usage: $(basename "$0") [scenario.sav]" >&2
  exit 1
fi

: "${FREECIV_BUILD_DIR:=/home/ubuntu/freeciv_test/freeciv_build_v3_2}"
: "${FREECIV_CLIENT_BIN:=freeciv-gtk3.22}"
: "${FREECIV_LUAREMOTE_PORT:=4444}"
: "${FREECIV_SERVER_PORT:=}"
: "${FREECIV_START_SCRIPT:=/tmp/freeciv_auto_start.serv}"

if [[ -z "${scenario_file}" ]]; then
  scenario_file="${FREECIV_SCENARIO_FILE:-}"
fi
if [[ -z "${scenario_file}" ]]; then
  echo "Set FREECIV_SCENARIO_FILE or pass a .sav path" >&2
  exit 1
fi

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | tail -n +2 | awk '{print $4}' | grep -q ":${port}$"
    return $?
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN -P -n >/dev/null 2>&1
    return $?
  fi
  return 1
}

base_luaremote_port="${FREECIV_LUAREMOTE_PORT}"
if [[ -z "${FREECIV_SERVER_PORT}" ]]; then
  base_server_port="$((base_luaremote_port + 1000))"
else
  base_server_port="${FREECIV_SERVER_PORT}"
fi

offset=0
luaremote_port="${base_luaremote_port}"
server_port="${base_server_port}"
while port_in_use "${luaremote_port}" || port_in_use "${server_port}"; do
  offset=$((offset + 1))
  luaremote_port=$((base_luaremote_port + offset))
  if [[ -z "${FREECIV_SERVER_PORT}" ]]; then
    server_port=$((luaremote_port + 1000))
  else
    server_port=$((base_server_port + offset))
  fi
done
if [[ "${luaremote_port}" != "${FREECIV_LUAREMOTE_PORT}" ]]; then
  echo "LuaRemote port ${FREECIV_LUAREMOTE_PORT} in use; using ${luaremote_port}" >&2
fi
if [[ -z "${FREECIV_SERVER_PORT}" ]]; then
  FREECIV_SERVER_PORT="${server_port}"
else
  if [[ "${server_port}" != "${FREECIV_SERVER_PORT}" ]]; then
    echo "Freeciv server port ${FREECIV_SERVER_PORT} in use; using ${server_port}" >&2
  fi
  FREECIV_SERVER_PORT="${server_port}"
fi
FREECIV_LUAREMOTE_PORT="${luaremote_port}"
export FREECIV_LUAREMOTE_PORT

cat > "${FREECIV_START_SCRIPT}" <<'EOF'
start
EOF

cd "${FREECIV_BUILD_DIR}"
exec xvfb-run -a ./run.sh "${FREECIV_CLIENT_BIN}" \
  --file "${scenario_file}" \
  --port "${FREECIV_SERVER_PORT}" \
  --read "${FREECIV_START_SCRIPT}"
