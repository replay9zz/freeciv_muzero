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
: "${FREECIV_START_SCRIPT:=/tmp/freeciv_auto_start.serv}"

if [[ -z "${scenario_file}" ]]; then
  scenario_file="${FREECIV_SCENARIO_FILE:-}"
fi
if [[ -z "${scenario_file}" ]]; then
  echo "Set FREECIV_SCENARIO_FILE or pass a .sav path" >&2
  exit 1
fi

export FREECIV_LUAREMOTE_PORT

cat > "${FREECIV_START_SCRIPT}" <<'EOF'
start
EOF

cd "${FREECIV_BUILD_DIR}"
exec xvfb-run -a ./run.sh "${FREECIV_CLIENT_BIN}" \
  --file "${scenario_file}" \
  --read "${FREECIV_START_SCRIPT}"
