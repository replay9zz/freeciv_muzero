#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"

SERVER_PORT="${SERVER_PORT:-5566}"
LUA_PORT="${LUA_PORT:-4451}"

cleanup_freeciv_all "${SERVER_PORT}" "${LUA_PORT}"

exec "${SCRIPT_DIR}/eval_gui.sh" "$@"
