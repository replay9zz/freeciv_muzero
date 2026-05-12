#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

default_build_dir() {
  local candidates=(
    "${ROOT_DIR}/freeciv_build_v3_2_uv"
    "${ROOT_DIR}/freeciv_build_v3_2"
    "${ROOT_DIR}/../freeciv_build_v3_2_uv"
    "${ROOT_DIR}/../freeciv_build_v3_2"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -x "${candidate}/run.sh" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${ROOT_DIR}/../freeciv_build_v3_2_uv"
}

default_scenario_path() {
  local candidates=(
    "${ROOT_DIR}/freeciv/data/scenarios/minimal_v4.sav"
    "${ROOT_DIR}/freeciv/scenarios/minimal_v4.sav"
    "${HOME}/.freeciv/scenarios/minimal_v4.sav"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${ROOT_DIR}/freeciv/data/scenarios/minimal_v4.sav"
}

latest_checkpoint() {
  find "${ROOT_DIR}/results/freeciv_remote" -mindepth 2 -maxdepth 2 -type f -name 'model.checkpoint' | sort | tail -n 1
}

init_python_env() {
  cd "${ROOT_DIR}"
  source .venv/bin/activate
}

cleanup_freeciv_ports() {
  local server_port="${1:-5566}"
  local lua_port="${2:-4451}"
  fuser -k -TERM "${server_port}/tcp" >/dev/null 2>&1 || true
  fuser -k -TERM "${lua_port}/tcp" >/dev/null 2>&1 || true
}

cleanup_freeciv_processes() {
  local server_port="${1:-5566}"
  local display_num="${2:-}"
  pkill -f "freeciv-gtk3.22.*-p ${server_port}" >/dev/null 2>&1 || true
  pkill -f "freeciv-server -p ${server_port}" >/dev/null 2>&1 || true
  if [ -n "${display_num}" ]; then
    pkill -f "Xvfb ${display_num}" >/dev/null 2>&1 || true
  fi
}

cleanup_freeciv_all() {
  local server_port="${1:-5566}"
  local lua_port="${2:-4451}"
  local display_num="${3:-}"
  cleanup_freeciv_ports "${server_port}" "${lua_port}"
  cleanup_freeciv_processes "${server_port}" "${display_num}"
}
