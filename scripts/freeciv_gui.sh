#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${FREECIV_BUILD_DIR:=${ROOT_DIR}/../freeciv_build_v3_2_uv}"
: "${FREECIV_CLIENT_BIN:=freeciv-gtk3.22}"
: "${FREECIV_CLIENT_ARGS:=}"

cd "${FREECIV_BUILD_DIR}"

extra_args=()
if [[ -n "${FREECIV_CLIENT_ARGS}" ]]; then
  read -r -a extra_args <<< "${FREECIV_CLIENT_ARGS}"
fi

exec ./run.sh "${FREECIV_CLIENT_BIN}" "${extra_args[@]}"
