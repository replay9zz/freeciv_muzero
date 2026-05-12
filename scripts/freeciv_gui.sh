#!/usr/bin/env bash
set -euo pipefail

: "${FREECIV_BUILD_DIR:=/home/ubuntu/freeciv_test/freeciv_build_v3_2}"
: "${FREECIV_CLIENT_BIN:=freeciv-gtk3.22}"
: "${FREECIV_CLIENT_ARGS:=}"

cd "${FREECIV_BUILD_DIR}"

extra_args=()
if [[ -n "${FREECIV_CLIENT_ARGS}" ]]; then
  read -r -a extra_args <<< "${FREECIV_CLIENT_ARGS}"
fi

exec ./run.sh "${FREECIV_CLIENT_BIN}" "${extra_args[@]}"
