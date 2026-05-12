#!/usr/bin/env bash
set -euo pipefail

signal="${1:-TERM}"
case "${signal}" in
  TERM|KILL) ;;
  *) echo "Usage: $(basename "$0") [TERM|KILL]" >&2; exit 1 ;;
esac

pids=$(pgrep -f "freeciv-gtk3.22" || true)
if [[ -z "${pids}" ]]; then
  echo "No freeciv-gtk3.22 processes found."
  exit 0
fi

echo "Stopping freeciv-gtk3.22 PIDs: ${pids}"
kill "-${signal}" ${pids}

if [[ "${signal}" == "TERM" ]]; then
  sleep 1
  remaining=$(pgrep -f "freeciv-gtk3.22" || true)
  if [[ -n "${remaining}" ]]; then
    echo "Still running, forcing kill: ${remaining}"
    kill -KILL ${remaining}
  fi
fi
