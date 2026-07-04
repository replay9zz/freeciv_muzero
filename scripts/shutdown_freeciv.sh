#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SIGNAL="${1:-TERM}"
case "${SIGNAL}" in
  TERM|KILL) ;;
  *) echo "Usage: $(basename "$0") [TERM|KILL]" >&2; exit 1 ;;
esac

PORTS="${FREECIV_SHUTDOWN_PORTS:-5566 5567 5568 5569 5570 4451 4452 4453 4454 4455}"
PATTERNS=(
  "freeciv-gtk3.22"
  "freeciv-server"
  "freeciv-client"
  "Xvfb :101"
  "Xvfb :102"
  "Xvfb :103"
  "python muzero.py freeciv_remote"
)

stop_pattern() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "${pattern}" || true)"
  if [ -z "${pids}" ]; then
    return 0
  fi
  echo "Stopping ${pattern}: ${pids}"
  kill "-${SIGNAL}" ${pids} 2>/dev/null || true
}

for pattern in "${PATTERNS[@]}"; do
  stop_pattern "${pattern}"
done

for port in ${PORTS}; do
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "-${SIGNAL}" "${port}/tcp" >/dev/null 2>&1 || true
  fi
done

if [ "${SIGNAL}" = "TERM" ]; then
  sleep "${FREECIV_SHUTDOWN_GRACE:-2}"
  for pattern in "${PATTERNS[@]}"; do
    pids="$(pgrep -f "${pattern}" || true)"
    if [ -n "${pids}" ]; then
      echo "Forcing ${pattern}: ${pids}"
      kill -KILL ${pids} 2>/dev/null || true
    fi
  done
fi

if [ "${FREECIV_SHUTDOWN_RAY:-1}" = "1" ]; then
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  elif [ -x "${ROOT_DIR}/.venv/bin/ray" ]; then
    "${ROOT_DIR}/.venv/bin/ray" stop --force >/dev/null 2>&1 || true
  fi
fi

echo "Remaining Freeciv/Ray processes:"
ps -ef | grep -E 'freeciv-(gtk3\.22|server|client)|Xvfb :10[1-3]|python muzero.py freeciv_remote|ray::|raylet|gcs_server' | grep -v grep || true

echo "Listening Freeciv ports:"
if command -v ss >/dev/null 2>&1; then
  ss -ltnp | grep -E ':(445[1-5]|556[6-9]|5570)\b' || true
fi
