#!/usr/bin/env bash
set -euo pipefail

REMOTE="${GOOGLE_DRIVE_REMOTE:-gdrive:}"
MOUNT_DIR="${GOOGLE_DRIVE_MOUNT_DIR:-${HOME}/gdrive}"
RESULTS_DIR="${GOOGLE_DRIVE_RESULTS_DIR:-${MOUNT_DIR}/freeciv_muzero/results}"
RCLONE="${RCLONE:-rclone}"
LOG_FILE="${GOOGLE_DRIVE_MOUNT_LOG:-/tmp/freeciv-muzero-rclone-mount.log}"

if ! command -v "${RCLONE}" >/dev/null 2>&1; then
  echo "rclone not found. Install it or set RCLONE=/path/to/rclone." >&2
  exit 1
fi

if ! "${RCLONE}" listremotes | grep -Fxq "${REMOTE}"; then
  cat >&2 <<EOF
Missing rclone remote: ${REMOTE}

Create it with:
  rclone config create ${REMOTE%:} drive config_is_local=false

If this host has no browser, run on a browser-capable machine:
  rclone authorize "drive"

Then paste the returned token into the config prompt on this host.
EOF
  exit 1
fi

mkdir -p "${MOUNT_DIR}"

if ! ls "${MOUNT_DIR}" >/dev/null 2>&1; then
  fusermount3 -u "${MOUNT_DIR}" 2>/dev/null || fusermount -u "${MOUNT_DIR}" 2>/dev/null || true
fi

if mountpoint -q "${MOUNT_DIR}"; then
  echo "Already mounted: ${MOUNT_DIR}"
else
  nohup "${RCLONE}" mount "${REMOTE}" "${MOUNT_DIR}" \
    --vfs-cache-mode writes \
    --dir-cache-time 1m \
    --poll-interval 1m \
    >"${LOG_FILE}" 2>&1 &
  for _ in $(seq 1 30); do
    mountpoint -q "${MOUNT_DIR}" && break
    sleep 1
  done
fi

if ! mountpoint -q "${MOUNT_DIR}"; then
  echo "Mount failed: ${MOUNT_DIR}" >&2
  echo "Log: ${LOG_FILE}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "Mounted ${REMOTE} at ${MOUNT_DIR}"
echo "Use:"
echo "  GOOGLE_DRIVE_RESULTS=\"${RESULTS_DIR}\" ./scripts/train_headless.sh"
