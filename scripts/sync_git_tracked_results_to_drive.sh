#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${GOOGLE_DRIVE_RESULTS:-}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_MEDIA="${INCLUDE_MEDIA:-0}"
INCLUDE_LOGS="${INCLUDE_LOGS:-0}"
MEDIA_ONLY="${MEDIA_ONLY:-0}"
VERBOSE="${VERBOSE:-1}"
YES="${YES:-0}"

if [ -z "${DEST}" ]; then
  echo "Set GOOGLE_DRIVE_RESULTS to an explicit rclone destination." >&2
  exit 2
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found." >&2
  exit 1
fi

cd "${ROOT_DIR}"
tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

if [ "${MEDIA_ONLY}" = "1" ]; then
  git ls-files results | grep -E '\.mp4$' | sed 's#^results/##' >"${tmp}"
elif [ "${INCLUDE_MEDIA}" = "1" ]; then
  git ls-files results | sed 's#^results/##' >"${tmp}"
else
  git ls-files results \
    | grep -Ev '(^|/)heatmaps/frames/' \
    | grep -Ev '(^|/)heatmaps/tb/' \
    | grep -Ev '(^|/)belief_tensorboard/' \
    | grep -Ev '(^|/)observer-home/' \
    | grep -Ev '\.tfevents\.' \
    | grep -Ev '\.png$' \
    | grep -Ev '\.cache-[^/]*$' \
    | grep -Ev '\.mp4$' \
    >"${tmp}.filtered"
  if [ "${INCLUDE_LOGS}" != "1" ]; then
    grep -Ev '\.log$' "${tmp}.filtered" >"${tmp}.nologs"
    mv "${tmp}.nologs" "${tmp}.filtered"
  fi
  sed 's#^results/##' "${tmp}.filtered" >"${tmp}"
fi
if [ ! -s "${tmp}" ]; then
  echo "No git-tracked files under results/."
  exit 0
fi

echo "Syncing git-tracked results to ${DEST}"
echo "Files: $(wc -l <"${tmp}")"
total_bytes="$(
  while IFS= read -r rel; do
    [ -f "results/${rel}" ] && stat -c '%s' "results/${rel}"
  done <"${tmp}" | awk '{sum += $1} END {printf "%.0f", sum}'
)"
total_human="$(numfmt --to=iec --suffix=B "${total_bytes}" 2>/dev/null || printf '%s bytes' "${total_bytes}")"
echo "Size: ${total_human}"
sed 's#^#  results/#' "${tmp}" | head -50
if [ "$(wc -l <"${tmp}")" -gt 50 ]; then
  echo "  ... $(($(wc -l <"${tmp}") - 50)) more"
fi
if [ "${DRY_RUN}" != "1" ] && [ "${YES}" != "1" ]; then
  read -r -p "Upload these files? [y/N] " answer
  case "${answer}" in
    y|Y|yes|YES) ;;
    *) echo "Canceled."; exit 0 ;;
  esac
fi
args=(
  copy results "${DEST}"
  --files-from "${tmp}"
  --create-empty-src-dirs
)
if [ "${VERBOSE}" = "1" ]; then
  args+=(--progress --stats-one-line --log-level INFO)
fi
if [ "${DRY_RUN}" = "1" ]; then
  args+=(--dry-run)
fi
rclone "${args[@]}"
