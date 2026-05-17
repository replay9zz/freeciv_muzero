#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $(basename "$0") results/recordings/file.mp4 [more.mp4 ...]" >&2
  exit 2
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is required. Install git-lfs first." >&2
  exit 1
fi

git lfs install --local >/dev/null

git add .gitattributes
for path in "$@"; do
  case "${path}" in
    results/recordings/*.mp4|results/recordings/**/*.mp4) ;;
    *)
      echo "Refusing non-recording mp4 path: ${path}" >&2
      echo "Expected path under results/recordings/ ending in .mp4" >&2
      exit 2
      ;;
  esac
  if [ ! -f "${path}" ]; then
    echo "Recording not found: ${path}" >&2
    exit 1
  fi
  git add -f "${path}"
done

git lfs ls-files | grep -Ff <(printf '%s\n' "$@" | sed 's#^\./##') || true
