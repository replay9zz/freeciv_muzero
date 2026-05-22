#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $(basename "$0") TENSORBOARD_LOGDIR [GAMEPLAY_MP4]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOGDIR="$1"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
GAMEPLAY_FILE="${2:-${GAMEPLAY_FILE:-${ROOT_DIR}/results/evals/${RUN_STAMP}/gameplay.mp4}}"
OUT_PREFIX="${OUT_PREFIX:-${ROOT_DIR}/results/evals/${RUN_STAMP}/overlays/threat-overlay}"
FRAME_DIR="${FRAME_DIR:-${OUT_PREFIX}-frames}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"
OPACITY="${OPACITY:-0.70}"
ALPHA_FLOOR="${ALPHA_FLOOR:-32}"
HEATMAP_START_DELAY="${HEATMAP_START_DELAY:-0}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

THREAT_ONLY_MP4="${THREAT_ONLY_MP4:-${OUT_PREFIX}-only.mp4}"
PREVIEW_MP4="${PREVIEW_MP4:-${OUT_PREFIX}-preview.mp4}"
ALPHA_MOV="${ALPHA_MOV:-${OUT_PREFIX}-alpha.mov}"
METADATA_JSON="${METADATA_JSON:-${OUT_PREFIX}.json}"
RENDER_PREVIEW="${RENDER_PREVIEW:-0}"

if [ ! -x "${PYTHON}" ]; then
  echo "Python not found: ${PYTHON}" >&2
  exit 1
fi
if [ ! -f "${GAMEPLAY_FILE}" ]; then
  echo "Gameplay video not found: ${GAMEPLAY_FILE}" >&2
  exit 1
fi

mkdir -p "${FRAME_DIR}" "$(dirname "${OUT_PREFIX}")"

"${PYTHON}" "${SCRIPT_DIR}/export_tb_heatmap_overlay.py" \
  --logdir "${LOGDIR}" \
  --out-dir "${FRAME_DIR}" \
  --tag threat \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --opacity "${OPACITY}" \
  --alpha-floor "${ALPHA_FLOOR}" \
  --metadata-out "${METADATA_JSON}"

frame_count="$(find "${FRAME_DIR}" -maxdepth 1 -type f -name 'threat_rgb_*.png' | wc -l)"
duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "${GAMEPLAY_FILE}")"
heatmap_fps="$(awk -v frames="${frame_count}" -v duration="${duration}" -v delay="${HEATMAP_START_DELAY}" 'BEGIN { active = duration - delay; if (active > 0) print frames / active; else print 1 }')"

ffmpeg \
  -hide_banner \
  -loglevel warning \
  -y \
  -framerate "${heatmap_fps}" \
  -i "${FRAME_DIR}/threat_rgb_%06d.png" \
  -vf "tpad=start_duration=${HEATMAP_START_DELAY}:start_mode=add:color=black" \
  -r 30 \
  -c:v libx264 \
  -preset veryfast \
  -pix_fmt yuv420p \
  "${THREAT_ONLY_MP4}"

if [ "${RENDER_PREVIEW}" = "1" ]; then
  ffmpeg \
    -hide_banner \
    -loglevel warning \
    -y \
    -i "${GAMEPLAY_FILE}" \
    -framerate "${heatmap_fps}" \
    -i "${FRAME_DIR}/threat_overlay_%06d.png" \
    -filter_complex "[1:v]format=rgba,tpad=start_duration=${HEATMAP_START_DELAY}:start_mode=add:color=black@0[heat];[0:v][heat]overlay=0:0:eof_action=repeat[v]" \
    -map "[v]" \
    -an \
    -c:v libx264 \
    -preset veryfast \
    -pix_fmt yuv420p \
    "${PREVIEW_MP4}"
fi

ffmpeg \
  -hide_banner \
  -loglevel warning \
  -y \
  -framerate "${heatmap_fps}" \
  -i "${FRAME_DIR}/threat_overlay_%06d.png" \
  -vf "format=rgba,tpad=start_duration=${HEATMAP_START_DELAY}:start_mode=add:color=black@0" \
  -r 30 \
  -c:v prores_ks \
  -profile:v 4444 \
  -pix_fmt yuva444p10le \
  "${ALPHA_MOV}"

echo "Threat-only preview: ${THREAT_ONLY_MP4}"
if [ "${RENDER_PREVIEW}" = "1" ]; then
  echo "Gameplay overlay preview: ${PREVIEW_MP4}"
fi
echo "Alpha overlay movie: ${ALPHA_MOV}"
echo "Transparent PNG sequence: ${FRAME_DIR}/threat_overlay_%06d.png"
echo "Metadata: ${METADATA_JSON}"
