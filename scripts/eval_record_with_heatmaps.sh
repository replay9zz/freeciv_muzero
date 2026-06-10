#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RECORD_DIR="${RECORD_DIR:-${ROOT_DIR}/results/evals/${RUN_STAMP}}"
GAMEPLAY_FILE="${GAMEPLAY_FILE:-${RECORD_DIR}/gameplay.mp4}"
COMBINED_FILE="${COMBINED_FILE:-${RECORD_DIR}/heatmaps/gameplay-heatmaps.mp4}"
HEATMAP_TB_DIR="${HEATMAP_TB_DIR:-${RECORD_DIR}/heatmaps/tb}"
HEATMAP_FRAME_DIR="${HEATMAP_FRAME_DIR:-${RECORD_DIR}/heatmaps/frames}"
HEATMAP_METADATA="${HEATMAP_METADATA:-${RECORD_DIR}/heatmaps/heatmap.json}"
HEATMAP_TAGS="${HEATMAP_TAGS:-belief_units,threat,visible_units,territory}"
DISPLAY_SIZE="${DISPLAY_SIZE:-1280x800}"
GAMEPLAY_WIDTH="${GAMEPLAY_WIDTH:-${DISPLAY_SIZE%x*}}"
GAMEPLAY_HEIGHT="${GAMEPLAY_HEIGHT:-${DISPLAY_SIZE#*x}}"
CLIENT_RESOLUTION="${CLIENT_RESOLUTION:-${DISPLAY_SIZE}}"
RECORD_SIZE="${RECORD_SIZE:-${DISPLAY_SIZE}}"
HEATMAP_PANEL_WIDTH="${HEATMAP_PANEL_WIDTH:-640}"
HEATMAP_PANEL_HEIGHT="${HEATMAP_PANEL_HEIGHT:-${GAMEPLAY_HEIGHT}}"
HEATMAP_TILE_SHAPE="${HEATMAP_TILE_SHAPE:-hex}"
HEATMAP_MAP_WIDTH="${HEATMAP_MAP_WIDTH:-${MAP_WIDTH:-32}}"
HEATMAP_MAP_HEIGHT="${HEATMAP_MAP_HEIGHT:-${MAP_HEIGHT:-32}}"
HEATMAP_START_DELAY="${HEATMAP_START_DELAY:-0}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to compose the combined video." >&2
  exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe is required to inspect the gameplay video duration." >&2
  exit 1
fi

if [ ! -x "${PYTHON}" ]; then
  echo "Python not found: ${PYTHON}" >&2
  exit 1
fi

mkdir -p "${RECORD_DIR}" "${HEATMAP_TB_DIR}" "${HEATMAP_FRAME_DIR}"

FREECIV_BELIEF_TENSORBOARD=1 \
FREECIV_BELIEF_TENSORBOARD_HEX="${FREECIV_BELIEF_TENSORBOARD_HEX:-1}" \
FREECIV_BELIEF_TENSORBOARD_INTERVAL="${FREECIV_BELIEF_TENSORBOARD_INTERVAL:-1}" \
FREECIV_BELIEF_TENSORBOARD_DIR="${HEATMAP_TB_DIR}" \
DISPLAY_SIZE="${DISPLAY_SIZE}" \
CLIENT_RESOLUTION="${CLIENT_RESOLUTION}" \
RECORD_SIZE="${RECORD_SIZE}" \
RECORD_FILE="${GAMEPLAY_FILE}" \
"${SCRIPT_DIR}/eval_record.sh" "$@"

"${PYTHON}" "${SCRIPT_DIR}/render_tb_heatmap_panel.py" \
  --logdir "${HEATMAP_TB_DIR}" \
  --out-dir "${HEATMAP_FRAME_DIR}" \
  --tags "${HEATMAP_TAGS}" \
  --width "${HEATMAP_PANEL_WIDTH}" \
  --height "${HEATMAP_PANEL_HEIGHT}" \
  --tile-shape "${HEATMAP_TILE_SHAPE}" \
  --map-width "${HEATMAP_MAP_WIDTH}" \
  --map-height "${HEATMAP_MAP_HEIGHT}" \
  --metadata-out "${HEATMAP_METADATA}"

frame_count="$(find "${HEATMAP_FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.png' | wc -l)"
if [ "${frame_count}" -lt 1 ]; then
  echo "No heatmap frames were produced. Gameplay recording remains at ${GAMEPLAY_FILE}" >&2
  exit 1
fi

duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "${GAMEPLAY_FILE}")"
heatmap_fps="$(awk -v frames="${frame_count}" -v duration="${duration}" -v delay="${HEATMAP_START_DELAY}" 'BEGIN { active = duration - delay; if (active > 0) print frames / active; else print 1 }')"

ffmpeg \
  -hide_banner \
  -loglevel warning \
  -y \
  -i "${GAMEPLAY_FILE}" \
  -framerate "${heatmap_fps}" \
  -i "${HEATMAP_FRAME_DIR}/frame_%06d.png" \
  -filter_complex "[0:v]scale=${GAMEPLAY_WIDTH}:${GAMEPLAY_HEIGHT},setpts=PTS-STARTPTS[game];[1:v]scale=${HEATMAP_PANEL_WIDTH}:${HEATMAP_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,pad=${HEATMAP_PANEL_WIDTH}:${HEATMAP_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,setpts=PTS-STARTPTS,tpad=start_duration=${HEATMAP_START_DELAY}:start_mode=clone[heat];[game][heat]hstack=inputs=2,fps=${RECORD_FPS:-30}[v]" \
  -map "[v]" \
  -an \
  -c:v libx264 \
  -preset veryfast \
  -pix_fmt yuv420p \
  -shortest \
  "${COMBINED_FILE}"

echo "Recorded gameplay to ${GAMEPLAY_FILE}"
echo "Rendered heatmap frames to ${HEATMAP_FRAME_DIR}"
echo "Combined video written to ${COMBINED_FILE}"
