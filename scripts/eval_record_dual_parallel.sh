#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

if [ "$#" -gt 0 ]; then
  CHECKPOINT="$1"
  shift
else
  CHECKPOINT="${CHECKPOINT:-$(latest_checkpoint)}"
fi
GPU_LIST="${GPU_LIST:-0,1,2,3,4}"
GAMES="${GAMES:-20}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
GAMES_PER_BATCH="${GAMES_PER_BATCH:-5}"
BASE_SERVER_PORT="${BASE_SERVER_PORT:-5566}"
BASE_LUA_PORT="${BASE_LUA_PORT:-4451}"
BASE_DISPLAY="${BASE_DISPLAY:-102}"
PORT_STRIDE="${PORT_STRIDE:-10}"
DISPLAY_STRIDE="${DISPLAY_STRIDE:-2}"
MAX_TURNS="${MAX_TURNS:-300}"
RECORD_FPS="${RECORD_FPS:-5}"
DISPLAY_SIZE="${DISPLAY_SIZE:-1920x1080}"
CLIENT_RESOLUTION="${CLIENT_RESOLUTION:-${DISPLAY_SIZE}}"
RUN_HEIGHT="${RUN_HEIGHT:-${CLIENT_RESOLUTION#*x}}"
RUN_LABEL="${RUN_LABEL:-eval-${MAX_TURNS}t-${GAMES}g-${RUN_HEIGHT}p${RECORD_FPS}fps}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)-${RUN_LABEL}}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/evals/${RUN_STAMP}}"

if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT=/path/to/model.checkpoint or pass it as the first argument." >&2
  exit 1
fi

IFS=',' read -r -a gpus <<<"${GPU_LIST}"
if [ "${#gpus[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty." >&2
  exit 1
fi

if [ -z "${MAX_PARALLEL}" ]; then
  MAX_PARALLEL="${GAMES_PER_BATCH}"
  if [ "${MAX_PARALLEL}" -gt "${#gpus[@]}" ]; then
    MAX_PARALLEL="${#gpus[@]}"
  fi
fi

mkdir -p "${OUT_DIR}"

echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUT_DIR}"
echo "Games: ${GAMES}"
echo "GPU_LIST: ${GPU_LIST}"
echo "MAX_PARALLEL: ${MAX_PARALLEL}"
echo "Run stamp: ${RUN_STAMP}"

pids=()
job_dirs=()
job_names=()
job_statuses=()
job_start_epochs=()
job_end_epochs=()
job_exit_codes=()
job_exit_files=()
finished_indices=()
completed_count=0
failed_count=0
total_completed_seconds=0
last_finished_idx=-1
script_start_epoch="$(date +%s)"
progress_fd_open=0
progress_rows=0
progress_done_stack_lines="${EVAL_DONE_STACK_LINES:-3}"
if ! [[ "${progress_done_stack_lines}" =~ ^[0-9]+$ ]]; then
  progress_done_stack_lines=3
fi
progress_height=$((progress_done_stack_lines + 3))

terminal_colors_allowed() {
  local value="${EVAL_TERMINAL_COLORS:-1}"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  [ "${value}" != "0" ] && [ "${value}" != "false" ] && [ "${value}" != "no" ] && [ "${value}" != "off" ]
}

print_status_line() {
  local color="$1"
  shift
  local text="$*"
  if terminal_colors_allowed; then
    printf '\033[%sm%s\033[0m\n' "${color}" "${text}"
  else
    printf '%s\n' "${text}"
  fi
}

cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -TERM "${pid}" >/dev/null 2>&1 || true
    fi
  done
}

terminal_progress_allowed() {
  local value="${EVAL_TERMINAL_PROGRESS_BAR:-1}"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  [ "${value}" != "0" ] && [ "${value}" != "false" ] && [ "${value}" != "no" ] && [ "${value}" != "off" ]
}

init_progress_bar() {
  if [ "${progress_fd_open}" = "1" ]; then
    return 0
  fi
  if ! terminal_progress_allowed; then
    return 1
  fi
  if ! exec 9>/dev/tty 2>/dev/null; then
    return 1
  fi
  progress_fd_open=1
  resize_progress_bar >/dev/null
  return 0
}

resize_progress_bar() {
  if [ "${progress_fd_open}" != "1" ]; then
    return 1
  fi
  local rows columns
  read -r rows columns < <(stty size </dev/tty 2>/dev/null || printf '24 80\n')
  rows="${rows:-24}"
  columns="${columns:-80}"
  if [ "${rows}" -lt 5 ]; then
    rows=5
  fi
  if [ "${columns}" -lt 40 ]; then
    columns=40
  fi
  if [ "${rows}" != "${progress_rows}" ]; then
    printf '\0337\033[1;%dr' "$((rows - progress_height))" >&9
    local line
    for ((line = rows - progress_height + 1; line <= rows; line++)); do
      printf '\033[%d;1H\033[2K' "${line}" >&9
    done
    printf '\0338' >&9
    progress_rows="${rows}"
  fi
  printf '%s %s\n' "${rows}" "${columns}"
}

clear_progress_bar() {
  if [ "${progress_fd_open}" != "1" ]; then
    return 0
  fi
  local rows columns line
  read -r rows columns < <(stty size </dev/tty 2>/dev/null || printf '%s 80\n' "${progress_rows:-24}")
  rows="${rows:-${progress_rows:-24}}"
  printf '\0337\033[r' >&9
  for ((line = rows - progress_height + 1; line <= rows; line++)); do
    if [ "${line}" -gt 0 ]; then
      printf '\033[%d;1H\033[2K' "${line}" >&9
    fi
  done
  printf '\0338' >&9
  exec 9>&-
  progress_fd_open=0
  progress_rows=0
}

format_duration() {
  local seconds="${1:-0}"
  if [ "${seconds}" -lt 0 ]; then
    seconds=0
  fi
  printf '%02d:%02d:%02d' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

running_count() {
  local count=0 idx
  for ((idx = 0; idx < ${#job_statuses[@]}; idx++)); do
    if [ "${job_statuses[idx]}" = "running" ]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

mark_job_finished() {
  local idx="$1"
  local exit_code="$2"
  local end_epoch="${3:-$(date +%s)}"
  local elapsed=$((end_epoch - job_start_epochs[idx]))

  job_end_epochs[idx]="${end_epoch}"
  job_exit_codes[idx]="${exit_code}"
  last_finished_idx="${idx}"
  finished_indices+=("${idx}")
  completed_count=$((completed_count + 1))
  total_completed_seconds=$((total_completed_seconds + elapsed))
  if [ "${exit_code}" = "0" ]; then
    job_statuses[idx]="done"
    print_status_line "42;30" "Finish ${job_names[idx]}: status=done elapsed=$(format_duration "${elapsed}")"
  else
    job_statuses[idx]="failed"
    failed_count=$((failed_count + 1))
    print_status_line "41;97" "Finish ${job_names[idx]}: status=failed exit=${exit_code} elapsed=$(format_duration "${elapsed}")"
  fi
}

reap_jobs() {
  local idx pid exit_file exit_code now
  now="$(date +%s)"
  for ((idx = 0; idx < ${#pids[@]}; idx++)); do
    if [ "${job_statuses[idx]}" != "running" ]; then
      continue
    fi
    pid="${pids[idx]}"
    exit_file="${job_exit_files[idx]}"
    if [ -f "${exit_file}" ]; then
      exit_code="$(tr -d '[:space:]' <"${exit_file}")"
      exit_code="${exit_code:-1}"
      wait "${pid}" >/dev/null 2>&1 || true
      mark_job_finished "${idx}" "${exit_code}" "${now}"
    elif ! kill -0 "${pid}" >/dev/null 2>&1; then
      if wait "${pid}" >/dev/null 2>&1; then
        exit_code=0
      else
        exit_code=$?
      fi
      mark_job_finished "${idx}" "${exit_code}" "${now}"
    fi
  done
}

append_job_segment() {
  local line_var="$1"
  local plain_len_var="$2"
  local plain="$3"
  local colored="$4"
  local columns="$5"
  local current_len="${!plain_len_var}"
  if [ $((current_len + ${#plain})) -gt "${columns}" ]; then
    return 1
  fi
  printf -v "${line_var}" '%s%s' "${!line_var}" "${colored}"
  printf -v "${plain_len_var}" '%s' "$((current_len + ${#plain}))"
  return 0
}

job_turn_progress() {
  local idx="$1"
  local status="${job_statuses[idx]}"
  if [ "${status}" = "done" ]; then
    printf '%s %s\n' "${MAX_TURNS}" 100
    return
  fi
  if [ "${status}" = "failed" ]; then
    printf '0 0\n'
    return
  fi

  local log_path="${job_dirs[idx]}/eval.log"
  local turn=0
  if [ -f "${log_path}" ]; then
    turn="$(
      tail -n 300 "${log_path}" \
        | sed -n 's/.*\[turn \([0-9][0-9]*\) step.*/\1/p' \
        | tail -n 1
    )"
    turn="${turn:-0}"
  fi
  if [ "${turn}" -lt 0 ]; then
    turn=0
  elif [ "${turn}" -gt "${MAX_TURNS}" ]; then
    turn="${MAX_TURNS}"
  fi
  local percent=0
  if [ "${MAX_TURNS}" -gt 0 ]; then
    percent=$((turn * 100 / MAX_TURNS))
  fi
  printf '%s %s\n' "${turn}" "${percent}"
}

job_segment() {
  local idx="$1"
  local now="$2"
  local status="${job_statuses[idx]}"
  local max_width="${3:-0}"
  local elapsed
  if [ "${status}" = "running" ]; then
    elapsed=$((now - job_start_epochs[idx]))
  else
    elapsed=$((job_end_epochs[idx] - job_start_epochs[idx]))
  fi

  local label color turn percent progress
  case "${status}" in
    running)
      label="RUN"
      color='46;30'
      ;;
    done)
      label="DONE"
      color='42;30'
      ;;
    failed)
      label="FAIL"
      color='41;97'
      ;;
    *)
      label="WAIT"
      color='43;30'
      ;;
  esac
  read -r turn percent < <(job_turn_progress "${idx}")
  progress="T${turn}/${MAX_TURNS} ${percent}%"
  local plain=" ${job_names[idx]} ${label} ${progress} $(format_duration "${elapsed}") "
  if [ "${max_width}" -gt 0 ]; then
    if [ "${#plain}" -gt "${max_width}" ]; then
      plain="${plain:0:max_width}"
    else
      plain="$(printf '%-*s' "${max_width}" "${plain}")"
    fi
  fi

  local fill_len=0
  if [ "${status}" = "done" ]; then
    fill_len="${#plain}"
  elif [ "${status}" = "failed" ]; then
    fill_len="${#plain}"
  elif [ "${percent}" -gt 0 ]; then
    fill_len=$((${#plain} * percent / 100))
    if [ "${fill_len}" -eq 0 ]; then
      fill_len=1
    fi
  fi
  local filled_text="${plain:0:fill_len}"
  local empty_text="${plain:fill_len}"
  printf '%s\t\033[%sm%s\033[0m%s' "${plain}" "${color}" "${filled_text}" "${empty_text}"
}

done_stack_line() {
  local stack_idx="$1"
  local now="$2"
  local columns="$3"
  local history_idx idx segment
  history_idx=$((${#finished_indices[@]} - 1 - stack_idx))
  if [ "${history_idx}" -lt 0 ]; then
    printf '%-*s' "${columns}" ""
    return
  fi
  idx="${finished_indices[history_idx]}"
  segment="$(job_segment "${idx}" "${now}" "${columns}")"
  printf '%s' "${segment#*	}"
}

overall_progress_percent() {
  local idx turn percent total=0
  if [ "${GAMES}" -le 0 ]; then
    printf '0\n'
    return
  fi
  for ((idx = 0; idx < ${#job_statuses[@]}; idx++)); do
    if [ "${job_statuses[idx]}" = "done" ] || [ "${job_statuses[idx]}" = "failed" ]; then
      total=$((total + 100))
    elif [ "${job_statuses[idx]}" = "running" ]; then
      read -r turn percent < <(job_turn_progress "${idx}")
      total=$((total + percent))
    fi
  done
  printf '%s\n' "$((total / GAMES))"
}

write_progress_bar() {
  init_progress_bar || return 0
  local size rows columns
  size="$(resize_progress_bar)" || return 0
  rows="${size%% *}"
  columns="${size##* }"

  local now elapsed percent ratio filled remaining eta_text finish_text running avg eta_seconds
  now="$(date +%s)"
  elapsed=$((now - script_start_epoch))
  running="$(running_count)"
  percent="$(overall_progress_percent)"
  filled=$((percent * columns / 100))
  if [ "${percent}" -gt 0 ] && [ "${filled}" -eq 0 ]; then
    filled=1
  fi

  remaining=$((GAMES - completed_count))
  eta_text="ETA --:--:--"
  finish_text="finish --:--:--"
  if [ "${completed_count}" -gt 0 ]; then
    avg=$((total_completed_seconds / completed_count))
    eta_seconds=$(((avg * remaining + MAX_PARALLEL - 1) / MAX_PARALLEL))
    eta_text="ETA $(format_duration "${eta_seconds}")"
    finish_text="finish $(date -d "@$((now + eta_seconds))" '+%H:%M:%S')"
  fi

  local text line filled_text empty_text fill_color
  text=" Eval ${completed_count}/${GAMES} ${percent}% | running ${running}/${MAX_PARALLEL} | elapsed $(format_duration "${elapsed}") | ${eta_text} | ${finish_text}"
  if [ "${failed_count}" -gt 0 ]; then
    fill_color='41;97'
  elif [ "${completed_count}" -eq "${GAMES}" ]; then
    fill_color='42;30'
  else
    fill_color='46;30'
  fi
  if [ "${#text}" -gt "${columns}" ]; then
    text="${text:0:columns}"
  fi
  text="$(printf '%-*s' "${columns}" "${text}")"
  filled_text="${text:0:filled}"
  empty_text="${text:filled}"
  line="$(printf '\033[%sm%s\033[0m%s' "${fill_color}" "${filled_text}" "${empty_text}")"

  local jobs_line="" segment idx running_slots slot_width remainder assigned_width
  running_slots="${running}"
  if [ "${running_slots}" -gt 0 ]; then
    slot_width=$((columns / running_slots))
    remainder=$((columns % running_slots))
  else
    slot_width="${columns}"
    remainder=0
  fi
  for ((idx = 0; idx < ${#job_statuses[@]}; idx++)); do
    if [ "${job_statuses[idx]}" = "running" ]; then
      assigned_width="${slot_width}"
      if [ "${remainder}" -gt 0 ]; then
        assigned_width=$((assigned_width + 1))
        remainder=$((remainder - 1))
      fi
      segment="$(job_segment "${idx}" "${now}" "${assigned_width}")"
      jobs_line="${jobs_line}${segment#*	}"
    fi
  done
  if [ -z "${jobs_line}" ]; then
    jobs_line="$(printf '%-*s' "${columns}" " waiting for next game")"
  fi

  local avg_text last_text output_text detail_line
  avg_text="avg --:--:--"
  if [ "${completed_count}" -gt 0 ]; then
    avg_text="avg $(format_duration "$((total_completed_seconds / completed_count))")/game"
  fi
  last_text="last none"
  if [ "${last_finished_idx}" -ge 0 ]; then
    idx="${last_finished_idx}"
    last_text="last ${job_names[idx]} ${job_statuses[idx]} $(format_duration "$((job_end_epochs[idx] - job_start_epochs[idx]))")"
  fi
  output_text="out ${OUT_DIR}"
  detail_line=" ${last_text} | ${avg_text} | failed ${failed_count} | ${output_text}"
  if [ "${#detail_line}" -gt "${columns}" ]; then
    detail_line="${detail_line:0:columns}"
  fi
  detail_line="$(printf '%-*s' "${columns}" "${detail_line}")"

  local first_progress_row done_row done_line
  first_progress_row=$((rows - progress_height + 1))
  printf '\0337\033[%d;1H\033[2K%s' "${first_progress_row}" "${jobs_line}" >&9
  for ((done_row = 0; done_row < progress_done_stack_lines; done_row++)); do
    done_line="$(done_stack_line "${done_row}" "${now}" "${columns}")"
    printf '\033[%d;1H\033[2K%s' "$((first_progress_row + 1 + done_row))" "${done_line}" >&9
  done
  printf '\033[%d;1H\033[2K%s\033[%d;1H\033[2K%s\0338' \
    "$((rows - 1))" "${line}" \
    "${rows}" "${detail_line}" >&9
}

trap 'cleanup; clear_progress_bar; exit 130' INT TERM
trap 'clear_progress_bar' EXIT

wait_for_slot() {
  while [ "$(running_count)" -ge "${MAX_PARALLEL}" ]; do
    reap_jobs
    write_progress_bar
    if [ "$(running_count)" -ge "${MAX_PARALLEL}" ]; then
      sleep 1
    fi
  done
}

for ((idx = 0; idx < GAMES; idx++)); do
  wait_for_slot

  gpu="${gpus[$((idx % ${#gpus[@]}))]}"
  server_port=$((BASE_SERVER_PORT + idx * PORT_STRIDE))
  lua_port=$((BASE_LUA_PORT + idx * PORT_STRIDE))
  display=":$((BASE_DISPLAY + idx * DISPLAY_STRIDE))"
  observer_display=":$((BASE_DISPLAY + idx * DISPLAY_STRIDE + 1))"
  job_name="$(printf 'game-%02d' "$((idx + 1))")"
  job_dir="${OUT_DIR}/${job_name}"
  mkdir -p "${job_dir}"
  job_dirs+=("${job_dir}")
  job_names+=("${job_name}")
  job_statuses+=("running")
  job_start_epochs+=("$(date +%s)")
  job_end_epochs+=("0")
  job_exit_codes+=("")
  job_exit_files+=("${job_dir}/.exit_status")
  rm -f "${job_dir}/.exit_status"

  print_status_line "44;97" "Start ${job_name}: gpu=${gpu} server=${server_port} lua=${lua_port} display=${display}/${observer_display}"
  (
    set +e
    echo "GPU: ${gpu}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export MAX_TURNS
    export RECORD_FPS
    export DISPLAY_SIZE
    export CLIENT_RESOLUTION
    export SERVER_PORT="${server_port}"
    export LUA_PORT="${lua_port}"
    export DISPLAY_NUM="${display}"
    export OBSERVER_DISPLAY_NUM="${observer_display}"
    export RECORD_DIR="${job_dir}"
    export EVAL_LOG="${job_dir}/eval.log"
    export JSON="${JSON:-1}"
    export JSON_OUT="${job_dir}/remote_play.jsonl"
    export SCORE_LOG="${job_dir}/scores.jsonl"
    export SCORE_LOG_INTERVAL="${SCORE_LOG_INTERVAL:-1}"
    export TURN_SCORE_CSV="${job_dir}/turn_scores.csv"
    export RUN_STAMP="${RUN_STAMP}-${job_name}"
    export EVAL_GAME_NAME="${job_name}"
    export EVAL_GAME_INDEX="$((idx + 1))"
    export EVAL_GPU="${gpu}"
    export FREECIV_TAKE_RETRIES="${FREECIV_TAKE_RETRIES:-60}"
    export FREECIV_TAKE_WAIT="${FREECIV_TAKE_WAIT:-1}"
    "${SCRIPT_DIR}/eval_record_dual_view.sh" "${CHECKPOINT}" "$@"
    rc="$?"
    printf '%s\n' "${rc}" >"${job_dir}/.exit_status"
    exit "${rc}"
  ) >"${job_dir}/runner.log" 2>&1 &
  pids+=("$!")
  write_progress_bar
done

status=0
while [ "${completed_count}" -lt "${GAMES}" ]; do
  reap_jobs
  write_progress_bar
  if [ "${completed_count}" -lt "${GAMES}" ]; then
    sleep 1
  fi
done
write_progress_bar
clear_progress_bar

if [ "${failed_count}" -gt 0 ]; then
  status=1
fi

echo "Done: ${OUT_DIR}"
for job_dir in "${job_dirs[@]}"; do
  if [ -f "${job_dir}/eval-agent.mp4" ] || [ -f "${job_dir}/eval-global.mp4" ]; then
    echo "${job_dir}"
  else
    echo "FAILED_OR_NO_VIDEO ${job_dir}"
  fi
done

exit "${status}"
