#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

default_build_dir() {
  local candidates=(
    "${ROOT_DIR}/freeciv_build_v3_2_uv"
    "${ROOT_DIR}/../freeciv_build_v3_2_uv"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -x "${candidate}/run.sh" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${ROOT_DIR}/../freeciv_build_v3_2_uv"
}

default_scenario_path() {
  local candidates=(
    "${ROOT_DIR}/freeciv/data/scenarios/minimal_v4.sav"
    "${ROOT_DIR}/freeciv/scenarios/minimal_v4.sav"
    "${HOME}/.freeciv/scenarios/minimal_v4.sav"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${ROOT_DIR}/freeciv/data/scenarios/minimal_v4.sav"
}

latest_checkpoint() {
  {
    find "${ROOT_DIR}/results/checkpoints" -type f -name 'model.checkpoint' -printf '%T@ %p\n' 2>/dev/null
    find "${ROOT_DIR}/results/freeciv_remote" -type f -name 'model.checkpoint' -printf '%T@ %p\n' 2>/dev/null
  } | sort -nr | awk 'NR == 1 { $1 = ""; sub(/^ /, ""); print; exit }'
}

server_rc_has_start() {
  local source_rc="$1"
  [ -f "${source_rc}" ] && grep -Eq '^[[:space:]]*start([[:space:]]|$)' "${source_rc}"
}

server_rc_rulesetdir() {
  local source_rc="$1"
  [ -f "${source_rc}" ] || return 0
  awk '
    /^[[:space:]]*rulesetdir[[:space:]]+/ {
      value = $0
      sub(/^[[:space:]]*rulesetdir[[:space:]]+/, "", value)
      sub(/[[:space:]]*(;.*)?$/, "", value)
      gsub(/^"|"$/, "", value)
      print value
      exit
    }
  ' "${source_rc}"
}

server_rc_set_value() {
  local source_rc="$1"
  local option="$2"
  [ -f "${source_rc}" ] || return 0
  awk -v option="${option}" '
    $1 == "set" && $2 == option {
      value = $0
      sub(/^[[:space:]]*set[[:space:]]+[^[:space:]]+[[:space:]]+/, "", value)
      sub(/[[:space:]]*(;.*)?$/, "", value)
      gsub(/^"|"$/, "", value)
      print value
      exit
    }
  ' "${source_rc}"
}

prepare_server_rc_template() {
  local source_rc="$1"
  local output_dir="$2"
  local start_port="$3"
  local port_stride="$4"
  local count="$5"
  local run_id="$6"
  local score_prefix="${FREECIV_SCOREFILE_PREFIX:-freeciv-score}"

  mkdir -p "${output_dir}"
  local idx port target scorefile
  for ((idx = 0; idx < count; idx++)); do
    port=$((start_port + idx * port_stride))
    target="${output_dir}/start-${port}.serv"
    scorefile="${score_prefix}-${run_id}-${port}.log"
    local tmp
    tmp="${target}.tmp"
    if grep -Eq '^[[:space:]]*set[[:space:]]+scorefile[[:space:]]+"' "${source_rc}"; then
      sed -E \
        "s|^([[:space:]]*set[[:space:]]+scorefile[[:space:]]+\").*(\".*)$|\\1${scorefile}\\2|" \
        "${source_rc}" >"${tmp}"
    else
      awk -v scorefile="${scorefile}" '
        /^[[:space:]]*start([[:space:]]|$)/ && !inserted {
          printf "set scorefile \"%s\"\n", scorefile
          inserted = 1
        }
        { print }
        END {
          if (!inserted) {
            printf "set scorefile \"%s\"\n", scorefile
          }
        }
      ' "${source_rc}" >"${tmp}"
    fi
    awk '
      /^[[:space:]]*set[[:space:]]+wrap([[:space:]]|$)/ { next }
      /^[[:space:]]*start([[:space:]]|$)/ && !inserted {
        print "set wrap \"\""
        inserted = 1
      }
      { print }
      END {
        if (!inserted) {
          print "set wrap \"\""
        }
      }
    ' "${tmp}" >"${target}"
    rm -f "${tmp}"
    if [ -n "${FREECIV_SEED:-}" ]; then
      tmp="${target}.tmp"
      awk -v seed="${FREECIV_SEED}" '
        /^[[:space:]]*set[[:space:]]+mapseed([[:space:]]|$)/ {
          print "set mapseed " seed
          mapseed_inserted = 1
          next
        }
        /^[[:space:]]*set[[:space:]]+gameseed([[:space:]]|$)/ {
          print "set gameseed " seed
          gameseed_inserted = 1
          next
        }
        /^[[:space:]]*start([[:space:]]|$)/ {
          if (!mapseed_inserted) {
            print "set mapseed " seed
            mapseed_inserted = 1
          }
          if (!gameseed_inserted) {
            print "set gameseed " seed
            gameseed_inserted = 1
          }
        }
        { print }
        END {
          if (!mapseed_inserted) {
            print "set mapseed " seed
          }
          if (!gameseed_inserted) {
            print "set gameseed " seed
          }
        }
      ' "${target}" >"${tmp}"
      mv "${tmp}" "${target}"
    fi
    if [ -n "${FREECIV_AIFILL:-}" ]; then
      tmp="${target}.tmp"
      awk -v aifill="${FREECIV_AIFILL}" '
        /^[[:space:]]*set[[:space:]]+aifill([[:space:]]|$)/ {
          print "set aifill " aifill
          inserted = 1
          next
        }
        /^[[:space:]]*start([[:space:]]|$)/ && !inserted {
          print "set aifill " aifill
          inserted = 1
        }
        { print }
        END {
          if (!inserted) {
            print "set aifill " aifill
          }
        }
      ' "${target}" >"${tmp}"
      mv "${tmp}" "${target}"
    fi
  done

  printf '%s\n' "${output_dir}/start-{server_port}.serv"
}

init_python_env() {
  cd "${ROOT_DIR}"
  source .venv/bin/activate
  export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-${RAY_MEMORY_USAGE_THRESHOLD:-0.99}}"
  export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
}

count_csv_items() {
  local csv="$1"
  local old_ifs="${IFS}"
  local -a items=()
  IFS=',' read -r -a items <<<"${csv}"
  IFS="${old_ifs}"
  printf '%s\n' "${#items[@]}"
}

first_csv_item() {
  local csv="$1"
  local old_ifs="${IFS}"
  local -a items=()
  IFS=',' read -r -a items <<<"${csv}"
  IFS="${old_ifs}"
  printf '%s\n' "${items[0]:-}"
}

tail_csv_items() {
  local csv="$1"
  local old_ifs="${IFS}"
  local -a items=()
  IFS=',' read -r -a items <<<"${csv}"
  IFS="${old_ifs}"
  if [ "${#items[@]}" -le 1 ]; then
    printf '%s\n' "${items[0]:-}"
    return
  fi
  local joined="" idx item
  for ((idx = 1; idx < ${#items[@]}; idx++)); do
    item="${items[idx]}"
    if [ -z "${joined}" ]; then
      joined="${item}"
    else
      joined="${joined},${item}"
    fi
  done
  printf '%s\n' "${joined}"
}

init_train_runtime() {
  local run_name="${1:-train}"
  local gpu_list="${TRAIN_GPU_LIST:-${GPU_LIST:-}}"

  if [ -n "${gpu_list}" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${gpu_list}}"
    export MUZERO_TRAIN_GPU_ID="${MUZERO_TRAIN_GPU_ID:-$(first_csv_item "${gpu_list}")}"
    export MUZERO_SELFPLAY_GPU_IDS="${MUZERO_SELFPLAY_GPU_IDS:-$(tail_csv_items "${gpu_list}")}"
    if [ -z "${MUZERO_MAX_NUM_GPUS:-}" ]; then
      MUZERO_MAX_NUM_GPUS="$(count_csv_items "${CUDA_VISIBLE_DEVICES}")"
      export MUZERO_MAX_NUM_GPUS
    fi
  fi

  if [ "${TRAIN_RESET_RAY:-0}" = "1" ]; then
    if command -v ray >/dev/null 2>&1; then
      ray stop --force >/dev/null 2>&1 || true
    elif [ -x "${ROOT_DIR}/.venv/bin/ray" ]; then
      "${ROOT_DIR}/.venv/bin/ray" stop --force >/dev/null 2>&1 || true
    fi
  fi

  printf '[init] run=%s cuda_visible_devices=%s max_num_gpus=%s reset_ray=%s\n' \
    "${run_name}" \
    "${CUDA_VISIBLE_DEVICES:-<unset>}" \
    "${MUZERO_MAX_NUM_GPUS:-<unset>}" \
    "${TRAIN_RESET_RAY:-0}"
  printf '[init] role_gpus train=%s selfplay=%s reanalyse=%s\n' \
    "${MUZERO_TRAIN_GPU_ID:-<auto>}" \
    "${MUZERO_SELFPLAY_GPU_IDS:-<auto>}" \
    "${MUZERO_REANALYSE_GPU_ID:-<auto>}"
}

format_elapsed() {
  local elapsed="$1"
  local hours minutes seconds
  hours=$((elapsed / 3600))
  minutes=$(((elapsed % 3600) / 60))
  seconds=$((elapsed % 60))
  printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}

email_notification_enabled() {
  [ -n "${NOTIFY_EMAIL_TO:-}" ]
}

send_run_email_notification() {
  local run_name="$1"
  local status="$2"
  local elapsed="$3"
  local log_path="${4:-}"
  local recipient="${NOTIFY_EMAIL_TO:-}"

  [ -n "${recipient}" ] || return 0
  case "${recipient}" in
    *$'\n'*|*$'\r'*)
      echo "Email notification skipped: NOTIFY_EMAIL_TO contains a newline." >&2
      return 1
      ;;
  esac

  local outcome outcome_lower host subject_prefix subject result_path checkpoint
  if [ "${status}" -eq 0 ]; then
    case "${NOTIFY_EMAIL_ON_SUCCESS:-1}" in
      0|false|False|FALSE|no|No|NO|off|Off|OFF) return 0 ;;
    esac
    outcome="SUCCESS"
    outcome_lower="succeeded"
  else
    case "${NOTIFY_EMAIL_ON_FAILURE:-1}" in
      0|false|False|FALSE|no|No|NO|off|Off|OFF) return 0 ;;
    esac
    outcome="FAILED"
    outcome_lower="failed"
  fi

  if ! command -v msmtp >/dev/null 2>&1; then
    echo "Email notification skipped: msmtp not found." >&2
    return 1
  fi

  host="$(hostname -f 2>/dev/null || hostname 2>/dev/null || printf 'unknown')"
  subject_prefix="${NOTIFY_EMAIL_SUBJECT_PREFIX:-[freeciv-muzero]}"
  subject="${subject_prefix} ${outcome}: ${run_name} on ${host}"
  result_path="${MUZERO_RESULTS_PATH:-<unset>}"
  checkpoint="<unset>"
  if [ -n "${MUZERO_RESULTS_PATH:-}" ] && [ -f "${MUZERO_RESULTS_PATH}/model.checkpoint" ]; then
    checkpoint="${MUZERO_RESULTS_PATH}/model.checkpoint"
  fi

  if {
    printf 'To: %s\n' "${recipient}"
    if [ -n "${NOTIFY_EMAIL_FROM:-}" ]; then
      printf 'From: %s\n' "${NOTIFY_EMAIL_FROM}"
    fi
    printf 'Subject: %s\n' "${subject}"
    printf 'Date: %s\n' "$(date -R)"
    printf 'Content-Type: text/plain; charset=UTF-8\n'
    printf '\n'
    printf 'Run %s.\n\n' "${outcome_lower}"
    printf 'Run: %s\n' "${run_name}"
    printf 'Host: %s\n' "${host}"
    printf 'Exit status: %s\n' "${status}"
    printf 'Elapsed: %s (%ss)\n' "$(format_elapsed "${elapsed}")" "${elapsed}"
    printf 'Results: %s\n' "${result_path}"
    printf 'Checkpoint: %s\n' "${checkpoint}"
    printf 'Log: %s\n' "${log_path:-<disabled>}"
  } | msmtp -t; then
    echo "Email notification sent: ${recipient}" >&2
    return 0
  fi

  echo "Email notification failed: ${recipient}" >&2
  return 1
}

drive_sync_should_list_file() {
  local file="$1"
  case "${file}" in
    */belief_tensorboard/*)
      return 1
      ;;
    *.tfevents.*)
      [ "${GOOGLE_DRIVE_RESULTS_INCLUDE_TENSORBOARD:-1}" = "1" ]
      return
      ;;
    */heatmaps/*)
      case "${file}" in
        */heatmaps/videos/*.mp4) return 0 ;;
        *) return 1 ;;
      esac
      ;;
  esac
  return 0
}

print_drive_sync_plan() {
  local source="$1"
  local tmp count total_bytes total_human shown line
  tmp="$(mktemp)"
  if [ -f "${source}" ]; then
    if drive_sync_should_list_file "${source}"; then
      printf '%s\n' "${source}" >"${tmp}"
    fi
  else
    while IFS= read -r -d '' line; do
      if drive_sync_should_list_file "${line}"; then
        printf '%s\n' "${line}" >>"${tmp}"
      fi
    done < <(find "${source}" -type f -print0)
  fi

  count="$(wc -l <"${tmp}")"
  total_bytes="$(
    while IFS= read -r line; do
      [ -f "${line}" ] && stat -c '%s' "${line}"
    done <"${tmp}" | awk '{sum += $1} END {printf "%.0f", sum}'
  )"
  total_human="$(numfmt --to=iec --suffix=B "${total_bytes}" 2>/dev/null || printf '%s bytes' "${total_bytes}")"
  echo "Drive sync files: ${count}" >&2
  echo "Drive sync size: ${total_human}" >&2
  shown=0
  while IFS= read -r line && [ "${shown}" -lt 50 ]; do
    if [[ "${line}" == "${source}/"* ]]; then
      echo "  ${line#"${source}/"}" >&2
    else
      echo "  ${line}" >&2
    fi
    shown=$((shown + 1))
  done <"${tmp}"
  if [ "${count}" -gt 50 ] 2>/dev/null; then
    echo "  ... $((count - 50)) more" >&2
  fi
  rm -f "${tmp}"
}

sync_results_to_drive() {
  local source="${1:-${ROOT_DIR}/results}"
  [ -n "${GOOGLE_DRIVE_RESULTS:-}" ] || return 0
  [ -e "${source}" ] || return 0
  local destination="${GOOGLE_DRIVE_RESULTS}"
  local source_abs results_abs rel dest_rel source_arg
  source_abs="$(cd "$(dirname "${source}")" && pwd)/$(basename "${source}")"
  results_abs="$(cd "${ROOT_DIR}/results" 2>/dev/null && pwd || true)"
  if [ -n "${results_abs}" ] && [[ "${source_abs}" == "${results_abs}/"* ]]; then
    rel="${source_abs#"${results_abs}/"}"
    if [ -d "${source}" ]; then
      dest_rel="${rel}"
    else
      dest_rel="$(dirname "${rel}")"
    fi
    if [ "${dest_rel}" != "." ]; then
      destination="${destination%/}/${dest_rel}"
    fi
  fi

  local interval="${GOOGLE_DRIVE_RESULTS_INTERVAL:-300}"
  local marker="/tmp/freeciv-muzero-drive-sync-$(printf '%s' "${source}" | tr '/ ' '__').stamp"
  if [ "${interval}" -gt 0 ] 2>/dev/null && [ -e "${marker}" ]; then
    local now last
    now="$(date +%s)"
    last="$(stat -c %Y "${marker}" 2>/dev/null || printf '0')"
    if [ $((now - last)) -lt "${interval}" ]; then
      return 0
    fi
  fi
  touch "${marker}" 2>/dev/null || true

  if [[ "${destination}" == *:* ]]; then
    if ! command -v rclone >/dev/null 2>&1; then
      echo "Google Drive sync skipped: rclone not found." >&2
      return 0
    fi
    echo "Google Drive sync: ${source} -> ${destination}" >&2
    if [ "${GOOGLE_DRIVE_RESULTS_VERBOSE:-0}" = "1" ]; then
      print_drive_sync_plan "${source}"
    fi
    rclone_args=(
      copy "${source}" "${destination}"
      --filter '+ heatmaps/videos/*.mp4'
      --filter '- heatmaps/**'
      --filter '- belief_tensorboard/**'
    )
    if [ "${GOOGLE_DRIVE_RESULTS_INCLUDE_TENSORBOARD:-1}" != "1" ]; then
      rclone_args+=(--filter '- *.tfevents.*')
    fi
    if [ "${GOOGLE_DRIVE_RESULTS_VERBOSE:-0}" = "1" ]; then
      rclone_args+=(--progress --stats-one-line --log-level INFO)
    fi
    if [ "${GOOGLE_DRIVE_RESULTS_BACKGROUND:-1}" = "1" ]; then
      if [ "${GOOGLE_DRIVE_RESULTS_VERBOSE:-0}" = "1" ]; then
        rclone_args+=(--stats 5s)
        rclone "${rclone_args[@]}" &
      else
        rclone "${rclone_args[@]}" >/dev/null 2>&1 &
      fi
    else
      rclone "${rclone_args[@]}" || true
    fi
  else
    mkdir -p "${destination}"
    echo "Google Drive sync: ${source} -> ${destination}" >&2
    if [ -d "${source}" ]; then
      source_arg="${source}/"
    else
      source_arg="${source}"
    fi
    if command -v rsync >/dev/null 2>&1; then
      if [ "${GOOGLE_DRIVE_RESULTS_BACKGROUND:-1}" = "1" ]; then
        rsync -a "${source_arg}" "${destination}/" >/dev/null 2>&1 &
      else
        rsync -a "${source_arg}" "${destination}/" || true
      fi
    elif [ "${GOOGLE_DRIVE_RESULTS_BACKGROUND:-1}" = "1" ]; then
      cp -a "${source_arg}" "${destination}/" >/dev/null 2>&1 &
    else
      cp -a "${source_arg}" "${destination}/" || true
    fi
  fi
}

run_with_timing_and_log() {
  local run_name="$1"
  shift

  local start_epoch end_epoch elapsed status stamp log_path
  start_epoch="$(date +%s)"
  stamp="$(date +%Y%m%d-%H%M%S)"
  log_path="${RUN_LOG:-${LOG_FILE:-}}"

  if [ "${SAVE_RUN_LOG:-1}" = "1" ]; then
    if [ -z "${log_path}" ]; then
      log_path="${RUN_LOG_DIR:-${ROOT_DIR}/results/logs}/${run_name}-${stamp}-$$.log"
    elif [ "${log_path}" = "0" ] || [ "${log_path}" = "false" ]; then
      log_path=""
    fi
  else
    log_path=""
  fi

  if [ -n "${log_path}" ]; then
    mkdir -p "$(dirname "${log_path}")"
    export MUZERO_TERMINAL_PROGRESS_BAR="${MUZERO_TERMINAL_PROGRESS_BAR:-0}"
    {
      printf 'Run: %s\n' "${run_name}"
      printf 'Started: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
      printf 'MUZERO_MAX_NUM_GPUS: %s\n' "${MUZERO_MAX_NUM_GPUS:-<unset>}"
      printf 'MUZERO_TRAIN_GPU_ID: %s\n' "${MUZERO_TRAIN_GPU_ID:-<auto>}"
      printf 'MUZERO_SELFPLAY_GPU_IDS: %s\n' "${MUZERO_SELFPLAY_GPU_IDS:-<auto>}"
      printf 'MUZERO_REANALYSE_GPU_ID: %s\n' "${MUZERO_REANALYSE_GPU_ID:-<auto>}"
      printf 'Command:'
      printf ' %q' "$@"
      printf '\n'
      printf 'Log: %s\n' "${log_path}"
    } | tee -a "${log_path}"

    set +e
    case "${STREAM_RUN_LOG:-1}" in
      0|false|False|FALSE|no|No|NO|off|Off|OFF)
        "$@" >>"${log_path}" 2>&1
        status="$?"
        ;;
      *)
        "$@" 2>&1 | tee -a "${log_path}"
        status="${PIPESTATUS[0]}"
        ;;
    esac
    set -e
  else
    printf 'Run: %s\n' "${run_name}"
    printf 'Started: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
    printf 'MUZERO_MAX_NUM_GPUS: %s\n' "${MUZERO_MAX_NUM_GPUS:-<unset>}"
    printf 'MUZERO_TRAIN_GPU_ID: %s\n' "${MUZERO_TRAIN_GPU_ID:-<auto>}"
    printf 'MUZERO_SELFPLAY_GPU_IDS: %s\n' "${MUZERO_SELFPLAY_GPU_IDS:-<auto>}"
    printf 'MUZERO_REANALYSE_GPU_ID: %s\n' "${MUZERO_REANALYSE_GPU_ID:-<auto>}"
    set +e
    "$@"
    status="$?"
    set -e
  fi

  end_epoch="$(date +%s)"
  elapsed=$((end_epoch - start_epoch))
  if [ -n "${log_path}" ]; then
    {
      printf '\n'
      printf 'Finished: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'Exit status: %s\n' "${status}"
      printf 'Elapsed: %s (%ss)\n' "$(format_elapsed "${elapsed}")" "${elapsed}"
      printf 'Log: %s\n' "${log_path}"
    } | tee -a "${log_path}"
  else
    printf '\n'
    printf 'Finished: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'Exit status: %s\n' "${status}"
    printf 'Elapsed: %s (%ss)\n' "$(format_elapsed "${elapsed}")" "${elapsed}"
  fi

  send_run_email_notification \
    "${run_name}" "${status}" "${elapsed}" "${log_path}" || true

  return "${status}"
}

cleanup_freeciv_ports() {
  local server_port="${1:-5566}"
  local lua_port="${2:-4451}"
  fuser -k -TERM "${server_port}/tcp" >/dev/null 2>&1 || true
  fuser -k -TERM "${lua_port}/tcp" >/dev/null 2>&1 || true
}

cleanup_freeciv_processes() {
  local server_port="${1:-5566}"
  local display_num="${2:-}"
  pkill -f "freeciv-gtk3.22.*-p ${server_port}" >/dev/null 2>&1 || true
  pkill -f "freeciv-server -p ${server_port}" >/dev/null 2>&1 || true
  if [ -n "${display_num}" ]; then
    pkill -f "Xvfb ${display_num}" >/dev/null 2>&1 || true
  fi
}

cleanup_freeciv_all() {
  local server_port="${1:-5566}"
  local lua_port="${2:-4451}"
  local display_num="${3:-}"
  cleanup_freeciv_ports "${server_port}" "${lua_port}"
  cleanup_freeciv_processes "${server_port}" "${display_num}"
}
