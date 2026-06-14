#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

default_build_dir() {
  local candidates=(
    "${ROOT_DIR}/freeciv_build_v3_2_uv"
    "${ROOT_DIR}/freeciv_build_v3_2"
    "${ROOT_DIR}/../freeciv_build_v3_2_uv"
    "${ROOT_DIR}/../freeciv_build_v3_2"
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

run_with_timing_and_log() {
  local run_name="$1"
  shift

  local start_epoch end_epoch elapsed status stamp log_path
  start_epoch="$(date +%s)"
  stamp="$(date +%Y%m%d-%H%M%S)"
  log_path="${RUN_LOG:-${LOG_FILE:-}}"

  if [ "${SAVE_RUN_LOG:-1}" = "1" ]; then
    if [ -z "${log_path}" ]; then
      log_path="${RUN_LOG_DIR:-${ROOT_DIR}/results/logs}/${run_name}-${stamp}.log"
    elif [ "${log_path}" = "0" ] || [ "${log_path}" = "false" ]; then
      log_path=""
    fi
  else
    log_path=""
  fi

  if [ -n "${log_path}" ]; then
    mkdir -p "$(dirname "${log_path}")"
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
    "$@" 2>&1 | tee -a "${log_path}"
    status="${PIPESTATUS[0]}"
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
