#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

if ! email_notification_enabled; then
  echo "Set NOTIFY_EMAIL_TO to the destination address." >&2
  exit 2
fi

send_run_email_notification "email_test" 0 0 ""
