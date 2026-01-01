#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 4 && "$3" == "--" ]] || {
  echo "Usage: run_with_pod_cleanup.sh <pod-id> <new-lifecycle-log.jsonl> -- <command...>" >&2
  exit 64
}
pod_id="$1"
lifecycle_log="$2"
shift 3
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ "${pod_id}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid Pod ID" >&2; exit 64; }
[[ "${lifecycle_log}" == *.jsonl && ! -e "${lifecycle_log}" ]] || {
  echo "Lifecycle log must be a new .jsonl path" >&2
  exit 73
}
command -v jq >/dev/null || { echo "Missing dependency jq" >&2; exit 69; }

umask 077
mkdir -p "$(dirname "${lifecycle_log}")"
touch "${lifecycle_log}"
chmod 0600 "${lifecycle_log}"
workflow_terminal_logged=false

log_event() {
  local event="$1"
  local status="$2"
  jq -nc \
    --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg event "${event}" \
    --arg status "${status}" \
    --arg pod_id "${pod_id}" \
    '{timestamp: $timestamp, event: $event, status: $status, pod_id: $pod_id}' \
    >> "${lifecycle_log}"
}

finalize() {
  local command_status="$?"
  local deletion_status=0
  trap - EXIT INT TERM
  if [[ "${workflow_terminal_logged}" != true ]]; then
    log_event "pod_workflow_finished" "failed"
  fi
  log_event "pod_deletion_started" "running"
  if "${script_dir}/delete_project_pod.sh" "${pod_id}"; then
    log_event "pod_deletion_verified" "completed"
  else
    deletion_status="$?"
    log_event "pod_deletion_verified" "failed"
  fi
  if (( command_status == 0 && deletion_status != 0 )); then
    command_status="${deletion_status}"
  fi
  exit "${command_status}"
}

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
log_event "pod_workflow_started" "running"
"$@"
log_event "pod_workflow_finished" "completed"
workflow_terminal_logged=true
