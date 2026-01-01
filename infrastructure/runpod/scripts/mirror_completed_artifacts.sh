#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 5 && $# -le 6 ]] || {
  echo "Usage: mirror_completed_artifacts.sh <pod-id> <ssh-host> <ssh-port> <ssh-key> <local-directory> [max-seconds]" >&2
  exit 64
}

pod_id="$1"
ssh_host="$2"
ssh_port="$3"
ssh_key="$4"
local_dir="${5%/}"
max_seconds="${6:-21600}"
poll_seconds="${TRANSLATION_AUDIT_MIRROR_POLL_SECONDS:-5}"
remote_root="/workspace/output-encrypted"
remote_stage_dir="${remote_root}/stages"
remote_terminal="${remote_root}/pipeline_terminal.json"

[[ "${pod_id}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid Pod ID" >&2; exit 64; }
[[ "${ssh_host}" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "Invalid SSH host" >&2; exit 64; }
[[ "${ssh_port}" =~ ^[0-9]+$ ]] || { echo "Invalid SSH port" >&2; exit 64; }
[[ -f "${ssh_key}" ]] || { echo "SSH key is unavailable" >&2; exit 66; }
[[ "${max_seconds}" =~ ^[1-9][0-9]*$ ]] || { echo "Maximum duration must be positive" >&2; exit 64; }
[[ "${poll_seconds}" =~ ^[1-9][0-9]*$ ]] || { echo "Poll interval must be positive" >&2; exit 64; }
command -v jq >/dev/null || { echo "Missing dependency jq" >&2; exit 69; }
command -v rsync >/dev/null || { echo "Missing dependency rsync" >&2; exit 69; }
command -v runpodctl >/dev/null || { echo "Missing dependency runpodctl" >&2; exit 69; }

pod_json="$(runpodctl pod get "${pod_id}" -o json)" || {
  echo "Project Pod is unavailable" >&2
  exit 66
}
pod_name="$(jq -er '.name' <<<"${pod_json}")"
[[ "${pod_name}" == cross-lingual-translation-audit-* ]] || {
  echo "Refusing to mirror from a Pod outside the project naming boundary" >&2
  exit 65
}

umask 077
identity_path="${local_dir}/mirror_identity.json"
if [[ -e "${local_dir}" && ! -f "${identity_path}" ]] && \
  find "${local_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Local mirror directory is not an initialized session" >&2
  exit 73
fi
mkdir -p "${local_dir}" "${local_dir}/stages" "${local_dir}/receipts" "${local_dir}/.incoming"
chmod 0700 "${local_dir}" "${local_dir}/stages" "${local_dir}/receipts" "${local_dir}/.incoming"
if [[ -f "${identity_path}" ]]; then
  jq -e --arg pod_id "${pod_id}" --arg pod_name "${pod_name}" \
    '.pod_id == $pod_id and .pod_name == $pod_name' "${identity_path}" >/dev/null || {
    echo "Local mirror directory belongs to a different Pod lease" >&2
    exit 65
  }
else
  identity_tmp="${identity_path}.tmp.$$"
  jq -nc \
    --arg pod_id "${pod_id}" \
    --arg pod_name "${pod_name}" \
    --arg created_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{pod_id: $pod_id, pod_name: $pod_name, created_at: $created_at}' > "${identity_tmp}"
  mv "${identity_tmp}" "${identity_path}"
  chmod 0400 "${identity_path}"
fi

mirror_log="${local_dir}/mirror.jsonl"
touch "${mirror_log}"
chmod 0600 "${mirror_log}"
ssh_options=(
  -i "${ssh_key}"
  -p "${ssh_port}"
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -o StrictHostKeyChecking=accept-new
)
printf -v rsync_shell 'ssh -i %q -p %q -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new' \
  "${ssh_key}" "${ssh_port}"
remote="root@${ssh_host}"
started_at="$(date +%s)"
mirror_completed=false

log_event() {
  local event="$1"
  local status="$2"
  local bundle_name="${3:-}"
  local digest="${4:-}"
  jq -nc \
    --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg event "${event}" \
    --arg status "${status}" \
    --arg pod_id "${pod_id}" \
    --arg bundle "${bundle_name}" \
    --arg sha256 "${digest}" \
    '{timestamp: $timestamp, event: $event, status: $status, pod_id: $pod_id, bundle: $bundle, sha256: $sha256}' \
    >> "${mirror_log}"
}

mark_mirror_failure() {
  local exit_code="$?"
  trap - EXIT
  if (( exit_code != 0 )) && [[ "${mirror_completed}" != true ]]; then
    ssh "${ssh_options[@]}" "${remote}" \
      'umask 077; mkdir -p /workspace/output-encrypted; : > /workspace/output-encrypted/mirror_failed' \
      >/dev/null 2>&1 || true
  fi
  exit "${exit_code}"
}
trap mark_mirror_failure EXIT

strict_digest() {
  local checksum_path="$1"
  local expected_name="$2"
  local line
  [[ -f "${checksum_path}" && ! -L "${checksum_path}" ]] || return 1
  [[ "$(wc -l < "${checksum_path}" | tr -d ' ')" == "1" ]] || return 1
  line="$(sed -n '1p' "${checksum_path}")"
  [[ "${line:0:64}" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ "${line:64}" == "  ${expected_name}" ]] || return 1
  printf '%s\n' "${line:0:64}"
}

acknowledge_remote() {
  local bundle_name="$1"
  local digest="$2"
  ssh "${ssh_options[@]}" "${remote}" bash -s -- "${bundle_name}" "${digest}" <<'REMOTE_ACK'
set -euo pipefail
bundle_name="$1"
digest="$2"
[[ "${bundle_name}" =~ ^[A-Za-z0-9_.-]+\.age$ && "${digest}" =~ ^[0-9a-f]{64}$ ]] || exit 64
ack_dir="/workspace/output-encrypted/acks"
mkdir -p "${ack_dir}"
ack_path="${ack_dir}/${bundle_name}.ack"
ack_tmp="${ack_path}.tmp.$$"
printf '%s\n' "${digest}" > "${ack_tmp}"
chmod 0400 "${ack_tmp}"
mv "${ack_tmp}" "${ack_path}"
REMOTE_ACK
}

mirror_bundle() {
  local checksum_name="$1"
  local bundle_name="${checksum_name%.sha256}"
  local incoming_dir="${local_dir}/.incoming/${bundle_name}"
  local incoming_bundle="${incoming_dir}/${bundle_name}"
  local incoming_checksum="${incoming_dir}/${checksum_name}"
  local final_bundle="${local_dir}/stages/${bundle_name}"
  local final_checksum="${local_dir}/stages/${checksum_name}"
  local receipt="${local_dir}/receipts/${bundle_name}.receipt.json"
  local digest actual_digest receipt_tmp

  [[ "${checksum_name}" =~ ^[A-Za-z0-9_.-]+\.age\.sha256$ ]] || {
    echo "Remote checksum name is outside the encrypted artifact allowlist" >&2
    return 65
  }
  ssh "${ssh_options[@]}" "${remote}" bash -s -- "${bundle_name}" "${checksum_name}" <<'REMOTE_VALIDATE'
set -euo pipefail
bundle_name="$1"
checksum_name="$2"
root="/workspace/output-encrypted/stages"
[[ "${bundle_name}" =~ ^[A-Za-z0-9_.-]+\.age$ && "${checksum_name}" == "${bundle_name}.sha256" ]] || exit 64
[[ -f "${root}/${bundle_name}" && ! -L "${root}/${bundle_name}" ]]
[[ -f "${root}/${checksum_name}" && ! -L "${root}/${checksum_name}" ]]
REMOTE_VALIDATE

  if [[ -f "${receipt}" ]]; then
    digest="$(jq -er '.sha256 | select(test("^[0-9a-f]{64}$"))' "${receipt}")"
    [[ -f "${final_bundle}" && -f "${final_checksum}" ]] || return 65
    [[ "$(strict_digest "${final_checksum}" "${bundle_name}")" == "${digest}" ]] || return 65
    [[ "$(shasum -a 256 "${final_bundle}" | awk '{print $1}')" == "${digest}" ]] || return 65
    acknowledge_remote "${bundle_name}" "${digest}"
    return 0
  fi

  mkdir -p "${incoming_dir}"
  rsync --archive --no-owner --no-group --partial \
    -e "${rsync_shell}" \
    "${remote}:${remote_stage_dir}/${bundle_name}" \
    "${remote}:${remote_stage_dir}/${checksum_name}" \
    "${incoming_dir}/"
  digest="$(strict_digest "${incoming_checksum}" "${bundle_name}")" || {
    echo "Remote encrypted artifact checksum has an invalid format" >&2
    return 65
  }
  actual_digest="$(shasum -a 256 "${incoming_bundle}" | awk '{print $1}')"
  [[ "${actual_digest}" == "${digest}" ]] || {
    echo "Mirrored encrypted artifact failed checksum verification" >&2
    return 65
  }

  if [[ -e "${final_bundle}" || -e "${final_checksum}" ]]; then
    [[ -f "${final_bundle}" && -f "${final_checksum}" ]] || return 65
    [[ "$(strict_digest "${final_checksum}" "${bundle_name}")" == "${digest}" ]] || return 65
    [[ "$(shasum -a 256 "${final_bundle}" | awk '{print $1}')" == "${digest}" ]] || return 65
  else
    chmod 0400 "${incoming_bundle}" "${incoming_checksum}"
    mv "${incoming_bundle}" "${final_bundle}"
    mv "${incoming_checksum}" "${final_checksum}"
  fi

  receipt_tmp="${receipt}.tmp.$$"
  jq -nc \
    --arg pod_id "${pod_id}" \
    --arg bundle "${bundle_name}" \
    --arg sha256 "${digest}" \
    --arg verified_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{pod_id: $pod_id, namespace: "completed_stage", bundle: $bundle, sha256: $sha256, verified_at: $verified_at}' \
    > "${receipt_tmp}"
  chmod 0400 "${receipt_tmp}"
  mv "${receipt_tmp}" "${receipt}"
  log_event "artifact_verification" "completed" "${bundle_name}" "${digest}"
  acknowledge_remote "${bundle_name}" "${digest}"
}

remote_checksums() {
  ssh "${ssh_options[@]}" "${remote}" \
    "find /workspace/output-encrypted/stages -maxdepth 1 -type f -name '*.age.sha256' -printf '%f\\n' | sort"
}

log_event "artifact_mirror_started" "running"
while true; do
  elapsed=$(( $(date +%s) - started_at ))
  if (( elapsed >= max_seconds )); then
    log_event "artifact_mirror_finished" "timed_out"
    echo "Artifact mirror reached its maximum duration" >&2
    exit 124
  fi

  checksum_listing="$(remote_checksums 2>/dev/null || true)"
  while IFS= read -r checksum_name; do
    [[ -n "${checksum_name}" ]] || continue
    mirror_bundle "${checksum_name}"
  done <<<"${checksum_listing}"

  if ssh "${ssh_options[@]}" "${remote}" \
    '[[ -f /workspace/output-encrypted/pipeline_terminal.json && ! -L /workspace/output-encrypted/pipeline_terminal.json ]]' 2>/dev/null; then
    final_listing="$(remote_checksums)"
    while IFS= read -r checksum_name; do
      [[ -n "${checksum_name}" ]] || continue
      mirror_bundle "${checksum_name}"
    done <<<"${final_listing}"
    rsync --archive --no-owner --no-group \
      -e "${rsync_shell}" "${remote}:${remote_terminal}" "${local_dir}/pipeline_terminal.json"
    jq -e '.status == "completed" or .status == "failed"' \
      "${local_dir}/pipeline_terminal.json" >/dev/null
    chmod 0400 "${local_dir}/pipeline_terminal.json"
    log_event "artifact_mirror_finished" "completed"
    mirror_completed=true
    echo "All committed encrypted stage artifacts were mirrored and acknowledged."
    exit 0
  fi

  if ! runpodctl pod get "${pod_id}" -o json >/dev/null 2>&1; then
    log_event "artifact_mirror_finished" "pod_absent"
    echo "Project Pod ended before the remote terminal inventory was drained" >&2
    exit 66
  fi
  sleep "${poll_seconds}"
done
