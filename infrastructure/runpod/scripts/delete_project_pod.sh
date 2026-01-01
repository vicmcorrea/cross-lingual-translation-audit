#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: delete_project_pod.sh <pod-id>" >&2; exit 64; }
pod_id="$1"
project_prefix="cross-lingual-translation-audit-"
[[ "${pod_id}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid Pod ID" >&2; exit 64; }
for executable in jq runpodctl; do
  command -v "${executable}" >/dev/null || { echo "Missing dependency ${executable}" >&2; exit 69; }
done

if ! pod_json="$(runpodctl pod get "${pod_id}" -o json 2>/dev/null)"; then
  echo "Pod ${pod_id} is already absent."
  exit 0
fi
pod_name="$(jq -er '.name // .pod.name' <<<"${pod_json}")"
[[ "${pod_name}" == "${project_prefix}"* ]] || {
  echo "Refusing to delete Pod outside the project naming boundary" >&2
  exit 65
}

runpodctl pod delete "${pod_id}" >/dev/null
for _attempt in {1..30}; do
  if ! runpodctl pod get "${pod_id}" -o json >/dev/null 2>&1; then
    echo "Deleted and verified Pod ${pod_id}."
    exit 0
  fi
  sleep 2
done

echo "Pod deletion could not be verified within 60 seconds" >&2
exit 75
