#!/usr/bin/env bash
set -euo pipefail

project_prefix="cross-lingual-translation-audit-"
for executable in jq runpodctl; do
  command -v "${executable}" >/dev/null || { echo "Missing dependency ${executable}" >&2; exit 69; }
done

pods_json="$(runpodctl pod list --all -o json)"
remaining="$(
  jq -c --arg prefix "${project_prefix}" \
    '[.[] | select((.name // "") | startswith($prefix)) | {id, name, status}]' \
    <<<"${pods_json}"
)"
[[ "${remaining}" == "[]" ]] || {
  echo "Project Pods still exist and must be deleted: ${remaining}" >&2
  exit 75
}
echo "Verified that no project Pod remains."
