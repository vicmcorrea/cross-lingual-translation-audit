#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: verify_run_seal.sh <run-directory>" >&2; exit 64; }
run_dir="${1%/}"
seal_path="${run_dir}/run_seal.json"

[[ ( "${run_dir}" == /workspace/runs/* || \
  "${run_dir}" == /dev/shm/translation-audit/imported-runs/* ) \
  && -d "${run_dir}" && -f "${seal_path}" ]] || {
  echo "Run seal verification is outside an approved run root" >&2
  exit 66
}
[[ -z "$(find "${run_dir}" -mindepth 1 ! -type f ! -type d -print -quit)" ]] || {
  echo "Run contains an unsupported filesystem entry" >&2
  exit 65
}
jq -e \
  --arg run_id "$(basename "${run_dir}")" \
  '.run_id == $run_id' \
  "${seal_path}" >/dev/null
jq -e \
  '.files_sha256 | type == "object" and all(to_entries[]; (.key | type == "string") and (.value | test("^[0-9a-f]{64}$")))' \
  "${seal_path}" >/dev/null

sealed_count="$(jq -r '.files_sha256 | length' "${seal_path}")"
actual_count="$(find "${run_dir}" -type f ! -path "${seal_path}" | wc -l | tr -d ' ')"
[[ "${sealed_count}" == "${actual_count}" ]] || {
  echo "Run contents do not match the sealed file count" >&2
  exit 65
}

while IFS=$'\t' read -r relative_path expected_sha; do
  [[ -n "${relative_path}" && -n "${expected_sha}" ]] || {
    echo "Run seal contains an invalid file entry" >&2
    exit 65
  }
  [[ "${relative_path}" != /* && "${relative_path}" != ".." && "${relative_path}" != ../* && "${relative_path}" != */../* && "${relative_path}" != */.. ]] || {
    echo "Run seal contains an unsafe file path" >&2
    exit 65
  }
  artifact_path="${run_dir}/${relative_path}"
  [[ -f "${artifact_path}" && ! -L "${artifact_path}" ]] || {
    echo "Run seal references a missing or non-regular file" >&2
    exit 65
  }
  actual_sha="$(shasum -a 256 "${artifact_path}" | awk '{print $1}')"
  [[ "${actual_sha}" == "${expected_sha}" ]] || {
    echo "Run artifact does not match its scientific seal" >&2
    exit 65
  }
done < <(jq -r '.files_sha256 | to_entries[] | [.key, .value] | @tsv' "${seal_path}")

echo "Verified the complete scientific run seal."
