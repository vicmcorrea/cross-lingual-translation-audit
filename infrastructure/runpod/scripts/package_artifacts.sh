#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || {
  echo "Usage: package_artifacts.sh <run-directory> <age-recipient> <output.age>" >&2
  exit 64
}
run_dir="$1"
recipient="$2"
output="$3"
[[ "${run_dir}" == /workspace/runs/* && -f "${run_dir}/run_seal.json" ]] || {
  echo "Run directory must be sealed and below /workspace/runs" >&2
  exit 66
}
[[ "${output}" == /workspace/output-encrypted/stages/*.age && ! -e "${output}" ]] || {
  echo "Output must be a new .age file below the completed-stage namespace" >&2
  exit 73
}
mkdir -p /workspace/output-encrypted/stages
"$(dirname "${BASH_SOURCE[0]}")/verify_run_seal.sh" "${run_dir}"
tar -C "$(dirname "${run_dir}")" -cf - "$(basename "${run_dir}")" \
  | age --recipient "${recipient}" --output "${output}"
(
  cd "$(dirname "${output}")"
  shasum -a 256 "$(basename "${output}")" > "$(basename "${output}").sha256"
)
chmod 0400 "${output}" "${output}.sha256"
echo "Created an encrypted, checksummed run artifact bundle."
