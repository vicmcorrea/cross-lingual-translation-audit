#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || {
  echo "Usage: collect_artifacts.sh <ssh-host> <remote-encrypted-artifact> <local-directory>" >&2
  exit 64
}
host="$1"
remote_bundle="$2"
local_dir="$3"
[[ "${remote_bundle}" == /workspace/output-encrypted/*.age ]] || {
  echo "Remote artifact must be an .age file below /workspace/output-encrypted" >&2
  exit 64
}
mkdir -p "${local_dir}"
rsync --archive --partial --protect-args \
  "${host}:${remote_bundle}" \
  "${host}:${remote_bundle}.sha256" \
  "${local_dir}/"
(
  cd "${local_dir}"
  shasum -a 256 -c "$(basename "${remote_bundle}").sha256"
)
