#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || {
  echo "Usage: sync_to_pod.sh <encrypted-bundle> <ssh-host> <remote-directory>" >&2
  exit 64
}
bundle="$1"
host="$2"
remote_dir="$3"
checksum="${bundle}.sha256"
[[ "${bundle}" == *.age && -f "${bundle}" && -f "${checksum}" ]] || {
  echo "Encrypted bundle or checksum does not exist" >&2
  exit 66
}
[[ "${remote_dir}" == /workspace/input-encrypted/* ]] || {
  echo "Remote directory must be below /workspace/input-encrypted" >&2
  exit 64
}
(
  cd "$(dirname "${bundle}")"
  shasum -a 256 -c "$(basename "${checksum}")"
)
rsync --archive --partial --protect-args "${bundle}" "${checksum}" "${host}:${remote_dir}/"
