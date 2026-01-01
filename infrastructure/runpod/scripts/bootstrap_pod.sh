#!/usr/bin/env bash
set -euo pipefail

umask 077
mkdir -p /workspace/cache /workspace/input-encrypted /workspace/output-encrypted /workspace/runs
[[ "$(findmnt -n -o FSTYPE --target /dev/shm)" == "tmpfs" ]] || {
  echo "The plaintext workspace requires tmpfs" >&2
  exit 78
}
mkdir -p /run/secrets /dev/shm/translation-audit/plaintext
chmod 0700 /dev/shm/translation-audit/plaintext
if [[ -n "${AGE_SECRET_KEY:-}" ]]; then
  printf '%s\n' "${AGE_SECRET_KEY}" > /run/secrets/age-identity
  chmod 0400 /run/secrets/age-identity
  unset AGE_SECRET_KEY
fi
python /usr/local/bin/verify-gpu.py
exec "$@"
