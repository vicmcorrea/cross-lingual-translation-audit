#!/usr/bin/env bash
set -euo pipefail

target="/dev/shm/translation-audit/plaintext/current"
[[ -d "${target}" ]] || { echo "No plaintext workspace exists"; exit 0; }
find "${target}" -type f -exec chmod u+w {} +
find "${target}" -depth -type d -exec chmod u+w {} +
rm -rf -- "${target}"
echo "Removed the ephemeral plaintext workspace."
