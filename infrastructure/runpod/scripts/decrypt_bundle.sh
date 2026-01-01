#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: decrypt_bundle.sh <bundle.age>" >&2; exit 64; }
bundle="$1"
identity_file="/run/secrets/age-identity"
destination_root="/dev/shm/translation-audit/plaintext"
destination="${destination_root}/current"
[[ "${bundle}" == /workspace/input-encrypted/*.age && -f "${bundle}" ]] || {
  echo "Bundle must be an existing .age file below /workspace/input-encrypted" >&2
  exit 66
}
[[ -f "${identity_file}" ]] || { echo "Age identity is unavailable" >&2; exit 78; }
[[ ! -e "${destination}" ]] || { echo "Plaintext workspace already exists" >&2; exit 73; }
[[ "$(findmnt -n -o FSTYPE --target /dev/shm)" == "tmpfs" ]] || {
  echo "Plaintext workspace must be backed by tmpfs" >&2
  exit 78
}
bundle_bytes="$(stat -c '%s' "${bundle}")"
available_bytes="$(( $(df -Pk /dev/shm | awk 'NR == 2 {print $4}') * 1024 ))"
required_bytes="$(( bundle_bytes * 3 + 16777216 ))"
(( available_bytes >= required_bytes )) || {
  echo "Insufficient memory-backed space for cohort decryption" >&2
  exit 75
}

umask 077
mkdir -p "${destination_root}" "${destination}"
temporary_tar="$(mktemp "${destination}/.bundle.XXXXXX.tar")"
validated=false
cleanup_on_exit() {
  rm -f -- "${temporary_tar}"
  if [[ "${validated}" != true && -d "${destination}" ]]; then
    find "${destination}" -type f -exec chmod u+w {} + 2>/dev/null || true
    find "${destination}" -depth -type d -exec chmod u+w {} + 2>/dev/null || true
    rm -rf -- "${destination}"
  fi
}
trap cleanup_on_exit EXIT
age --decrypt --identity "${identity_file}" --output "${temporary_tar}" "${bundle}"
while IFS= read -r entry; do
  [[ "${entry}" == data/* && "${entry}" != *".."* && "${entry}" != /* ]] || {
    echo "Unsafe encrypted archive entry" >&2
    exit 65
  }
done < <(tar -tf "${temporary_tar}")
tar -C "${destination}" -xf "${temporary_tar}"
rm -f -- "${temporary_tar}"

manifest="$(find "${destination}/data/manifests" -maxdepth 1 -type f -name '*.json' -print -quit)"
[[ -n "${manifest}" ]] || { echo "Sanitized bundle manifest is missing" >&2; exit 65; }
cohort_name="$(basename "${manifest}" .json)"
while IFS= read -r filename; do
  expected="$(jq -er --arg name "${filename}" '.output_sha256[$name]' "${manifest}")"
  payload="${destination}/data/curated/${cohort_name}/${filename}"
  actual="$(shasum -a 256 "${payload}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || { echo "Decrypted checksum mismatch" >&2; exit 65; }
done < <(jq -er '.gpu_transfer_allowlist[]' "${manifest}")
validated=true
echo "Decrypted allowlisted cohort into ephemeral storage."
