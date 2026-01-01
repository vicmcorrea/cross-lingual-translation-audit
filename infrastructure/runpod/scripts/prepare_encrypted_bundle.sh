#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 4 ]] || {
  echo "Usage: prepare_encrypted_bundle.sh <curated-dir> <sanitized-manifest> <age-recipient> <output.age>" >&2
  exit 64
}
curated_dir="$1"
manifest="$2"
recipient="$3"
output="$4"
[[ -d "${curated_dir}" && -f "${manifest}" ]] || { echo "Cohort or manifest is missing" >&2; exit 66; }
[[ "${output}" == *.age && ! -e "${output}" && ! -e "${output}.sha256" ]] || {
  echo "Output must be a new .age path" >&2
  exit 73
}
for executable in age jq shasum tar; do
  command -v "${executable}" >/dev/null || { echo "Missing dependency ${executable}" >&2; exit 69; }
done

staging="$(mktemp -d -t translation-audit-bundle.XXXXXX)"
temporary_tar="$(mktemp -t translation-audit-bundle.XXXXXX)"
completed=false
cleanup_on_exit() {
  chmod -R u+w "${staging}" 2>/dev/null || true
  rm -rf -- "${staging}"
  rm -f -- "${temporary_tar}"
  if [[ "${completed}" != true ]]; then
    chmod u+w "${output}" "${output}.sha256" 2>/dev/null || true
    rm -f -- "${output}" "${output}.sha256"
  fi
}
trap cleanup_on_exit EXIT
cohort_name="$(basename "${curated_dir}")"
payload_dir="${staging}/data/curated/${cohort_name}"
mkdir -p "${payload_dir}" "${staging}/data/manifests"

while IFS= read -r filename; do
  [[ "${filename}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Unsafe allowlist entry" >&2; exit 65; }
  source_path="${curated_dir}/${filename}"
  [[ -f "${source_path}" ]] || { echo "Missing allowlisted file ${filename}" >&2; exit 66; }
  expected="$(jq -er --arg name "${filename}" '.output_sha256[$name]' "${manifest}")"
  actual="$(shasum -a 256 "${source_path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || { echo "Checksum mismatch for ${filename}" >&2; exit 65; }
  install -m 0400 "${source_path}" "${payload_dir}/${filename}"
done < <(jq -er '.gpu_transfer_allowlist[]' "${manifest}")
install -m 0400 "${manifest}" "${staging}/data/manifests/${cohort_name}.json"

tar_options=(-C "${staging}" --no-xattrs)
COPYFILE_DISABLE=1 tar "${tar_options[@]}" -cf "${temporary_tar}" data
while IFS= read -r entry; do
  [[ "${entry}" == data/* && "${entry}" != *".."* && "${entry}" != /* ]] || {
    echo "Unsafe archive entry produced while preparing the bundle" >&2
    exit 65
  }
done < <(tar -tf "${temporary_tar}")
age --recipient "${recipient}" --output "${output}" "${temporary_tar}"
(
  cd "$(dirname "${output}")"
  shasum -a 256 "$(basename "${output}")" > "$(basename "${output}").sha256"
)
chmod 0400 "${output}" "${output}.sha256"
completed=true
echo "Created an encrypted, checksummed allowlisted bundle."
