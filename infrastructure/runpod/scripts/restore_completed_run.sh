#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: restore_completed_run.sh <run-bundle.age>" >&2; exit 64; }
bundle="$1"
identity_file="/run/secrets/age-identity"
destination_root="/dev/shm/translation-audit/imported-runs"
[[ "${bundle}" == /workspace/input-encrypted/resume/*.age && -f "${bundle}" && ! -L "${bundle}" ]] || {
  echo "Resume bundle must be a regular .age file below the resume namespace" >&2
  exit 66
}
[[ -f "${identity_file}" && ! -L "${identity_file}" ]] || {
  echo "Age identity is unavailable" >&2
  exit 78
}
checksum="${bundle}.sha256"
[[ -f "${checksum}" && ! -L "${checksum}" ]] || {
  echo "Resume bundle checksum is unavailable" >&2
  exit 66
}
bundle_name="$(basename "${bundle}")"
checksum_line="$(sed -n '1p' "${checksum}")"
[[ "$(wc -l < "${checksum}" | tr -d ' ')" == "1" \
  && "${checksum_line:0:64}" =~ ^[0-9a-f]{64}$ \
  && "${checksum_line:64}" == "  ${bundle_name}" ]] || {
  echo "Resume bundle checksum has an invalid format" >&2
  exit 65
}
[[ "$(shasum -a 256 "${bundle}" | awk '{print $1}')" == "${checksum_line:0:64}" ]] || {
  echo "Resume bundle failed checksum verification" >&2
  exit 65
}

[[ "$(findmnt -n -o FSTYPE --target /dev/shm)" == "tmpfs" ]] || {
  echo "Completed-run restore requires tmpfs" >&2
  exit 78
}

umask 077
mkdir -p /dev/shm/translation-audit "${destination_root}"
bundle_bytes="$(stat -c '%s' "${bundle}")"
available_bytes="$(( $(df -Pk /dev/shm | awk 'NR == 2 {print $4}') * 1024 ))"
(( available_bytes > bundle_bytes + 536870912 )) || {
  echo "Insufficient tmpfs space for the completed run" >&2
  exit 70
}
top_level="$(
  age --decrypt --identity "${identity_file}" "${bundle}" \
    | python3 -c '
import pathlib
import sys
import tarfile

top_levels = set()
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
    for member in archive:
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise SystemExit("unsafe archive path")
        if not (member.isfile() or member.isdir()):
            raise SystemExit("unsupported archive member type")
        top_levels.add(path.parts[0])
if len(top_levels) != 1:
    raise SystemExit("archive must contain exactly one run")
print(next(iter(top_levels)))
'
)"
case "${top_level}" in
  compute_embeddings_*) expected_stage="compute_embeddings" ;;
  estimate_translation_quality_*) expected_stage="estimate_translation_quality" ;;
  compute_emotion_features_*) expected_stage="compute_emotion_features" ;;
  *) echo "Resume bundle contains a stage that cannot be imported" >&2; exit 65 ;;
esac
[[ ! -e "${destination_root}/${top_level}" ]] || {
  echo "Completed run restore destination already exists" >&2
  exit 73
}
staging_root="$(mktemp -d "${destination_root}/.restore-${top_level}.XXXXXX")"
trap 'rm -rf -- "${staging_root}"' EXIT
age --decrypt --identity "${identity_file}" "${bundle}" \
  | tar --delay-directory-restore --no-same-owner --no-same-permissions --mode='u+rwX' \
      -C "${staging_root}" -xf -
run_dir="${staging_root}/${top_level}"
[[ -f "${run_dir}/run_seal.json" && -f "${run_dir}/manifests/999_end.json" ]] || {
  echo "Restored run is not sealed" >&2
  exit 65
}
[[ -z "$(find "${run_dir}" -mindepth 1 ! -type f ! -type d -print -quit)" ]] || {
  echo "Restored run contains an unsupported filesystem entry" >&2
  exit 65
}
jq -e --arg run_id "${top_level}" '.run_id == $run_id' "${run_dir}/run_seal.json" >/dev/null || {
  echo "Restored run seal has the wrong run identifier" >&2
  exit 65
}
jq -e --arg stage "${expected_stage}" \
  '.status == "completed" and .stage == $stage' \
  "${run_dir}/manifests/999_end.json" >/dev/null
"$(dirname "${BASH_SOURCE[0]}")/verify_run_seal.sh" "${run_dir}" >/dev/null
find "${run_dir}" -type f -exec chmod 0440 {} +
find "${run_dir}" -depth -type d -exec chmod 0550 {} +
final_run_dir="${destination_root}/${top_level}"
mv "${run_dir}" "${final_run_dir}"
printf '%s\t%s\n' "${expected_stage}" "${final_run_dir}"
