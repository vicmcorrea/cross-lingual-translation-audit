#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 7 ]] || {
  echo "Usage: execute_full_pipeline.sh <pod-id> <ssh-host> <ssh-port> <ssh-key> <cohort.age> <age-recipient> <new-export-dir> [completed-run.age ...]" >&2
  exit 64
}

pod_id="$1"
ssh_host="$2"
ssh_port="$3"
ssh_key="$4"
cohort_bundle="$5"
age_recipient="$6"
export_dir="$7"
resume_bundles=("${@:8}")
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../../.." && pwd)"
hf_token_file="${TRANSLATION_AUDIT_HF_TOKEN_FILE:?Set the approved Hugging Face token file path}"
age_identity_file="${TRANSLATION_AUDIT_AGE_IDENTITY_FILE:?Set the approved Age identity file path}"

[[ "${pod_id}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid Pod ID" >&2; exit 64; }
[[ "${ssh_host}" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "Invalid SSH host" >&2; exit 64; }
[[ "${ssh_port}" =~ ^[0-9]+$ ]] || { echo "Invalid SSH port" >&2; exit 64; }
[[ -f "${ssh_key}" && -f "${cohort_bundle}" && -f "${cohort_bundle}.sha256" ]] || {
  echo "SSH key or encrypted cohort input is unavailable" >&2
  exit 66
}
[[ -f "${hf_token_file}" && -f "${age_identity_file}" ]] || {
  echo "Approved local secret files are unavailable" >&2
  exit 78
}
(( ${#resume_bundles[@]} <= 3 )) || { echo "At most three completed runs can be imported" >&2; exit 64; }
for completed_run_bundle in "${resume_bundles[@]}"; do
  [[ "${completed_run_bundle}" == *.age \
    && -f "${completed_run_bundle}" \
    && ! -L "${completed_run_bundle}" \
    && -f "${completed_run_bundle}.sha256" \
    && ! -L "${completed_run_bundle}.sha256" ]] || {
    echo "Completed run bundle or checksum is unavailable" >&2
    exit 66
  }
done
[[ "${age_recipient}" =~ ^age1[a-z0-9]+$ ]] || { echo "Invalid Age recipient" >&2; exit 64; }
[[ ! -e "${export_dir}" ]] || { echo "Export directory must be new" >&2; exit 73; }

(
  cd "$(dirname "${cohort_bundle}")"
  shasum -a 256 -c "$(basename "${cohort_bundle}.sha256")"
)

ssh_options=(
  -i "${ssh_key}"
  -p "${ssh_port}"
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -o StrictHostKeyChecking=accept-new
)
remote="root@${ssh_host}"
ready=false
for _attempt in {1..60}; do
  if ssh "${ssh_options[@]}" "${remote}" true 2>/dev/null; then
    ready=true
    break
  fi
  sleep 5
done
[[ "${ready}" == true ]] || { echo "Pod SSH did not become ready" >&2; exit 75; }

ssh "${ssh_options[@]}" "${remote}" \
  'umask 077; mkdir -p /opt/translation-audit /workspace/input-encrypted/cohort /workspace/input-encrypted/resume /workspace/output-encrypted/stages /workspace/output-encrypted/acks /workspace/cache /workspace/runs /run/secrets'

printf -v rsync_shell 'ssh -i %q -p %q -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new' \
  "${ssh_key}" "${ssh_port}"

rsync_with_retry() {
  local attempt
  local status=1
  for attempt in {1..5}; do
    rsync "$@" && return 0
    status="$?"
    echo "Encrypted transfer attempt ${attempt} failed; preserving the Pod and retrying." >&2
    sleep 5
  done
  return "${status}"
}

rsync_with_retry --archive --no-owner --no-group --compress --partial --partial-dir=.rsync-partial \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude '.venv/' \
  --exclude '**/.venv/' \
  --exclude '**/__pycache__/' \
  --exclude '**/.pytest_cache/' \
  --exclude '**/.ruff_cache/' \
  --exclude '**/.coverage' \
  --exclude '/artifacts/' \
  --exclude '/data/' \
  --exclude '/third_party/' \
  --exclude '/experiments/legacy/' \
  -e "${rsync_shell}" \
  "${project_root}/" "${remote}:/opt/translation-audit/"

rsync_with_retry --archive --no-owner --no-group --partial --partial-dir=.rsync-partial -e "${rsync_shell}" \
  "${cohort_bundle}" "${cohort_bundle}.sha256" \
  "${remote}:/workspace/input-encrypted/cohort/"
for completed_run_bundle in "${resume_bundles[@]}"; do
  (
    cd "$(dirname "${completed_run_bundle}")"
    shasum -a 256 -c "$(basename "${completed_run_bundle}.sha256")"
  )
  rsync_with_retry --archive --no-owner --no-group --partial --partial-dir=.rsync-partial -e "${rsync_shell}" \
    "${completed_run_bundle}" "${completed_run_bundle}.sha256" \
    "${remote}:/workspace/input-encrypted/resume/"
done
rsync_with_retry --archive --no-owner --no-group --partial -e "${rsync_shell}" \
  "${hf_token_file}" "${remote}:/run/secrets/hf-token"
rsync_with_retry --archive --no-owner --no-group --partial -e "${rsync_shell}" \
  "${age_identity_file}" "${remote}:/run/secrets/age-identity"

ssh "${ssh_options[@]}" "${remote}" bash -s <<'REMOTE_SETUP'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends age ca-certificates git jq rsync util-linux
rm -rf /var/lib/apt/lists/*
python3 -m pip install --break-system-packages --no-cache-dir uv==0.12.5
chmod 0400 /run/secrets/hf-token /run/secrets/age-identity
mapfile -t resume_bundles < <(find /workspace/input-encrypted/resume -maxdepth 1 -type f -name '*.age' -print | sort)
(( ${#resume_bundles[@]} <= 3 )) || { echo "More than three completed-run bundles were supplied" >&2; exit 65; }
declare -A restored_stages=()
for resume_bundle in "${resume_bundles[@]}"; do
  restore_record="$(/opt/translation-audit/infrastructure/runpod/scripts/restore_completed_run.sh "${resume_bundle}")"
  IFS=$'\t' read -r restored_stage restored_path <<<"${restore_record}"
  [[ -n "${restored_stage}" && -n "${restored_path}" && -z "${restored_stages[${restored_stage}]:-}" ]] || {
    echo "Duplicate or invalid imported stage" >&2
    exit 65
  }
  restored_stages["${restored_stage}"]="${restored_path}"
  marker="/dev/shm/translation-audit/restored-${restored_stage}-path"
  printf '%s\n' "${restored_path}" > "${marker}"
  chmod 0400 "${marker}"
  echo "Restored and verified completed ${restored_stage} run before environment setup."
done
export UV_CACHE_DIR=/workspace/cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/cache/python
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export HF_TOKEN="$(tr -d '\r\n' < /run/secrets/hf-token)"
cd /opt/translation-audit/experiments
uv sync --frozen --extra gpu --python 3.14.7
uv run --frozen python - <<'PY'
import sys
if sys.version_info[:3] != (3, 14, 7) or sys.version_info.releaselevel != "final":
    raise SystemExit(f"Expected stable Python 3.14.7, found {sys.version}")
print({"python_version": sys.version.split()[0], "releaselevel": sys.version_info.releaselevel})
PY
uv run --frozen python - <<'PY'
import json
from pathlib import Path

import polars as pl

root = Path("/dev/shm/translation-audit")
for stage in ("compute_embeddings", "estimate_translation_quality", "compute_emotion_features"):
    marker = root / f"restored-{stage}-path"
    if not marker.is_file():
        continue
    run = Path(marker.read_text(encoding="utf-8").strip()).resolve(strict=True)
    if not run.is_relative_to(root / "imported-runs"):
        raise SystemExit("Imported run escaped the approved tmpfs root")
    forbidden = {"source_pt", "translation_en"}
    parquet_paths = sorted((run / "artifacts").rglob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"Imported {stage} run has no Parquet artifacts")
    for parquet_path in parquet_paths:
        if forbidden.intersection(pl.read_parquet_schema(parquet_path)):
            raise SystemExit(f"Imported {stage} run contains response text")
    if stage == "compute_embeddings":
        manifest = json.loads((run / "artifacts/embeddings/manifest.json").read_text())
        if manifest.get("model") != {
            "repository": "Qwen/Qwen3-Embedding-4B",
            "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
            "fingerprint": manifest.get("model", {}).get("fingerprint"),
            "normalize_embeddings": True,
        }:
            raise SystemExit("Imported embedding run is not the pinned 4B baseline")
    elif stage == "estimate_translation_quality":
        manifest = json.loads(
            (run / "artifacts/translation_quality/manifest.json").read_text()
        )
        if (
            manifest.get("model_repository") != "Unbabel/wmt23-cometkiwi-da-xl"
            or manifest.get("model_revision")
            != "33858b2239a139d497d9c74952c88b89a8c06213"
            or manifest.get("reference_free") is not True
        ):
            raise SystemExit("Imported COMET run does not match the frozen protocol")
    else:
        manifest = json.loads((run / "artifacts/emotion_features/manifest.json").read_text())
        if (
            manifest.get("revision") != "17b16d736e6935a63ccbbddeff739cd7baddb66c"
            or manifest.get("response_text_columns_persisted") is not False
        ):
            raise SystemExit("Imported emotion run does not match the frozen protocol")
print("Verified imported model identities and text-free schemas.")
PY
uv run --frozen ruff check src tests
uv run --frozen pyright
uv run --frozen pytest tests/unit
uv run --frozen python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable')
if torch.version.cuda != "12.8":
    raise SystemExit(f"Expected the locked CUDA 12.8 PyTorch runtime, found {torch.version.cuda}")
print({
    'cuda_available': True,
    'torch_version': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'gpu_count': torch.cuda.device_count(),
    'gpu_name': torch.cuda.get_device_name(0),
})
PY
if [[ ! -f /dev/shm/translation-audit/restored-estimate_translation_quality-path ]]; then
  cd /opt/translation-audit/tools/comet
  uv sync --frozen --python 3.11
  uv run --frozen ruff check .
  comet_smoke_output="$(
    printf '%s\n' '{"request_id":1,"samples":[{"src":"Este e um teste de integracao.","mt":"This is an integration test."}]}' \
      | timeout 1800 uv run --frozen python worker.py \
          --repository Unbabel/wmt23-cometkiwi-da-xl \
          --revision 33858b2239a139d497d9c74952c88b89a8c06213 \
          --batch-size 1 \
          --device cuda \
          --precision float16
  )"
  jq -s -e \
    'length == 2 and .[0].status == "ready" and .[1].request_id == 1 and (.[1].scores | length) == 1' \
    <<<"${comet_smoke_output}" >/dev/null
  echo "Pinned COMET worker integration smoke passed."
else
  echo "Skipped COMET setup because its sealed run was restored."
fi
if [[ ! -f /dev/shm/translation-audit/restored-compute_emotion_features-path ]]; then
  cd /opt/translation-audit/tools/emoatlas
  uv sync --frozen --python 3.11
  uv run --frozen ruff check .
  uv run --frozen pyright
  uv run --frozen pytest
else
  echo "Skipped EmoAtlas setup because its sealed run was restored."
fi
REMOTE_SETUP

remote_status=0
mirror_output="$(mktemp "${TMPDIR:-/tmp}/translation-audit-mirror.XXXXXX.log")"
mirror_pid=""
cleanup_local_controller() {
  local exit_code="$?"
  trap - EXIT INT TERM
  if [[ -n "${mirror_pid}" ]] && kill -0 "${mirror_pid}" 2>/dev/null; then
    kill "${mirror_pid}" 2>/dev/null || true
    wait "${mirror_pid}" 2>/dev/null || true
  fi
  if [[ -n "${mirror_output}" ]]; then
    rm -f -- "${mirror_output}"
  fi
  exit "${exit_code}"
}
trap cleanup_local_controller EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${script_dir}/mirror_completed_artifacts.sh" \
  "${pod_id}" "${ssh_host}" "${ssh_port}" "${ssh_key}" "${export_dir}" 21000 \
  >"${mirror_output}" 2>&1 &
mirror_pid="$!"
ssh "${ssh_options[@]}" "${remote}" bash -s -- \
  "$(basename "${cohort_bundle}")" \
  "${age_recipient}" <<'REMOTE_RUN' || remote_status="$?"
set -euo pipefail
bundle_name="$1"
age_recipient="$2"
export UV_CACHE_DIR=/workspace/cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/cache/python
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export HF_TOKEN="$(tr -d '\r\n' < /run/secrets/hf-token)"
export TRANSLATION_AUDIT_DATA_ROOT=/dev/shm/translation-audit/plaintext/current/data
export TRANSLATION_AUDIT_ARTIFACT_ROOT=/workspace

remote_cleanup() {
  local exit_code="$?"
  local cleanup_status=0
  local terminal_status="failed"
  local terminal_tmp="/workspace/output-encrypted/.pipeline_terminal.tmp.$$"
  trap - EXIT INT TERM
  /opt/translation-audit/infrastructure/runpod/scripts/cleanup_plaintext.sh || cleanup_status="$?"
  rm -f /run/secrets/hf-token /run/secrets/age-identity
  if (( exit_code == 0 && cleanup_status != 0 )); then
    exit_code="${cleanup_status}"
  fi
  if (( exit_code == 0 )); then
    terminal_status="completed"
  fi
  jq -nc \
    --arg status "${terminal_status}" \
    --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{status: $status, finished_at: $timestamp, namespace: "completed_stage"}' \
    > "${terminal_tmp}"
  chmod 0400 "${terminal_tmp}"
  mv "${terminal_tmp}" /workspace/output-encrypted/pipeline_terminal.json
  exit "${exit_code}"
}
trap remote_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

embedding_4b_manifest=""
comet_manifest=""
emotion_manifest=""
if [[ -f /dev/shm/translation-audit/restored-compute_embeddings-path ]]; then
  restored_embedding_run="$(cat /dev/shm/translation-audit/restored-compute_embeddings-path)"
  [[ "${restored_embedding_run}" == /dev/shm/translation-audit/imported-runs/compute_embeddings_* ]] || {
    echo "Restored embedding path is outside the approved tmpfs root" >&2
    exit 65
  }
  /opt/translation-audit/infrastructure/runpod/scripts/verify_run_seal.sh \
    "${restored_embedding_run}" >/dev/null
  embedding_4b_manifest="${restored_embedding_run}/manifests/999_end.json"
fi
if [[ -f /dev/shm/translation-audit/restored-estimate_translation_quality-path ]]; then
  restored_comet_run="$(cat /dev/shm/translation-audit/restored-estimate_translation_quality-path)"
  [[ "${restored_comet_run}" == /dev/shm/translation-audit/imported-runs/estimate_translation_quality_* ]] || {
    echo "Restored COMET path is outside the approved tmpfs root" >&2
    exit 65
  }
  /opt/translation-audit/infrastructure/runpod/scripts/verify_run_seal.sh "${restored_comet_run}" >/dev/null
  comet_manifest="${restored_comet_run}/manifests/999_end.json"
fi
if [[ -f /dev/shm/translation-audit/restored-compute_emotion_features-path ]]; then
  restored_emotion_run="$(cat /dev/shm/translation-audit/restored-compute_emotion_features-path)"
  [[ "${restored_emotion_run}" == /dev/shm/translation-audit/imported-runs/compute_emotion_features_* ]] || {
    echo "Restored emotion path is outside the approved tmpfs root" >&2
    exit 65
  }
  /opt/translation-audit/infrastructure/runpod/scripts/verify_run_seal.sh "${restored_emotion_run}" >/dev/null
  emotion_manifest="${restored_emotion_run}/manifests/999_end.json"
fi

/opt/translation-audit/infrastructure/runpod/scripts/decrypt_bundle.sh \
  "/workspace/input-encrypted/cohort/${bundle_name}"

cd /opt/translation-audit/experiments
run_stage() {
  local stage_name="$1"
  shift
  local run_uuid
  local run_date
  local run_dir
  local run_name
  local bundle_name
  local bundle_digest
  local ack_path
  run_uuid="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
  run_date="$(date -u +%Y-%m-%d)"
  run_dir="/workspace/runs/${run_date}/${stage_name}_$(date -u +%H-%M-%S)_${run_uuid}"
  uv run --frozen translation-audit \
    "stage=${stage_name}" \
    runtime=runpod_secure \
    runtime.execute=true \
    runtime.resource_creation_allowed=true \
    project.imported_run_root=/dev/shm/translation-audit/imported-runs \
    "hydra.run.dir=${run_dir}" \
    "$@"
  jq -e --arg stage "${stage_name}" \
    '.status == "completed" and .stage == $stage' \
    "${run_dir}/manifests/999_end.json" >/dev/null
  [[ -f "${run_dir}/run_seal.json" ]] || { echo "Run was not sealed" >&2; exit 70; }
  run_name="$(basename "${run_dir}")"
  bundle_name="${run_name}.age"
  /opt/translation-audit/infrastructure/runpod/scripts/package_artifacts.sh \
    "${run_dir}" \
    "${age_recipient}" \
    "/workspace/output-encrypted/stages/${bundle_name}"
  bundle_digest="$(awk '{print $1}' "/workspace/output-encrypted/stages/${bundle_name}.sha256")"
  [[ "${bundle_digest}" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid completed-stage digest" >&2; exit 65; }
  ack_path="/workspace/output-encrypted/acks/${bundle_name}.ack"
  for _ack_attempt in {1..900}; do
    [[ ! -f /workspace/output-encrypted/mirror_failed ]] || {
      echo "Local artifact mirror failed before acknowledgement" >&2
      exit 74
    }
    if [[ -f "${ack_path}" ]] && grep -Fxq "${bundle_digest}" "${ack_path}"; then
      break
    fi
    sleep 2
  done
  [[ -f "${ack_path}" ]] && grep -Fxq "${bundle_digest}" "${ack_path}" || {
    echo "Completed stage was not acknowledged by local verified storage" >&2
    exit 75
  }
  LAST_RUN_DIR="${run_dir}"
  LAST_MANIFEST="${run_dir}/manifests/999_end.json"
}

run_stage prepare_cohort
prepare_manifest="${LAST_MANIFEST}"

run_stage validate_languages \
  "+stage.upstream_manifests.prepare_cohort=${prepare_manifest}"
language_manifest="${LAST_MANIFEST}"

if [[ -z "${embedding_4b_manifest}" ]]; then
  run_stage compute_embeddings \
    "+stage.upstream_manifests.validate_languages=${language_manifest}"
  embedding_4b_manifest="${LAST_MANIFEST}"
fi

if [[ -z "${comet_manifest}" ]]; then
  run_stage estimate_translation_quality \
    "+stage.upstream_manifests.validate_languages=${language_manifest}"
  comet_manifest="${LAST_MANIFEST}"
fi

if [[ -z "${emotion_manifest}" ]]; then
  run_stage compute_emotion_features \
    "+stage.upstream_manifests.validate_languages=${language_manifest}"
  emotion_manifest="${LAST_MANIFEST}"
fi

run_stage analyze \
  "+stage.upstream_manifests.compute_embeddings=${embedding_4b_manifest}" \
  "+stage.upstream_manifests.estimate_translation_quality=${comet_manifest}" \
  "+stage.upstream_manifests.compute_emotion_features=${emotion_manifest}"

run_stage compute_embeddings encoder=qwen3_embedding_8b \
  "+stage.upstream_manifests.validate_languages=${language_manifest}"
embedding_8b_manifest="${LAST_MANIFEST}"

run_stage analyze \
  "+stage.upstream_manifests.compute_embeddings=${embedding_8b_manifest}" \
  "+stage.upstream_manifests.estimate_translation_quality=${comet_manifest}" \
  "+stage.upstream_manifests.compute_emotion_features=${emotion_manifest}"
REMOTE_RUN

if (( remote_status != 0 )) && ! ssh "${ssh_options[@]}" "${remote}" \
  '[[ -f /workspace/output-encrypted/pipeline_terminal.json ]]' 2>/dev/null; then
  if ! ssh "${ssh_options[@]}" "${remote}" bash -s <<'REMOTE_TERMINAL' 2>/dev/null
set -euo pipefail
terminal_tmp="/workspace/output-encrypted/.pipeline_terminal.host.$$"
jq -nc \
  --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{status: "failed", finished_at: $timestamp, namespace: "completed_stage"}' \
  > "${terminal_tmp}"
chmod 0400 "${terminal_tmp}"
mv "${terminal_tmp}" /workspace/output-encrypted/pipeline_terminal.json
REMOTE_TERMINAL
  then
    kill "${mirror_pid}" 2>/dev/null || true
  fi
fi

mirror_status=0
wait "${mirror_pid}" || mirror_status="$?"
mirror_pid=""
if (( mirror_status != 0 )); then
  cat "${mirror_output}" >&2
else
  cat "${mirror_output}"
fi
rm -f -- "${mirror_output}"
mirror_output=""
if (( remote_status != 0 )); then
  exit "${remote_status}"
fi
if (( mirror_status != 0 )); then
  exit "${mirror_status}"
fi
echo "Collected, verified, and acknowledged every completed scientific stage."
