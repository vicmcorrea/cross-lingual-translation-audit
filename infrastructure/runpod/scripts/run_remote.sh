#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 3 ]] || {
  echo "Usage: run_remote.sh <ssh-host> <remote-project-dir> <hydra-overrides...>" >&2
  exit 64
}
host="$1"
project_dir="$2"
shift 2
[[ "${project_dir}" == /opt/translation-audit/experiments ]] || {
  echo "Unexpected remote project directory" >&2
  exit 64
}

ssh "${host}" bash -s -- "${project_dir}" "$@" <<'REMOTE_SCRIPT'
set -euo pipefail
project_dir="$1"
shift
cd "${project_dir}"
exec uv run translation-audit "$@"
REMOTE_SCRIPT
