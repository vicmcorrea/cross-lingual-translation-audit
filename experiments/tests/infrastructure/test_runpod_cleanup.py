import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = PROJECT_ROOT / "infrastructure" / "runpod" / "scripts"


def _fake_environment(tmp_path: Path, pod_name: str) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    state_file = tmp_path / "deleted"
    runpodctl = fake_bin / "runpodctl"
    runpodctl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$1:$2" in
  pod:get)
    [[ ! -f "${STATE_FILE}" ]] || exit 1
    printf '{"id":"pod123","name":"%s","status":"RUNNING"}\n' "${POD_NAME}"
    ;;
  pod:delete)
    touch "${STATE_FILE}"
    ;;
  pod:list)
    if [[ -f "${STATE_FILE}" ]]; then
      printf '[]\n'
    else
      printf '[{"id":"pod123","name":"%s","status":"RUNNING"}]\n' "${POD_NAME}"
    fi
    ;;
  *)
    exit 64
    ;;
esac
""",
        encoding="utf-8",
    )
    runpodctl.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "POD_NAME": pod_name,
            "STATE_FILE": str(state_file),
        }
    )
    return environment, state_file


def test_cleanup_wrapper_deletes_pod_when_workflow_fails(tmp_path: Path) -> None:
    environment, state_file = _fake_environment(tmp_path, "cross-lingual-translation-audit-test")
    lifecycle_log = tmp_path / "lifecycle.jsonl"
    result = subprocess.run(
        [
            str(SCRIPT_ROOT / "run_with_pod_cleanup.sh"),
            "pod123",
            str(lifecycle_log),
            "--",
            "bash",
            "-c",
            "exit 17",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 17
    assert state_file.is_file()
    events = [json.loads(line)["event"] for line in lifecycle_log.read_text(encoding="utf-8").splitlines()]
    assert events == [
        "pod_workflow_started",
        "pod_workflow_finished",
        "pod_deletion_started",
        "pod_deletion_verified",
    ]

    verification = subprocess.run(
        [str(SCRIPT_ROOT / "verify_no_project_pods.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert verification.returncode == 0


def test_deletion_refuses_nonproject_pod(tmp_path: Path) -> None:
    environment, state_file = _fake_environment(tmp_path, "unrelated-workload")
    result = subprocess.run(
        [str(SCRIPT_ROOT / "delete_project_pod.sh"), "pod123"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 65
    assert not state_file.exists()
    assert "outside the project naming boundary" in result.stderr


def test_completed_stage_mirror_publishes_only_after_checksum_verification(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_remote = tmp_path / "remote"
    remote_stages = fake_remote / "stages"
    remote_stages.mkdir(parents=True)
    bundle = remote_stages / "compute_embeddings_test.age"
    bundle.write_bytes(b"encrypted-scientific-artifact")
    digest = subprocess.run(
        ["shasum", "-a", "256", str(bundle)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[0]
    (remote_stages / f"{bundle.name}.sha256").write_text(
        f"{digest}  {bundle.name}\n",
        encoding="utf-8",
    )
    (fake_remote / "pipeline_terminal.json").write_text(
        '{"status":"completed"}\n',
        encoding="utf-8",
    )

    runpodctl = fake_bin / "runpodctl"
    runpodctl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '{"id":"pod123","name":"cross-lingual-translation-audit-test"}\n'
""",
        encoding="utf-8",
    )
    runpodctl.chmod(0o755)
    ssh = fake_bin / "ssh"
    ssh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
arguments="$*"
if [[ "${arguments}" == *"find /workspace/output-encrypted/stages"* ]]; then
  printf 'compute_embeddings_test.age.sha256\n'
elif [[ "${arguments}" == *"pipeline_terminal.json"* ]]; then
  exit 0
else
  cat >/dev/null || true
fi
""",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    rsync = fake_bin / "rsync"
    rsync.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
destination="${!#}"
if [[ "${destination}" == *.json ]]; then
  mkdir -p "$(dirname "${destination}")"
else
  mkdir -p "${destination}"
fi
for argument in "$@"; do
  [[ "${argument}" == root@*:/workspace/output-encrypted/* ]] || continue
  relative="${argument#*:/workspace/output-encrypted/}"
  cp "${FAKE_REMOTE}/${relative}" "${destination}"
done
""",
        encoding="utf-8",
    )
    rsync.chmod(0o755)
    ssh_key = tmp_path / "key"
    ssh_key.write_text("test-only", encoding="utf-8")
    local_export = tmp_path / "export"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "FAKE_REMOTE": str(fake_remote),
            "TRANSLATION_AUDIT_MIRROR_POLL_SECONDS": "1",
        }
    )

    result = subprocess.run(
        [
            str(SCRIPT_ROOT / "mirror_completed_artifacts.sh"),
            "pod123",
            "example.test",
            "22",
            str(ssh_key),
            str(local_export),
            "30",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert (local_export / "stages" / bundle.name).read_bytes() == bundle.read_bytes()
    receipt = json.loads(
        (local_export / "receipts" / f"{bundle.name}.receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["pod_id"] == "pod123"
    assert receipt["sha256"] == digest
    assert json.loads((local_export / "pipeline_terminal.json").read_text())["status"] == (
        "completed"
    )
