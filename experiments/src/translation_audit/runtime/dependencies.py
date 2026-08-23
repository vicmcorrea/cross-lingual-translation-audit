"""Shared verification for sealed upstream scientific runs."""

import json
from pathlib import Path
from typing import cast

from omegaconf import DictConfig

from translation_audit.runtime.files import sha256_file

_RUNPOD_IMPORTED_ROOT = Path("/dev/shm/translation-audit/imported-runs")
_REQUIRED_SEALED_FILES = {
    "artifacts/config/overrides.txt",
    "artifacts/config/resolved.yaml",
    "manifests/000_start.json",
    "manifests/999_end.json",
}


def _allowed_run_roots(cfg: DictConfig) -> tuple[Path, ...]:
    local_root = (Path(str(cfg.project.artifact_root)).resolve() / "runs").resolve()
    roots = [local_root]
    imported_value = cfg.project.get("imported_run_root")
    if imported_value in (None, "", "null"):
        return tuple(roots)
    imported = Path(str(imported_value))
    if not imported.is_absolute():
        raise ValueError("project.imported_run_root must be an absolute path")
    if str(cfg.get("runtime", {}).get("name", "")) == "runpod_secure" and imported != _RUNPOD_IMPORTED_ROOT:
        raise ValueError("RunPod imported runs must use the fixed tmpfs boundary")
    if imported.exists():
        if imported.is_symlink():
            raise ValueError("project.imported_run_root cannot be a symlink")
        roots.append(imported.resolve(strict=True))
    return tuple(roots)


def verified_named_upstream_runs(
    cfg: DictConfig,
    expected_stages: dict[str, str],
) -> tuple[dict[str, Path], dict[str, str]]:
    """Return verified run roots keyed by a stable analytical role."""
    if not expected_stages:
        return {}, {}
    configured = cfg.stage.get("upstream_manifests", {})
    allowed_roots = _allowed_run_roots(cfg)
    run_roots: dict[str, Path] = {}
    fingerprints: dict[str, str] = {}
    for role, expected_stage in expected_stages.items():
        manifest_value = configured.get(role)
        if not manifest_value:
            raise ValueError(f"Missing upstream run manifest for role {role}")
        manifest_unresolved = Path(str(manifest_value))
        if manifest_unresolved.is_symlink():
            raise ValueError(f"Upstream manifest for {role} is outside the artifact root")
        manifest_path = manifest_unresolved.resolve(strict=True)
        if (
            manifest_path.name != "999_end.json"
            or manifest_path.parent.name != "manifests"
            or not manifest_path.is_file()
            or not any(manifest_path.is_relative_to(root) for root in allowed_roots)
        ):
            raise ValueError(f"Upstream manifest for {role} is outside the artifact root")
        manifest_value_raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_value_raw, dict):
            raise ValueError(f"Upstream run manifest is not a successful {expected_stage} stage")
        manifest = cast(dict[str, object], manifest_value_raw)
        if manifest.get("status") != "completed" or manifest.get("stage") != expected_stage:
            raise ValueError(f"Upstream run manifest is not a successful {expected_stage} stage")

        run_root = manifest_path.parents[1]
        seal_path = (run_root / "run_seal.json").resolve(strict=True)
        if seal_path.parent != run_root or not seal_path.is_file() or seal_path.is_symlink():
            raise ValueError(f"Upstream run for {role} has no valid seal")
        seal_value: object = json.loads(seal_path.read_text(encoding="utf-8"))
        if not isinstance(seal_value, dict):
            raise ValueError(f"Upstream run for {role} has no valid seal")
        sealed_files = cast(dict[str, object], seal_value).get("files_sha256")
        if not isinstance(sealed_files, dict):
            raise ValueError(f"Upstream run for {role} has no valid seal")
        sealed_file_map = cast(dict[object, object], sealed_files)
        if not _REQUIRED_SEALED_FILES.issubset(sealed_file_map):
            raise ValueError(f"Upstream run for {role} has an incomplete seal")
        for relative_path, expected_hash in sealed_file_map.items():
            if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
                raise ValueError(f"Upstream run seal verification failed for {role}")
            candidate_unresolved = run_root / relative_path
            if candidate_unresolved.is_symlink():
                raise ValueError(f"Upstream run seal verification failed for {role}")
            candidate = candidate_unresolved.resolve(strict=True)
            if (
                not candidate.is_relative_to(run_root)
                or not candidate.is_file()
                or sha256_file(candidate) != expected_hash
            ):
                raise ValueError(f"Upstream run seal verification failed for {role}")
        run_roots[role] = run_root
        fingerprints[role] = sha256_file(seal_path)
    return run_roots, fingerprints


def verified_upstream_runs(
    cfg: DictConfig,
    dependencies: tuple[str, ...],
) -> tuple[dict[str, Path], dict[str, str]]:
    """Return verified run roots keyed by their required stage names."""
    return verified_named_upstream_runs(cfg, {dependency: dependency for dependency in dependencies})
