"""Restore an encrypted scientific run into the local ignored artifact namespace."""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tarfile
import tempfile
from typing import cast

import polars as pl

_FORBIDDEN_TEXT_COLUMNS = {"source_pt", "translation_en"}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_bundle(bundle: pathlib.Path) -> None:
    checksum = bundle.with_name(f"{bundle.name}.sha256")
    if not checksum.is_file() or checksum.is_symlink():
        raise ValueError("Encrypted bundle checksum is unavailable")
    lines = checksum.read_text(encoding="utf-8").splitlines()
    expected_suffix = f"  {bundle.name}"
    if (
        len(lines) != 1
        or len(lines[0]) != 64 + len(expected_suffix)
        or lines[0][64:] != expected_suffix
        or any(character not in "0123456789abcdef" for character in lines[0][:64])
    ):
        raise ValueError("Encrypted bundle checksum has an invalid format")
    if _sha256(bundle) != lines[0][:64]:
        raise ValueError("Encrypted bundle failed checksum verification")


def _extract(bundle: pathlib.Path, identity: pathlib.Path, staging: pathlib.Path) -> str:
    process = subprocess.Popen(
        ["age", "--decrypt", "--identity", str(identity), str(bundle)],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("Age did not expose a decrypted stream")
    top_levels: set[str] = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
            for member in archive:
                path = pathlib.PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts:
                    raise ValueError("Encrypted run contains an unsafe archive path")
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Encrypted run contains an unsupported archive member")
                top_levels.add(path.parts[0])
                archive.extract(member, path=staging, filter="data")
    finally:
        process.stdout.close()
    if process.wait() != 0:
        raise RuntimeError("Age failed to decrypt the scientific run")
    expected = bundle.name.removesuffix(".age")
    if top_levels != {expected}:
        raise ValueError("Encrypted bundle top level differs from its run identifier")
    return expected


def _verify_run(run_root: pathlib.Path) -> None:
    seal_path = run_root / "run_seal.json"
    terminal_path = run_root / "manifests" / "999_end.json"
    seal_raw: object = json.loads(seal_path.read_text(encoding="utf-8"))
    terminal_raw: object = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not isinstance(seal_raw, dict) or not isinstance(terminal_raw, dict):
        raise ValueError("Restored run has invalid provenance documents")
    seal = cast(dict[str, object], seal_raw)
    terminal = cast(dict[str, object], terminal_raw)
    if seal.get("run_id") != run_root.name or terminal.get("status") != "completed":
        raise ValueError("Restored run identity or terminal status is invalid")
    expected_raw = seal.get("files_sha256")
    if not isinstance(expected_raw, dict):
        raise ValueError("Restored run seal has no file map")
    expected = cast(dict[str, object], expected_raw)
    actual_files = sorted(
        path for path in run_root.rglob("*") if path.is_file() and path != seal_path
    )
    if len(actual_files) != len(expected):
        raise ValueError("Restored run differs from the sealed file count")
    for path in actual_files:
        relative = path.relative_to(run_root).as_posix()
        if expected.get(relative) != _sha256(path):
            raise ValueError("Restored run failed scientific seal verification")
    for path in run_root.rglob("*.parquet"):
        if _FORBIDDEN_TEXT_COLUMNS.intersection(pl.read_parquet_schema(path)):
            raise ValueError("Restored scientific artifact contains response text")


def _make_read_only(run_root: pathlib.Path) -> None:
    for path in run_root.rglob("*"):
        if path.is_file():
            path.chmod(0o440)
    for path in sorted(
        (path for path in run_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        path.chmod(0o550)
    run_root.chmod(0o550)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=pathlib.Path)
    parser.add_argument("identity", type=pathlib.Path)
    parser.add_argument("destination_root", type=pathlib.Path)
    args = parser.parse_args()

    bundle = args.bundle.expanduser().resolve(strict=True)
    identity = args.identity.expanduser().resolve(strict=True)
    destination_root = args.destination_root.expanduser().resolve(strict=True)
    if bundle.suffix != ".age" or bundle.is_symlink() or identity.is_symlink():
        raise ValueError("Bundle or identity path is invalid")
    _verify_bundle(bundle)

    expected_name = bundle.name.removesuffix(".age")
    destination = destination_root / expected_name
    if destination.exists():
        raise FileExistsError("Restored run destination already exists")
    with tempfile.TemporaryDirectory(
        prefix=".restore-sealed-run-", dir=destination_root
    ) as temporary:
        staging = pathlib.Path(temporary)
        run_name = _extract(bundle, identity, staging)
        run_root = staging / run_name
        _verify_run(run_root)
        os.replace(run_root, destination)
        _make_read_only(destination)
    print(f"Restored and verified sealed text-free run {expected_name}")


if __name__ == "__main__":
    main()
