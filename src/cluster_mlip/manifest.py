from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from . import __version__

_PACKAGE_DIR = Path(__file__).resolve().parent


class FileRecord(TypedDict):
    path: str
    size_bytes: int | None
    sha256: str | None


class ExperimentManifest(TypedDict):
    generated_utc: str
    hostname: str
    python: str
    cluster_mlip_version: str
    git_commit: str | None
    git_dirty: bool | None
    dataset_dir: str
    dataset_files: dict[str, FileRecord]
    config: FileRecord | None
    notes: str


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> FileRecord:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path),
    }


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_PACKAGE_DIR,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def write_experiment_manifest(
    dataset_dir: Path,
    output: Path,
    config: Path | None = None,
    notes: str = "",
) -> ExperimentManifest:
    """Bundle a dataset + training config + code revision + checksums.

    The README's "Model and data storage" section says raw warehouses,
    datasets, and checkpoints belong in object storage, not Git, and that
    their "checksums and immutable storage URIs" should be recorded in
    experiment manifests -- but nothing in the pipeline wrote that record.
    This closes that gap for the dataset side (the storage-URI half is still
    on the caller, since this tool has no opinion on where object storage
    lives).
    """
    dataset_dir = dataset_dir.resolve()
    dataset_files: dict[str, FileRecord] = {}
    for name in ("all.extxyz", "train.extxyz", "valid.extxyz", "test.extxyz"):
        path = dataset_dir / name
        if path.is_file():
            dataset_files[name] = _file_record(path)

    config_record: FileRecord | None = None
    if config is not None:
        config_record = _file_record(config.resolve())

    commit = _git("rev-parse", "HEAD")
    dirty_output = _git("status", "--porcelain") if commit is not None else None

    manifest: ExperimentManifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "cluster_mlip_version": __version__,
        "git_commit": commit,
        "git_dirty": None if dirty_output is None else bool(dirty_output),
        "dataset_dir": str(dataset_dir),
        "dataset_files": dataset_files,
        "config": config_record,
        "notes": notes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
