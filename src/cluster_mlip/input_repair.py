"""Repair not-yet-run inputs whose Geom=Checkpoint stages still carry atom lines.

Before PR #22, ``prepare-spin-restarts`` wrote first-stage restarts that read
``Geom=Checkpoint`` but kept the stage's atom lines; Gaussian reads those as
the next input section and dies in l101 before computing anything. The fix
stops new ones, but restarts written earlier sit in the campaign waiting to be
submitted. This rewrites exactly those inputs, and only the broken stages:

- in each stage that reads ``Geom=Checkpoint`` and still has atoms, the atom
  lines between the charge/multiplicity line and the next blank line are
  dropped (the structure comes from the checkpoint, as the route already says);
- the file ends with the blank line that terminates its last section;
- nothing else changes -- every other line is checked to be identical.

Inputs that already ran (any non-empty log) are left alone: the restart pass
rebuilds those from their checkpoints. Inputs whose attempt is unconfirmed
(a ``.started`` marker with no later ``.finished``) are skipped unless
``assume_stopped``. Originals and the manifest are backed up first, and the
manifest's ``input_sha256`` is updated for the repaired inputs.
"""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import write_text_lf
from .route_audit import find_manifest, read_manifest
from .routes import checkpoint_stages_with_coordinates, drop_molecule_coordinates

_LINK1_KEEP_RE = re.compile(r"(^\s*--\s*link1\s*--\s*$)", re.IGNORECASE | re.MULTILINE)
_ROUTE_LINE_RE = re.compile(r"^\s*#.*$", re.MULTILINE)
PLAN_COLUMNS = ["batch", "input", "repaired_stages", "atom_lines_removed",
                "sha256_before", "sha256_after", "backup"]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def repaired_text(text: str) -> tuple[str, list[int], int]:
    """``text`` with atom lines dropped from its Geom=Checkpoint stages.

    Returns the new text, the repaired stage indices and the number of lines
    removed. Raises if the result changes anything but those lines.
    """
    broken = checkpoint_stages_with_coordinates(text)
    if not broken:
        return text, [], 0
    parts = _LINK1_KEEP_RE.split(text)  # stage, separator, stage, separator, ...
    for stage in broken:
        index = 2 * stage
        section = parts[index]
        route = _ROUTE_LINE_RE.search(section)
        if route is None:
            raise ValueError(f"stage {stage} has no route line")
        parts[index] = section[:route.end()] + drop_molecule_coordinates(section[route.end():])
    new = "".join(parts)
    if not new.endswith("\n\n"):
        new = new.rstrip("\n") + "\n\n"
    if checkpoint_stages_with_coordinates(new):
        raise RuntimeError("repair left a Geom=Checkpoint stage with coordinates")
    if len(_LINK1_KEEP_RE.findall(new)) != len(_LINK1_KEEP_RE.findall(text)):
        raise RuntimeError("repair changed the number of Link1 stages")
    old_lines = [line for line in text.splitlines() if line.strip()]
    new_lines = [line for line in new.splitlines() if line.strip()]
    removed = len(old_lines) - len(new_lines)
    # Only lines were removed, in order: new_lines must be a subsequence of old_lines.
    cursor = iter(old_lines)
    if not all(any(line == candidate for candidate in cursor) for line in new_lines):
        raise RuntimeError("repair changed a line it should have kept")
    return new, broken, removed


def _unconfirmed(batch: Path, stem: str) -> bool:
    started, finished = batch / f"{stem}.started", batch / f"{stem}.finished"
    return started.is_file() and (
        not finished.is_file() or started.stat().st_mtime > finished.stat().st_mtime
    )


def repair_checkpoint_inputs(
    campaign: Path,
    *,
    start: int = 1,
    end: int | None = None,
    assume_stopped: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Find and (unless ``dry_run``) repair not-started inputs; see the module doc."""
    campaign = campaign.resolve()
    manifest = find_manifest(campaign)
    fields, rows = read_manifest(manifest)
    batches = sorted(
        (int(path.name[6:]), path) for path in (campaign / "slurm_batches").glob("batch_*")
        if path.is_dir() and path.name[6:].isdigit()
    )
    final = batches[-1][0] if batches else 0
    end = final if end is None else end
    if final and not 1 <= start <= end <= final:
        raise ValueError(f"batch range must satisfy 1 <= start <= end <= {final}")

    plan: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    writes: list[tuple[list[Path], str, str, dict[str, str]]] = []
    for number, batch in batches:
        if not start <= number <= end:
            continue
        listing = batch / "inputs.txt"
        if not listing.is_file():
            continue
        for name in listing.read_text(encoding="utf-8", errors="replace").split():
            listed = batch / name
            if not listed.is_file():
                continue
            text = listed.read_text(encoding="utf-8", errors="replace")
            new, stages, removed = repaired_text(text)
            if not stages:
                continue
            stem = Path(name).stem
            log = batch / f"{stem}.log"
            if log.is_file() and log.stat().st_size > 0:
                skipped.append({"batch": batch.name, "input": name, "reason": (
                    "already ran; prepare-spin-restarts rebuilds it from its checkpoint")})
                continue
            if _unconfirmed(batch, stem) and not assume_stopped:
                skipped.append({"batch": batch.name, "input": name, "reason": (
                    "activity_unconfirmed: pass --assume-stopped once both queues are clear")})
                continue
            # The listed file is a symlink to inputs/ or a copy of it; repair every
            # distinct real file that holds this input, and only if they agree.
            targets: dict[Path, Path] = {}
            for candidate in (listed, campaign / "inputs" / name):
                if candidate.is_file():
                    targets.setdefault(candidate.resolve(), candidate)
            texts = {path.read_text(encoding="utf-8", errors="replace") for path in targets}
            if len(texts) != 1:
                skipped.append({"batch": batch.name, "input": name, "reason": (
                    "the batch copy and inputs/ copy differ; repair by hand")})
                continue
            before, after = _sha256_text(text), _sha256_text(new)
            plan_row = {
                "batch": batch.name, "input": name,
                "repaired_stages": ";".join(map(str, stages)),
                "atom_lines_removed": str(removed),
                "sha256_before": before, "sha256_after": after, "backup": "",
            }
            plan.append(plan_row)
            writes.append((list(targets), new, after, plan_row))

    result: dict[str, Any] = {
        "campaign": str(campaign), "input_count": len(plan), "plan": plan,
        "skipped": skipped, "dry_run": dry_run, "backup": "",
    }
    if dry_run or not writes:
        return result

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = campaign / "input_repair_backups" / stamp
    backup.mkdir(parents=True)
    shutil.copy2(manifest, backup / manifest.name)
    repaired_names: dict[str, str] = {}
    for real_files, new, after, plan_row in writes:
        for real in real_files:
            relative = real.relative_to(campaign) if real.is_relative_to(campaign) else Path(real.name)
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(real, saved)
            write_text_lf(real, new)
        plan_row["backup"] = str(backup.relative_to(campaign))
        repaired_names[plan_row["input"]] = after
    for row in rows:
        name = Path((row.get("input") or "").strip()).name
        if name in repaired_names and row.get("input_sha256"):
            row["input_sha256"] = repaired_names[name]
    temporary = manifest.with_name(manifest.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)
    with (campaign / "input_repair_plan.csv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_COLUMNS)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(plan)
    result["backup"] = str(backup)
    return result
