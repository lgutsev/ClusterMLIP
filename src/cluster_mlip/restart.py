"""Restart interrupted Gaussian spin ladders in their existing batch folders."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .spin import SPIN_MANIFEST_COLUMNS, SPIN_RESTART_COLUMNS, parse_spin_diagnostics


_LINK1_RE = re.compile(r"^\s*--link1--\s*$", re.IGNORECASE | re.MULTILINE)
_ROUTE_RE = re.compile(r"^(\s*#.*)$", re.IGNORECASE | re.MULTILINE)
_CHK_RE = re.compile(r"^\s*%chk\s*=.*$", re.IGNORECASE | re.MULTILINE)
_OLDCHK_RE = re.compile(r"^\s*%oldchk\s*=.*$", re.IGNORECASE | re.MULTILINE)
_MARKER_SUFFIXES = (".status", ".rc", ".started", ".finished")


@dataclass
class RestartAction:
    input_path: str
    input_name: str
    batch: Path
    source_input: Path
    source_output: Path
    archived_output: Path
    source_checkpoint: Path
    seed_checkpoint: Path
    new_input: Path
    new_input_name: str
    new_text: str
    old_rows: list[dict[str, str]]
    new_rows: list[dict[str, str]]
    plan_row: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_locations(campaign: Path) -> dict[str, Path]:
    locations: dict[str, Path] = {}
    for batch in sorted((campaign / "slurm_batches").glob("batch_*")):
        listing = batch / "inputs.txt"
        if not listing.is_file():
            continue
        for name in listing.read_text(errors="replace").splitlines():
            name = name.strip()
            if not name:
                continue
            if name in locations:
                raise ValueError(f"input occurs in multiple active batches: {name}")
            locations[name] = batch
    return locations


def _attempt_is_unconfirmed(batch: Path, stem: str) -> bool:
    started = batch / f"{stem}.started"
    finished = batch / f"{stem}.finished"
    return started.is_file() and (
        not finished.is_file() or started.stat().st_mtime > finished.stat().st_mtime
    )


def _rewrite_stage(section: str, *, oldchk: str, chk: str) -> str:
    if not _CHK_RE.search(section):
        raise ValueError("generated stage has no %chk directive")
    section = _CHK_RE.sub(f"%chk={chk}", section, count=1)
    if _OLDCHK_RE.search(section):
        section = _OLDCHK_RE.sub(f"%oldchk={oldchk}", section, count=1)
    else:
        section = f"%oldchk={oldchk}\n" + section.lstrip("\n")
    route_match = _ROUTE_RE.search(section)
    if route_match is None:
        raise ValueError("generated stage has no Gaussian route")
    route = route_match.group(1).strip()
    geom = re.search(r"\bGeom\s*=\s*([^\s]+)", route, re.IGNORECASE)
    if geom is not None:
        route = route[:geom.start()] + "Geom=Checkpoint" + route[geom.end():]
    else:
        route += " Geom=Checkpoint"
    guess = re.search(r"\bGuess\s*=\s*(\([^)]*\)|[^\s]+)", route, re.IGNORECASE)
    if guess is not None:
        route = route[:guess.start()] + "Guess=Read" + route[guess.end():]
    else:
        route += " Guess=Read"
    section = section[:route_match.start()] + route + section[route_match.end():]
    return section.strip("\n")


def _active(row: dict[str, str]) -> bool:
    return row.get("submission_active", "").strip().lower() not in {"false", "0", "no"}


def _next_attempt(rows: list[dict[str, str]]) -> int:
    values = []
    for row in rows:
        try:
            values.append(int(row.get("restart_attempt", "") or 0))
        except ValueError:
            pass
    return max(values, default=0) + 1


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare_spin_restarts(
    campaign: Path,
    *,
    start: int = 1,
    end: int | None = None,
    assume_stopped: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Archive interrupted attempts and activate shortened inputs in place."""
    campaign = campaign.resolve()
    manifest = campaign / "spin_jobs.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing spin manifest: {manifest}")
    original_fields, all_rows = _read_csv(manifest)
    fields = list(dict.fromkeys(original_fields + SPIN_MANIFEST_COLUMNS + SPIN_RESTART_COLUMNS))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        if _active(row):
            grouped[row["input"]].append(row)
    locations = _batch_locations(campaign)
    final_batch = max((int(path.name[6:]) for path in locations.values()), default=0)
    end = final_batch if end is None else end
    if final_batch == 0:
        raise FileNotFoundError(f"no generated batch input lists under {campaign}")
    if not 1 <= start <= end <= final_batch:
        raise ValueError(f"batch range must satisfy 1 <= start <= end <= {final_batch}")

    actions: list[RestartAction] = []
    skipped: list[dict[str, str]] = []
    for input_path, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["stage_index"]))
        input_name = Path(input_path).name
        batch = locations.get(input_name)
        if batch is None:
            continue
        batch_number = int(batch.name[6:])
        if not start <= batch_number <= end:
            continue
        source_input = campaign / input_path
        stem = Path(input_name).stem
        source_output = batch / f"{stem}.log"
        if not source_output.is_file() or source_output.stat().st_size == 0:
            skipped.append({"input": input_path, "batch": batch.name, "reason": "not_started"})
            continue
        if _attempt_is_unconfirmed(batch, stem) and not assume_stopped:
            skipped.append({
                "input": input_path, "batch": batch.name, "reason": "activity_unconfirmed"
            })
            continue
        text = source_output.read_text(errors="replace")
        diagnostics = parse_spin_diagnostics(text)
        unfinished_index = next((
            index for index, row in enumerate(rows)
            if not any(
                item.charge == int(row["intended_charge"])
                and item.multiplicity == int(row["intended_multiplicity"])
                and item.normal_termination and item.optimized
                for item in diagnostics
            )
        ), None)
        if unfinished_index is None:
            skipped.append({"input": input_path, "batch": batch.name, "reason": "complete"})
            continue
        current = batch / rows[unfinished_index]["checkpoint"]
        predecessor_name = rows[unfinished_index].get("predecessor_checkpoint", "")
        candidates = [current] + ([batch / predecessor_name] if predecessor_name else [])
        source_checkpoint = next((
            candidate for candidate in candidates
            if candidate.is_file() and candidate.stat().st_size > 0
        ), None)
        if source_checkpoint is None:
            skipped.append({
                "input": input_path, "batch": batch.name,
                "reason": "unfinished_checkpoint_missing",
            })
            continue
        sections = _LINK1_RE.split(source_input.read_text(encoding="utf-8", errors="replace"))
        if len(sections) != len(rows):
            raise ValueError(f"{source_input}: Link1 sections disagree with active manifest rows")
        root_input = rows[0].get("restart_root_input") or input_path
        related = [
            row for row in all_rows
            if (row.get("restart_root_input") or row["input"]) == root_input
        ]
        attempt = _next_attempt(related)
        restart_mult = rows[unfinished_index]["intended_multiplicity"]
        restart_stem = f"{Path(root_input).stem}__restart{attempt:02d}-from-m{restart_mult}"
        new_input_name = restart_stem + source_input.suffix
        new_input = campaign / "inputs" / new_input_name
        archived_output = batch / f"{stem}__before-restart{attempt:02d}.log"
        seed_name = f"{Path(rows[unfinished_index]['checkpoint']).stem}-seed-r{attempt:02d}.chk"
        seed_checkpoint = batch / seed_name
        for target in (new_input, archived_output, seed_checkpoint, batch / new_input_name):
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"restart target already exists: {target}")
        seed_sha = _sha256(source_checkpoint)
        remaining = rows[unfinished_index:]
        new_sections: list[str] = []
        new_rows: list[dict[str, str]] = []
        prior_checkpoint = seed_name
        prior_job = ""
        lineage = ""
        for local_stage, (old_row, section) in enumerate(zip(remaining, sections[unfinished_index:])):
            multiplicity = int(old_row["intended_multiplicity"])
            checkpoint = f"{Path(old_row['checkpoint']).stem}-r{attempt:02d}.chk"
            new_sections.append(_rewrite_stage(section, oldchk=prior_checkpoint, chk=checkpoint))
            job_id = f"{old_row['job_id']}-r{attempt:02d}"
            lineage = (
                f"restart:{seed_sha}:m{multiplicity}:{checkpoint}" if local_stage == 0
                else f"{lineage}>m{multiplicity}:{checkpoint}"
            )
            row = {column: old_row.get(column, "") for column in fields}
            row.update({
                "job_id": job_id,
                "chain_id": f"{old_row['chain_id']}-r{attempt:02d}",
                "stage_index": str(local_stage),
                "pathway": "multiplicity_ladder_restart",
                "initialization": (
                    "interrupted_checkpoint_restart" if local_stage == 0 else "checkpoint_spin_flip"
                ),
                "audit_classification": (
                    "checkpoint_restart" if local_stage == 0 else "sequential_checkpoint_spin_flip"
                ),
                "high_spin_multiplicity": remaining[0]["intended_multiplicity"],
                "spin_flip_index": str(local_stage),
                "predecessor_job_id": prior_job,
                "predecessor_multiplicity": (
                    "" if local_stage == 0 else remaining[local_stage - 1]["intended_multiplicity"]
                ),
                "predecessor_checkpoint": "" if local_stage == 0 else prior_checkpoint,
                "checkpoint": checkpoint,
                "checkpoint_lineage": lineage,
                "input": f"inputs/{new_input_name}",
                "output": f"{restart_stem}.log",
                "submission_active": "true",
                "restart_attempt": str(attempt),
                "restart_root_input": root_input,
                "restart_of_job_id": old_row["job_id"],
                "restart_source_output": str(archived_output.relative_to(campaign)),
                "restart_source_checkpoint": str(source_checkpoint.relative_to(campaign)),
                "restart_seed_checkpoint": str(seed_checkpoint.relative_to(campaign)),
                "restart_seed_sha256": seed_sha,
            })
            new_rows.append(row)
            prior_checkpoint = checkpoint
            prior_job = job_id
        new_text = "\n\n--Link1--\n".join(new_sections) + "\n"
        input_sha = hashlib.sha256(new_text.encode()).hexdigest()
        for row in new_rows:
            row["input_sha256"] = input_sha
        plan_row = {
            "attempt": str(attempt), "batch": batch.name, "original_input": root_input,
            "archived_input": input_path,
            "archived_output": str(archived_output.relative_to(campaign)),
            "restart_charge": remaining[0]["intended_charge"],
            "restart_multiplicity": remaining[0]["intended_multiplicity"],
            "completed_stages_in_attempt": str(unfinished_index),
            "remaining_stages": str(len(remaining)),
            "source_checkpoint": str(source_checkpoint.relative_to(campaign)),
            "seed_checkpoint": str(seed_checkpoint.relative_to(campaign)),
            "new_input": f"inputs/{new_input_name}",
            "new_output": str((batch / f"{restart_stem}.log").relative_to(campaign)),
        }
        actions.append(RestartAction(
            input_path, input_name, batch, source_input, source_output, archived_output,
            source_checkpoint, seed_checkpoint, new_input, new_input_name, new_text,
            rows, new_rows, plan_row,
        ))

    if not actions and not dry_run:
        reasons = defaultdict(int)
        for row in skipped:
            reasons[row["reason"]] += 1
        detail = ", ".join(f"{key}={value}" for key, value in sorted(reasons.items()))
        raise RuntimeError(f"no interrupted checkpoint-backed inputs to restart ({detail})")
    if dry_run:
        return {"campaign": str(campaign), "input_count": len(actions),
                "stage_count": sum(len(action.new_rows) for action in actions),
                "plan": [action.plan_row for action in actions], "skipped": skipped,
                "dry_run": True}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_dir = campaign / "restart_backups"
    backup_dir.mkdir(exist_ok=True)
    shutil.copy2(manifest, backup_dir / f"spin_jobs.{stamp}.csv")
    campaign_metadata = campaign / "spin_campaign.json"
    if campaign_metadata.is_file():
        shutil.copy2(campaign_metadata, backup_dir / f"spin_campaign.{stamp}.json")
    touched_batches = sorted({action.batch for action in actions})
    for batch in touched_batches:
        shutil.copy2(batch / "inputs.txt", backup_dir / f"{batch.name}.inputs.{stamp}.txt")

    for action in actions:
        action.new_input.write_text(action.new_text, encoding="utf-8")
        shutil.copy2(action.source_checkpoint, action.seed_checkpoint)
        if _sha256(action.seed_checkpoint) != action.new_rows[0]["restart_seed_sha256"]:
            raise RuntimeError(f"checkpoint changed while copied: {action.source_checkpoint}")
        action.source_output.rename(action.archived_output)
        old_stem = Path(action.input_name).stem
        archived_stem = action.archived_output.stem
        for suffix in _MARKER_SUFFIXES:
            marker = action.batch / f"{old_stem}{suffix}"
            if marker.exists():
                marker.rename(action.batch / f"{archived_stem}{suffix}")
        link = action.batch / action.new_input_name
        try:
            link.symlink_to(os.path.relpath(action.new_input, action.batch))
        except OSError:
            shutil.copy2(action.new_input, link)

    for batch in touched_batches:
        replacements = {
            action.input_name: action.new_input_name for action in actions if action.batch == batch
        }
        listing = batch / "inputs.txt"
        names = listing.read_text(errors="replace").splitlines()
        listing.write_text(
            "\n".join(replacements.get(name, name) for name in names) + "\n",
            encoding="utf-8",
        )
    for action in actions:
        archived_name = action.archived_output.name
        for row in action.old_rows:
            row["submission_active"] = "false"
            row["output"] = archived_name
        all_rows.extend(action.new_rows)
    _write_csv(manifest, fields, all_rows)

    existing_plan: list[dict[str, str]] = []
    plan_path = campaign / "restart_plan.csv"
    if plan_path.is_file():
        _, existing_plan = _read_csv(plan_path)
    plan_rows = existing_plan + [action.plan_row for action in actions]
    _write_csv(plan_path, list(plan_rows[0]), plan_rows)
    _write_csv(campaign / "skipped_restarts.csv", ["input", "batch", "reason"], skipped)
    if campaign_metadata.is_file():
        metadata = json.loads(campaign_metadata.read_text(encoding="utf-8"))
        metadata["manifest_sha256"] = _sha256(manifest)
        history = metadata.setdefault("restart_history", [])
        history.append({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "batch_range": [start, end], "input_count": len(actions),
            "stage_count": sum(len(action.new_rows) for action in actions),
        })
        campaign_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"campaign": str(campaign), "input_count": len(actions),
            "stage_count": sum(len(action.new_rows) for action in actions),
            "plan": [action.plan_row for action in actions], "skipped": skipped,
            "dry_run": False, "backup": str(backup_dir)}
