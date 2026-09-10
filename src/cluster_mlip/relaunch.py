"""Relaunch jobs whose route searched for the wrong stationary point.

The correctness rule that shapes this whole module: a transition state that was
launched with a plain ``Opt`` walked downhill and converged to a *minimum*. That
converged geometry, and every checkpoint written from it, is not a transition
state guess -- it is the wrong answer. So unlike ``restart.py``, which resumes
an interrupted ladder from its own checkpoint, a route relaunch always rebuilds
from the **original input geometry** and writes to **fresh checkpoint names**,
leaving the poisoned outputs and checkpoints archived and untouched beside it.

Everything happens in place, in the campaign's existing batch folders, so the
saved Slurm batch map stays valid and the campaign is resubmitted with the same
head launcher. Nothing is ever deleted.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .route_audit import audit_campaign_routes, find_manifest, read_manifest
from .routes import (
    corrected_minimum_route,
    corrected_saddle_route,
    route_optimizes,
    stage_route,
)

_LINK1_SPLIT_RE = re.compile(r"(^\s*--\s*link1\s*--\s*$)", re.IGNORECASE | re.MULTILINE)
_CHK_RE = re.compile(r"^(\s*%(?:old)?chk\s*=\s*)(\S+)\s*$", re.IGNORECASE | re.MULTILINE)
_MARKER_SUFFIXES = (".status", ".rc", ".started", ".finished")

RELAUNCH_COLUMNS = [
    "submission_active", "relaunch_attempt", "relaunch_reason", "relaunch_root_input",
    "relaunch_of_job_id", "relaunch_source_output", "relaunch_route_before",
    "relaunch_route_after",
    # Stamped on the *superseded* rows so the archived log is permanently
    # self-describing: `collect` refuses those labels on this column alone,
    # without having to re-derive the verdict from files that may move.
    "route_invalidated", "superseded_by_job_id",
]

PLAN_COLUMNS = [
    "attempt", "batch", "original_input", "new_input", "config_type", "findings",
    "previous_state", "archived_output", "new_output", "route_before", "route_after",
    "checkpoints_renamed", "preserved_body_sha256",
]


@dataclass
class RelaunchAction:
    audit_row: dict[str, Any]
    batch: Path | None
    source_input: Path
    new_input: Path
    new_text: str
    source_output: Path | None
    archived_output: Path | None
    old_rows: list[dict[str, str]]
    new_rows: list[dict[str, str]]
    plan_row: dict[str, str]
    renamed_checkpoints: list[str] = field(default_factory=list)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _attempt_is_unconfirmed(directory: Path, stem: str) -> bool:
    started = directory / f"{stem}.started"
    finished = directory / f"{stem}.finished"
    return started.is_file() and (
        not finished.is_file() or started.stat().st_mtime > finished.stat().st_mtime
    )


def _next_attempt(rows: list[dict[str, str]]) -> int:
    values: list[int] = []
    for row in rows:
        try:
            values.append(int(row.get("relaunch_attempt", "") or 0))
        except ValueError:
            pass
    return max(values, default=0) + 1


def _rewrite_routes(text: str, intent: str, order: int) -> tuple[str, str, str]:
    """Correct every optimizing route in a multi-stage input.

    Returns the rewritten text plus the before/after of the first optimizing
    route, for the plan and manifest record. Non-optimizing stages -- the
    ``Force`` label stage of a generated job -- are left exactly as they were.
    """
    pieces = _LINK1_SPLIT_RE.split(text)
    before = after = ""
    for index, piece in enumerate(pieces):
        if _LINK1_SPLIT_RE.fullmatch(piece):
            continue
        route = stage_route(piece)
        if not route or not route_optimizes(route):
            continue
        corrected = (
            corrected_saddle_route(route, order) if intent == "saddle"
            else corrected_minimum_route(route)
        )
        if corrected == route:
            continue
        if not before:
            before, after = route, corrected
        pieces[index] = _substitute_route(piece, route, corrected)
    return "".join(pieces), before, after


def _substitute_route(section: str, route: str, corrected: str) -> str:
    """Replace a stage's route block with a single corrected route line.

    The route may have been wrapped over several continuation lines; all of
    them are consumed so the corrected route is not appended to a fragment of
    the original.
    """
    lines = section.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.strip().startswith("#")), None)
    if start is None:
        return section
    end = start + 1
    while end < len(lines) and lines[end].strip():
        end += 1
    newline = "\n" if lines[start].endswith("\n") else ""
    return "".join(lines[:start]) + corrected + newline + "".join(lines[end:])


def _rename_checkpoints(text: str, suffix: str) -> tuple[str, list[str]]:
    """Point every %chk/%oldchk at a fresh name.

    The old checkpoints hold the wrong stationary point. Renaming keeps them on
    disk for inspection while guaranteeing the relaunch neither reads nor
    overwrites them, and it preserves each stage's lineage inside the job
    because every reference is renamed the same way.
    """
    renamed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        value = Path(match.group(2))
        new_name = f"{value.stem}{suffix}{value.suffix or '.chk'}"
        renamed.append(f"{value.name}->{new_name}")
        return f"{match.group(1)}{value.with_name(new_name).as_posix()}"

    return _CHK_RE.sub(replace, text), renamed


def _body_digest(text: str) -> str:
    """Hash everything a route rewrite must not touch.

    Excludes route lines and ``%`` link directives; what remains is the title,
    the charge/multiplicity lines, the coordinates and any basis block. Compared
    before against after, this is a by-construction proof that the relaunch
    reuses the original guess geometry rather than anything derived from the
    discarded run.
    """
    body = [
        line.rstrip()
        for line in text.splitlines()
        if not line.lstrip().startswith(("#", "%"))
    ]
    return hashlib.sha256("\n".join(body).encode()).hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare_route_relaunch(
    campaign: Path,
    *,
    start: int = 1,
    end: int | None = None,
    assume_stopped: bool = False,
    dry_run: bool = False,
    saddle_order: int | None = None,
) -> dict[str, Any]:
    """Audit routes, then rebuild and activate corrected inputs in place."""
    campaign = campaign.resolve()
    manifest = find_manifest(campaign)
    audit = audit_campaign_routes(campaign)
    candidates = audit["relaunch"]
    assert isinstance(candidates, list)

    original_fields, all_rows = read_manifest(manifest)
    fields = list(dict.fromkeys(original_fields + RELAUNCH_COLUMNS))
    rows_by_input: dict[str, list[dict[str, str]]] = {}
    for row in all_rows:
        rows_by_input.setdefault((row.get("input") or "").strip(), []).append(row)

    batch_root = campaign / "slurm_batches"
    numbers = [
        int(path.name[6:]) for path in batch_root.glob("batch_*")
        if path.is_dir() and path.name[6:].isdigit()
    ]
    final_batch = max(numbers, default=0)
    end = final_batch if end is None else end
    if final_batch and not 1 <= start <= end <= final_batch:
        raise ValueError(f"batch range must satisfy 1 <= start <= end <= {final_batch}")

    actions: list[RelaunchAction] = []
    skipped: list[dict[str, str]] = []

    def skip(row: dict[str, Any], reason: str) -> None:
        skipped.append({"input": row["input"], "batch": row["batch"], "reason": reason})

    for row in candidates:
        reference = row["input"]
        batch = batch_root / row["batch"] if row["batch"] else None
        if batch is not None:
            number = int(batch.name[6:])
            if not start <= number <= end:
                continue
        elif final_batch:
            skip(row, "input is not listed in any batch inputs.txt")
            continue
        if row["config_type"].removesuffix("_rattled") == "higher_order_saddle" \
                and saddle_order is None:
            skip(row, "higher_order_saddle needs an explicit --saddle-order")
            continue
        source_input = campaign / reference
        stem = source_input.stem
        directory = batch if batch is not None else source_input.parent
        if _attempt_is_unconfirmed(directory, stem) and not assume_stopped:
            skip(row, "activity_unconfirmed: pass --assume-stopped once squeue is clear")
            continue

        group = rows_by_input.get(reference, [])
        if not group:
            skip(row, "input is absent from the campaign manifest")
            continue
        root_input = group[0].get("relaunch_root_input") or reference
        related = [
            item for item in all_rows
            if (item.get("relaunch_root_input") or item.get("input", "")) == root_input
        ]
        attempt = _next_attempt(related)
        suffix = f"-routefix{attempt:02d}"
        new_stem = f"{Path(root_input).stem}__routefix{attempt:02d}"
        new_input = source_input.with_name(new_stem + source_input.suffix)

        text = source_input.read_text(encoding="utf-8", errors="replace")
        intent = row["intent"]
        order = saddle_order or 1
        rewritten, before, after = _rewrite_routes(text, intent, order)
        if not before:
            skip(row, (
                "route already requests the right search, so a rewrite would change "
                f"nothing ({row['findings']}); this needs a better guess geometry, "
                "not a relaunch"
            ))
            continue
        rewritten, renamed = _rename_checkpoints(rewritten, suffix)
        if _body_digest(rewritten) != _body_digest(text):
            raise RuntimeError(
                f"{reference}: rewriting the route changed the molecular specification; "
                "refusing to relaunch from a geometry this tool altered"
            )

        source_output = campaign / row["output"] if row["output"] else None
        archived_output = None
        if source_output is not None and source_output.is_file():
            archived_output = source_output.with_name(
                f"{source_output.stem}__before-routefix{attempt:02d}{source_output.suffix}"
            )
        targets = [new_input] + ([archived_output] if archived_output else [])
        if batch is not None:
            targets.append(batch / new_input.name)
        existing = [target for target in targets if target.exists() or target.is_symlink()]
        if existing:
            skip(row, f"relaunch target already exists: {existing[0].name}")
            continue

        new_rows: list[dict[str, str]] = []
        for old_row in group:
            new_row = {column: old_row.get(column, "") for column in fields}
            new_row.update({
                "input": str(Path(reference).with_name(new_input.name).as_posix()),
                "output": f"{new_stem}.log",
                # Replaced with the digest of the file as written; see below.
                "input_sha256": _sha256_text(rewritten),
                "submission_active": "true",
                "relaunch_attempt": str(attempt),
                "relaunch_reason": row["findings"],
                "relaunch_root_input": root_input,
                "relaunch_of_job_id": old_row.get("job_id", ""),
                "relaunch_source_output": (
                    str(archived_output.relative_to(campaign)) if archived_output else ""
                ),
                "relaunch_route_before": before,
                "relaunch_route_after": after,
                "route_invalidated": "",
                "superseded_by_job_id": "",
            })
            if old_row.get("job_id"):
                new_row["job_id"] = f"{old_row['job_id']}-rf{attempt:02d}"
            if "first_route" in new_row and new_row["first_route"]:
                new_row["first_route"] = after
            new_rows.append(new_row)

        plan_row = {
            "attempt": str(attempt),
            "batch": batch.name if batch else "",
            "original_input": reference,
            "new_input": str(Path(reference).with_name(new_input.name).as_posix()),
            "config_type": row["config_type"],
            "findings": row["findings"],
            "previous_state": row["state"],
            "archived_output": (
                str(archived_output.relative_to(campaign)) if archived_output else ""
            ),
            "new_output": f"{new_stem}.log",
            "route_before": before,
            "route_after": after,
            "checkpoints_renamed": ";".join(renamed),
            "preserved_body_sha256": _body_digest(rewritten),
        }
        actions.append(RelaunchAction(
            audit_row=row, batch=batch, source_input=source_input, new_input=new_input,
            new_text=rewritten, source_output=source_output, archived_output=archived_output,
            old_rows=group, new_rows=new_rows, plan_row=plan_row, renamed_checkpoints=renamed,
        ))

    result: dict[str, Any] = {
        "campaign": str(campaign),
        "audit_summary": audit["summary"],
        "input_count": len(actions),
        "job_row_count": sum(len(action.new_rows) for action in actions),
        "plan": [action.plan_row for action in actions],
        "skipped": skipped,
        "dry_run": dry_run,
        "report": str(audit["destination"]),
    }
    if dry_run:
        return result
    if not actions:
        raise RuntimeError(
            "no jobs need a route relaunch"
            + (f" ({len(skipped)} skipped; see the audit report)" if skipped else "")
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backups = campaign / "route_fix_backups"
    backups.mkdir(exist_ok=True)
    shutil.copy2(manifest, backups / f"{manifest.stem}.{stamp}.csv")
    touched = sorted({action.batch for action in actions if action.batch is not None})
    for batch in touched:
        shutil.copy2(batch / "inputs.txt", backups / f"{batch.name}.inputs.{stamp}.txt")

    for action in actions:
        action.new_input.parent.mkdir(parents=True, exist_ok=True)
        action.new_input.write_text(action.new_text, encoding="utf-8")
        written = _sha256_file(action.new_input)
        for row in action.new_rows:
            row["input_sha256"] = written
        if action.source_output is not None and action.archived_output is not None:
            action.source_output.rename(action.archived_output)
            for suffix in _MARKER_SUFFIXES:
                marker = action.source_output.with_suffix(suffix)
                if marker.exists():
                    marker.rename(action.archived_output.with_suffix(suffix))
        if action.batch is not None:
            link = action.batch / action.new_input.name
            try:
                link.symlink_to(os.path.relpath(action.new_input, action.batch))
            except OSError:
                # Unprivileged symlink creation is disabled by default on
                # Windows; a copy behaves identically for the batch worker.
                shutil.copy2(action.new_input, link)

    for batch in touched:
        replacements = {
            action.source_input.name: action.new_input.name
            for action in actions if action.batch == batch
        }
        listing = batch / "inputs.txt"
        names = listing.read_text(errors="replace").splitlines()
        listing.write_text(
            "\n".join(replacements.get(name.strip(), name) for name in names if name.strip())
            + "\n",
            encoding="utf-8",
        )

    for action in actions:
        replacement_by_job = {
            row.get("relaunch_of_job_id", ""): row.get("job_id", "")
            for row in action.new_rows
        }
        for old_row in action.old_rows:
            old_row["submission_active"] = "false"
            old_row["route_invalidated"] = action.audit_row["findings"]
            old_row["superseded_by_job_id"] = replacement_by_job.get(
                old_row.get("job_id", ""), ""
            )
            if action.archived_output is not None:
                old_row["output"] = action.archived_output.name
        all_rows.extend(action.new_rows)
    _write_csv(manifest, fields, all_rows)

    plan_path = campaign / "route_fix_plan.csv"
    existing_plan: list[dict[str, str]] = []
    if plan_path.is_file():
        _, existing_plan = read_manifest(plan_path)
    _write_csv(plan_path, PLAN_COLUMNS,
               existing_plan + [action.plan_row for action in actions])
    _write_csv(campaign / "skipped_route_fixes.csv",
               ["input", "batch", "reason"], skipped)

    for name in ("spin_campaign.json", "campaign_manifest.json"):
        metadata_path = campaign / name
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "manifest_sha256" in metadata:
            metadata["manifest_sha256"] = _sha256_file(manifest)
        if "jobs_csv_sha256" in metadata:
            metadata["jobs_csv_sha256"] = _sha256_file(manifest)
        history = metadata.setdefault("route_fix_history", [])
        history.append({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "batch_range": [start, end],
            "input_count": len(actions),
            "job_row_count": sum(len(action.new_rows) for action in actions),
        })
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    result["backup"] = str(backups)
    return result
