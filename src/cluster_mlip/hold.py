"""Hold chosen inputs out of a campaign's batches, and release them again.

The generated launcher submits whole batches: every name in a batch's
``inputs.txt`` that has not finished is run. There was no way to submit a
batch range while leaving some of its inputs alone -- duplicates of another
job, inputs waiting for a policy decision, inputs known to be broken. A hold
does exactly that and nothing more:

- the name is removed from its batch's ``inputs.txt`` and recorded in that
  batch's ``inputs.held.txt`` (name, reason, UTC time);
- ``CAMPAIGN/held_inputs.csv`` lists every input currently held, and
  ``CAMPAIGN/hold_history.csv`` logs each hold and release;
- the input file, its manifest rows and everything else stay as they are, so
  a release puts the name back and the campaign is as it was.

What is never held: an input that already produced output (holding it would
hide a result or strand a partial run -- that is the restart pass's job), an
input whose attempt is unconfirmed (a ``.started`` marker with no later
``.finished``) unless ``assume_stopped``, and the last listed input of a batch
unless ``allow_empty_batches`` -- the launcher refuses a batch range that
contains an empty ``inputs.txt``, so a fully held batch must be left out of
every submitted range.

While an input is held it is absent from ``campaign-status`` (which follows the
batch listings); ``audit-routes`` still reports it, with an empty batch.
"""
from __future__ import annotations

import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import write_text_lf

HELD_COLUMNS = ["batch", "input", "reason", "held_utc"]
HISTORY_COLUMNS = ["utc", "action", "batch", "input", "reason"]


def read_input_list(path: Path) -> list[str]:
    """Input names from a list file: one per line, first column, ``#`` comments.

    Accepts a plain list, a TSV/CSV whose first column is the input (a header
    cell ``input`` is skipped), and campaign-relative paths (``inputs/x.gjf``).
    """
    names: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        first = line.replace(",", "\t").split("\t")[0].strip().strip('"')
        if not first or first.lower() in ("input", "name"):
            continue
        name = Path(first.replace("\\", "/")).name
        if name not in names:
            names.append(name)
    return names


def _batches(campaign: Path) -> list[Path]:
    return sorted(
        (path for path in (campaign / "slurm_batches").glob("batch_*")
         if path.is_dir() and path.name[6:].isdigit()),
        key=lambda path: int(path.name[6:]),
    )


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in
            path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def _held_entries(batch: Path) -> list[list[str]]:
    return [(line.split("\t") + ["", ""])[:3] for line in _lines(batch / "inputs.held.txt")]


def _unconfirmed(batch: Path, stem: str) -> bool:
    started, finished = batch / f"{stem}.started", batch / f"{stem}.finished"
    return started.is_file() and (
        not finished.is_file() or started.stat().st_mtime > finished.stat().st_mtime
    )


def _write_campaign_tables(campaign: Path, events: list[dict[str, str]]) -> None:
    held: list[dict[str, str]] = []
    for batch in _batches(campaign):
        for name, reason, when in _held_entries(batch):
            held.append({"batch": batch.name, "input": name, "reason": reason, "held_utc": when})
    with (campaign / "held_inputs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HELD_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(held)
    history = campaign / "hold_history.csv"
    new = not history.is_file()
    with history.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerows(events)


def hold_inputs(
    campaign: Path,
    names: list[str],
    *,
    reason: str = "",
    release: bool = False,
    assume_stopped: bool = False,
    allow_empty_batches: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Hold (or, with ``release``, put back) the named inputs; see the module doc."""
    campaign = campaign.resolve()
    batches = _batches(campaign)
    if not batches:
        raise FileNotFoundError(f"no slurm_batches/batch_* under {campaign}")
    if "\t" in reason or "\n" in reason:
        raise ValueError("the reason must be a single line without tabs")
    listed: dict[str, Path] = {}
    held_in: dict[str, Path] = {}
    for batch in batches:
        for name in _lines(batch / "inputs.txt"):
            listed.setdefault(name, batch)
        for name, _, _ in _held_entries(batch):
            held_in.setdefault(name, batch)

    wanted = list(dict.fromkeys(Path(name.replace("\\", "/")).name for name in names))
    plan: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    def skip(name: str, batch: Path | None, why: str) -> None:
        skipped.append({"batch": batch.name if batch else "", "input": name, "reason": why})

    remaining = {batch: len(_lines(batch / "inputs.txt")) for batch in batches}
    for name in wanted:
        if release:
            holder = held_in.get(name)
            if holder is None:
                skip(name, listed.get(name), "not held")
                continue
            plan.append({"batch": holder.name, "input": name, "reason": reason})
            continue
        found = listed.get(name)
        if found is None:
            skip(name, held_in.get(name),
                 "already held" if name in held_in else "not listed in any batch inputs.txt")
            continue
        batch = found
        stem = Path(name).stem
        log = batch / f"{stem}.log"
        if log.is_file() and log.stat().st_size > 0:
            skip(name, batch, "already has output; holds are for inputs that have not run")
            continue
        if _unconfirmed(batch, stem) and not assume_stopped:
            skip(name, batch, "activity_unconfirmed: pass --assume-stopped once every queue "
                              "that sees the campaign is clear")
            continue
        if remaining[batch] <= 1 and not allow_empty_batches:
            skip(name, batch, "would leave the batch with no inputs, and the launcher refuses a "
                              "range containing an empty batch: pass --allow-empty-batches and "
                              "leave this batch out of submitted ranges")
            continue
        remaining[batch] -= 1
        plan.append({"batch": batch.name, "input": name, "reason": reason})

    emptied = sorted(batch.name for batch, count in remaining.items() if count == 0)
    result: dict[str, Any] = {
        "campaign": str(campaign), "action": "release" if release else "hold",
        "input_count": len(plan), "plan": plan, "skipped": skipped,
        "batches_touched": len({row["batch"] for row in plan}),
        "empty_batches": emptied, "dry_run": dry_run, "backup": "",
    }
    if dry_run or not plan:
        return result

    now = datetime.now(timezone.utc)
    # Microseconds: a hold and a release can follow each other within a second.
    stamp, when = now.strftime("%Y%m%dT%H%M%S%fZ"), now.isoformat(timespec="seconds")
    backup = campaign / "hold_backups" / stamp
    backup.mkdir(parents=True)
    by_batch: dict[str, list[str]] = {}
    for row in plan:
        by_batch.setdefault(row["batch"], []).append(row["input"])
    events: list[dict[str, str]] = []
    for batch_name, batch_names in by_batch.items():
        batch = campaign / "slurm_batches" / batch_name
        listing, held_file = batch / "inputs.txt", batch / "inputs.held.txt"
        shutil.copy2(listing, backup / f"{batch_name}.inputs.txt")
        if held_file.is_file():
            shutil.copy2(held_file, backup / f"{batch_name}.inputs.held.txt")
        current, held = _lines(listing), _held_entries(batch)
        chosen = set(batch_names)
        if release:
            held = [entry for entry in held if entry[0] not in chosen]
            current += [name for name in batch_names if name not in current]
        else:
            current = [name for name in current if name not in chosen]
            held += [[name, reason, when] for name in batch_names]
        write_text_lf(listing, "".join(f"{name}\n" for name in current))
        if held:
            write_text_lf(held_file, "".join("\t".join(entry) + "\n" for entry in held))
        elif held_file.is_file():
            held_file.unlink()
        events += [{"utc": when, "action": result["action"], "batch": batch_name,
                    "input": name, "reason": reason} for name in batch_names]
    _write_campaign_tables(campaign, events)
    result["backup"] = str(backup)
    return result
