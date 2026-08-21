from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path
from typing import TypedDict

from .gaussian import HARTREE_TO_EV, parse_final_force_frame
from .models import LabeledFrame, geometry_signature


class ProgressResult(TypedDict):
    rows: list[dict[str, object]]
    summary: dict[str, object]
    csv_path: Path
    summary_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_optional(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(errors="ignore").strip()


def _one(index: dict[str, list[Path]], name: str) -> tuple[Path | None, bool]:
    matches = index.get(name, [])
    return (matches[0] if matches else None, len(matches) > 1)


def write_campaign_progress(campaign: Path, destination: Path | None = None) -> ProgressResult:
    campaign = campaign.resolve()
    manifest = campaign / "jobs.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"jobs.csv not found under {campaign}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle))
    if not jobs:
        raise ValueError(f"{manifest} contains no jobs")

    file_index: dict[str, list[Path]] = {}
    for path in campaign.rglob("*"):
        if path.is_file() and not path.is_symlink():
            file_index.setdefault(path.name, []).append(path)

    rows: list[dict[str, object]] = []
    for job in jobs:
        output_name = job.get("output") or f"{job['job_id']}.log"
        stem = Path(output_name).stem
        output, duplicate_output = _one(file_index, output_name)
        status_file, _ = _one(file_index, f"{stem}.status")
        started_file, _ = _one(file_index, f"{stem}.started")
        finished_file, _ = _one(file_index, f"{stem}.finished")
        rc_file, _ = _one(file_index, f"{stem}.rc")
        status_text = _read_optional(status_file)
        started = _read_optional(started_file)
        finished = _read_optional(finished_file)
        rc = _read_optional(rc_file)

        normal = False
        frame: LabeledFrame | None = None
        parse_error = ""
        if output is not None:
            text = output.read_text(errors="ignore")
            normal = "Normal termination of Gaussian" in text
            try:
                frame = parse_final_force_frame(text, output)
            except Exception as exc:  # reporting must survive one corrupt output
                parse_error = str(exc)

        if duplicate_output:
            state = "ambiguous_duplicate_output"
        elif normal and frame is not None:
            state = "complete"
        elif normal:
            state = "terminated_no_force"
        elif status_text.startswith("ERROR") or (rc and rc != "0"):
            state = "failed"
        elif started and not finished:
            state = "running"
        elif output is not None:
            state = "incomplete"
        else:
            state = "pending"

        new_energy_hartree: float | str = ""
        output_geometry_sha256 = ""
        if frame is not None:
            new_energy_hartree = frame.energy_ev / HARTREE_TO_EV
            output_geometry_sha256 = hashlib.sha256(
                geometry_signature(frame.record.atoms).encode()
            ).hexdigest()
        legacy_raw = job.get("legacy_energy_hartree", "")
        raw_delta: float | str = ""
        if (
            legacy_raw not in (None, "")
            and new_energy_hartree != ""
            and job.get("variant") == "reference"
        ):
            raw_delta = float(new_energy_hartree) - float(legacy_raw)

        batch = ""
        if output is not None and output.parent.name.startswith("batch_"):
            batch = output.parent.name
        row: dict[str, object] = dict(job)
        row.update(
            {
                "campaign_state": state,
                "batch": batch,
                "status_detail": status_text,
                "return_code": rc,
                "started": started,
                "finished": finished,
                "normal_termination": normal,
                "force_frame_parsed": frame is not None,
                "parse_error": parse_error,
                "output_path": "" if output is None else str(output.relative_to(campaign)),
                "output_sha256": "" if output is None else _sha256(output),
                "output_geometry_sha256": output_geometry_sha256,
                "new_label_energy_hartree": new_energy_hartree,
                "raw_new_minus_legacy_hartree": raw_delta,
            }
        )
        rows.append(row)

    counts = collections.Counter(str(row["campaign_state"]) for row in rows)
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        source = str(row.get("source", ""))
        source_counts = by_source.setdefault(source, {})
        state = str(row["campaign_state"])
        source_counts[state] = source_counts.get(state, 0) + 1
        source_counts["total"] = source_counts.get("total", 0) + 1
    summary: dict[str, object] = {
        "campaign": str(campaign),
        "jobs_csv_sha256": _sha256(manifest),
        "total": len(rows),
        "by_state": dict(sorted(counts.items())),
        "by_source": dict(sorted(by_source.items())),
    }

    csv_path = (destination or (campaign / "progress.csv")).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = csv_path.with_name(f"{csv_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary, "csv_path": csv_path, "summary_path": summary_path}
