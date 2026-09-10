"""Campaign-wide route-intent audit: which jobs searched for the wrong
stationary point, and which of those already produced results we must discard.

``campaign-status --audit`` answers "did this batch run to completion". That
question cannot see the failure this module exists for: a transition state
launched with a plain ``Opt`` completes normally, reports a converged geometry
and a clean final force frame, and is wrong anyway. This audit reads every job
in the campaign -- unstarted, running, and long finished alike -- compares its
structural label against the search its route actually requests, and marks the
finished ones whose results have to be thrown away.

Read-only: nothing here modifies inputs, checkpoints, logs, or manifests.
``relaunch.py`` performs the remediation this reports.
"""
from __future__ import annotations

import collections
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gaussian import gaussian_job_complete
from .routes import FINDINGS, config_type_from_stem, geometry_source, inspect_job

AUDIT_COLUMNS = [
    "input", "batch", "job_ids", "config_type", "config_type_source", "intent",
    "search_kinds", "executed_search_kinds", "state", "expected_imaginary_modes",
    "final_imaginary_modes", "severity", "must_relaunch", "findings",
    # How a relaunch would have to rebuild this job: an input carrying its own
    # coordinates is corrected in place, whereas a spin-ladder restart seeded
    # by Geom=Checkpoint has to be rebuilt from its root ladder.
    "geometry_source", "restart_root_input", "output", "route",
]


def find_manifest(campaign: Path) -> Path:
    for name in ("jobs.csv", "spin_jobs.csv"):
        candidate = campaign / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no jobs.csv or spin_jobs.csv in {campaign}")


def read_manifest(manifest: Path) -> tuple[list[str], list[dict[str, str]]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def row_is_active(row: dict[str, str]) -> bool:
    return (row.get("submission_active") or "").strip().lower() not in {"false", "0", "no"}


def batch_locations(campaign: Path) -> dict[str, Path]:
    """Map each listed input filename to the batch directory that runs it.

    Tolerant by design: unlike the restart path, an audit must still report on
    a campaign whose batch listings are inconsistent rather than refuse to look.
    """
    locations: dict[str, Path] = {}
    for batch in sorted((campaign / "slurm_batches").glob("batch_*")):
        listing = batch / "inputs.txt"
        if not listing.is_file():
            continue
        for name in listing.read_text(errors="replace").splitlines():
            name = name.strip()
            if name:
                locations.setdefault(name, batch)
    return locations


def resolve_config_type(row: dict[str, str], input_name: str) -> tuple[str, str]:
    """The job's structural label, and where we got it.

    The manifest column is authoritative where it exists. Spin campaigns
    prepared before the manifest carried ``config_type`` still encode it in the
    generated filename, which is the only label those older campaigns have --
    without this fallback every archived spin campaign would audit as
    unconstrained and its mislaunched saddles would stay invisible.
    """
    value = (row.get("config_type") or "").strip()
    if value:
        return value, "manifest"
    inferred = config_type_from_stem(input_name)
    return inferred, "filename" if inferred else "unavailable"


def _job_state(output: Path, input_text: str) -> tuple[str, str]:
    if not output.is_file():
        return "not_started", ""
    text = output.read_text(errors="replace")
    stages = 1 + sum(
        1 for line in input_text.splitlines() if line.strip().lower() == "--link1--"
    )
    if gaussian_job_complete(text, stages):
        return "complete", text
    if "Error termination" in text:
        return "failed", text
    return "incomplete", text


def _candidate_outputs(campaign: Path, input_path: Path, batch: Path | None,
                       declared: str) -> list[Path]:
    stem = input_path.stem
    names = [declared] if declared else []
    names.append(f"{stem}.log")
    names.append(f"{stem}.out")
    candidates: list[Path] = []
    for name in names:
        relative = Path(name)
        if batch is not None:
            candidates.append(batch / relative.name)
        candidates.append(input_path.parent / relative.name)
        candidates.append(campaign / relative)
        candidates.append(campaign / "slurm_outputs" / relative.name)
    return candidates


def audit_campaign_routes(
    campaign: Path,
    destination: Path | None = None,
    *,
    include_inactive: bool = False,
) -> dict[str, Any]:
    """Audit every job's route intent; write CSV/JSON/Markdown reports."""
    campaign = campaign.resolve()
    manifest = find_manifest(campaign)
    _, manifest_rows = read_manifest(manifest)
    locations = batch_locations(campaign)

    grouped: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in manifest_rows:
        if not include_inactive and not row_is_active(row):
            continue
        reference = (row.get("input") or "").strip()
        if reference:
            grouped[reference].append(row)

    rows: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for reference, group in sorted(grouped.items()):
        input_path = campaign / reference
        input_name = input_path.name
        batch = locations.get(input_name)
        if not input_path.is_file():
            unreadable.append({"input": reference, "reason": "input file is missing"})
            continue
        input_text = input_path.read_text(errors="replace")
        config_type, label_source = resolve_config_type(group[0], input_name)
        if label_source == "unavailable":
            unreadable.append({
                "input": reference,
                "reason": "no config_type in the manifest and none encoded in the filename",
            })
            continue
        declared_output = (group[0].get("output") or "").strip()
        output = next(
            (candidate for candidate in _candidate_outputs(
                campaign, input_path, batch, declared_output) if candidate.is_file()),
            None,
        )
        state, output_text = _job_state(output, input_text) if output else ("not_started", "")
        verdict = inspect_job(config_type, input_text, output_text)
        optimizing = verdict["optimizing_routes"]
        assert isinstance(optimizing, list)
        findings = verdict["findings"]
        assert isinstance(findings, list)
        rows.append({
            "input": reference,
            "batch": batch.name if batch else "",
            "geometry_source": geometry_source(input_text),
            "restart_root_input": (group[0].get("restart_root_input") or "").strip(),
            "job_ids": ";".join(row.get("job_id", "") for row in group),
            "config_type": config_type,
            "config_type_source": label_source,
            "intent": verdict["intent"],
            "search_kinds": verdict["search_kinds"],
            "executed_search_kinds": verdict["executed_search_kinds"],
            "state": state,
            "expected_imaginary_modes": _expected(config_type),
            "final_imaginary_modes": (
                "" if verdict["final_imaginary_modes"] is None
                else verdict["final_imaginary_modes"]
            ),
            "severity": verdict["severity"],
            "must_relaunch": "true" if verdict["must_relaunch"] else "false",
            "findings": ";".join(findings),
            "output": str(output.relative_to(campaign)) if output else "",
            "route": optimizing[0] if optimizing else "",
        })

    finding_counts = collections.Counter(
        code for row in rows for code in row["findings"].split(";") if code
    )
    relaunch_rows = [row for row in rows if row["must_relaunch"] == "true"]
    wasted = collections.Counter(row["state"] for row in relaunch_rows)
    rebuild = collections.Counter(row["geometry_source"] for row in relaunch_rows)
    summary: dict[str, Any] = {
        "campaign": str(campaign),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest.name,
        "inputs_audited": len(rows),
        "by_intent": dict(collections.Counter(row["intent"] for row in rows)),
        "by_severity": dict(collections.Counter(row["severity"] for row in rows)),
        "findings": dict(finding_counts),
        "must_relaunch": len(relaunch_rows),
        "must_relaunch_by_state": dict(wasted),
        "must_relaunch_by_geometry_source": dict(rebuild),
        "completed_but_invalid": wasted.get("complete", 0),
        "unreadable": unreadable,
    }

    destination = (destination or campaign / "monitoring").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "route_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (destination / "route_relaunch_candidates.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(relaunch_rows)
    (destination / "route_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "route_audit.md").write_text(
        _report(campaign, summary, rows, relaunch_rows), encoding="utf-8"
    )
    return {
        "summary": summary,
        "rows": rows,
        "relaunch": relaunch_rows,
        "unreadable": unreadable,
        "destination": destination,
    }


def _expected(config_type: str) -> str:
    from .routes import expected_imaginary_modes

    value = expected_imaginary_modes(config_type)
    return "" if value is None else str(value)


def _report(campaign: Path, summary: dict[str, Any], rows: list[dict[str, Any]],
            relaunch: list[dict[str, Any]]) -> str:
    lines = [
        "# Route-intent audit",
        "",
        f"- Campaign: `{campaign}`",
        f"- Generated: {summary['generated_utc']}",
        f"- Inputs audited: {summary['inputs_audited']}",
        f"- Jobs whose results are invalid and must be relaunched: {summary['must_relaunch']}",
        f"- Of those, already finished (wasted allocation): {summary['completed_but_invalid']}",
        "",
        "## Findings",
        "",
    ]
    if summary["findings"]:
        lines += ["| Finding | Severity | Jobs | Meaning |", "|---|---|---:|---|"]
        for code, count in sorted(summary["findings"].items(), key=lambda item: -item[1]):
            severity, _, explanation = FINDINGS[code]
            lines.append(f"| `{code}` | {severity} | {count} | {explanation} |")
    else:
        lines.append("Every job's route matches its structural label.")
    lines += ["", "## Jobs to relaunch", ""]
    if relaunch:
        sources = summary["must_relaunch_by_geometry_source"]
        lines += [
            "How each one has to be rebuilt:",
            "",
            f"- `input_coordinates` ({sources.get('input_coordinates', 0)}): the input "
            "carries its own geometry, so its routes are corrected in place and its "
            "spin-flip chain is preserved stage by stage.",
            f"- `checkpoint` ({sources.get('checkpoint', 0)}): a spin-ladder restart with "
            "no coordinates of its own, seeded by `Geom=Checkpoint` from the run being "
            "discarded. The whole restart lineage is retired and the root ladder is "
            "rebuilt from its real coordinates, restoring the complete pathway.",
            f"- `unknown` ({sources.get('unknown', 0)}): skipped; needs a look by hand.",
            "",
            "| Input | Batch | Label | Route search | State | Imag | Rebuild from | Findings |",
            "|---|---|---|---|---|---:|---|---|",
        ]
        for row in relaunch:
            lines.append(
                f"| `{row['input']}` | {row['batch']} | {row['config_type']} | "
                f"{row['search_kinds'] or 'none'} | {row['state']} | "
                f"{row['final_imaginary_modes']} | {row['geometry_source']} | "
                f"{row['findings']} |"
            )
        lines += [
            "",
            "Relaunch these with `cluster-mlip relaunch-routes CAMPAIGN --dry-run` and then",
            "without `--dry-run`. Each replacement rebuilds the corrected saddle search from",
            "the *original input geometry*: the collapsed minimum an incorrect plain-Opt run",
            "converged to is not a transition-state guess and is never used as the restart",
            "geometry.",
        ]
    else:
        lines.append("None.")
    if summary["unreadable"]:
        lines += ["", "## Not audited", "", "| Input | Reason |", "|---|---|"]
        for item in summary["unreadable"]:
            lines.append(f"| `{item['input']}` | {item['reason']} |")
    return "\n".join(lines) + "\n"
