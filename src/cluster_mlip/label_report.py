from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Callable, TypedDict

from .dataset import split_coverage
from .gaussian import rms_force
from .models import LabeledFrame
from .stratify import STRATA_FIELDS, classify_record, strata_value


class GroupStats(TypedDict):
    group: str
    n_frames: int
    energy_ev_min: float
    energy_ev_mean: float
    energy_ev_max: float
    force_rms_ev_ang_mean: float
    force_rms_ev_ang_median: float
    force_rms_ev_ang_max: float


class OutlierFrame(TypedDict):
    record_id: str
    source: str
    group: str
    force_rms_ev_ang: float


class LabelSummary(TypedDict):
    n_frames: int
    n_groups: int
    force_outlier_threshold_ev_ang: float
    groups: list[GroupStats]
    outliers: list[OutlierFrame]
    by_stratum: dict[str, list[GroupStats]]
    split_coverage: list[dict[str, object]]


def _group_key(frame: LabeledFrame) -> str:
    rec = frame.record
    return f"q={rec.charge}, mult={rec.multiplicity}"


def _stats_by_key(frames: list[LabeledFrame], key_fn: Callable[[LabeledFrame], str]) -> list[GroupStats]:
    groups: dict[str, list[LabeledFrame]] = {}
    for frame in frames:
        groups.setdefault(key_fn(frame), []).append(frame)
    rows: list[GroupStats] = []
    for key in sorted(groups):
        group = groups[key]
        energies = [f.energy_ev for f in group]
        force_rms = [rms_force(f) for f in group]
        rows.append(
            {
                "group": key,
                "n_frames": len(group),
                "energy_ev_min": min(energies),
                "energy_ev_mean": statistics.mean(energies),
                "energy_ev_max": max(energies),
                "force_rms_ev_ang_mean": statistics.mean(force_rms),
                "force_rms_ev_ang_median": statistics.median(force_rms),
                "force_rms_ev_ang_max": max(force_rms),
            }
        )
    return rows


def summarize_labels(
    frames: list[LabeledFrame],
    force_outlier_threshold_ev_ang: float = 5.0,
    splits: dict[str, list[LabeledFrame]] | None = None,
    stratify_by: tuple[str, ...] = STRATA_FIELDS,
) -> LabelSummary:
    """Per charge/multiplicity and per-stratum energy/force statistics, plus
    outlier frames and (when `splits` is given) split coverage.

    This is a first, concrete piece of the README's "Validation priorities"
    section: "energy and force errors by charge and multiplicity" needs a
    trained model (see `cluster-mlip evaluate`), but *label* quality can be
    checked the moment `collect` finishes -- a non-converged SCF root or a
    blown-up rattled geometry shows up as an outsized force norm, and
    stratify.classify_record's other axes (whether a frame is a minimum vs.
    a saddle vs. a rattled displacement, an under-coordinated atom, a
    fragmenting cluster) surface the same way a bad charge/multiplicity
    assignment would: pooled into one number until you break it out.
    """
    group_stats = _stats_by_key(frames, _group_key)

    def _stratum_key_fn(field: str) -> Callable[[LabeledFrame], str]:
        def key_fn(frame: LabeledFrame) -> str:
            return strata_value(classify_record(frame.record), field)
        return key_fn

    by_stratum: dict[str, list[GroupStats]] = {}
    for field in stratify_by:
        by_stratum[field] = _stats_by_key(frames, _stratum_key_fn(field))

    outliers: list[OutlierFrame] = [
        {
            "record_id": frame.record.record_id,
            "source": frame.record.source,
            "group": _group_key(frame),
            "force_rms_ev_ang": rms_force(frame),
        }
        for frame in frames
        if rms_force(frame) > force_outlier_threshold_ev_ang
    ]
    outliers.sort(key=lambda row: row["force_rms_ev_ang"], reverse=True)

    return {
        "n_frames": len(frames),
        "n_groups": len(group_stats),
        "force_outlier_threshold_ev_ang": force_outlier_threshold_ev_ang,
        "groups": group_stats,
        "outliers": outliers,
        "by_stratum": by_stratum,
        "split_coverage": split_coverage(splits, stratify_by) if splits else [],
    }


def write_label_report(
    frames: list[LabeledFrame],
    output: Path,
    force_outlier_threshold_ev_ang: float = 5.0,
    splits: dict[str, list[LabeledFrame]] | None = None,
    stratify_by: tuple[str, ...] = STRATA_FIELDS,
) -> LabelSummary:
    summary = summarize_labels(frames, force_outlier_threshold_ev_ang, splits, stratify_by)
    output.mkdir(parents=True, exist_ok=True)
    (output / "label_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def stats_table(rows: list[GroupStats]) -> list[str]:
        table = [
            "| Group | Frames | Energy min (eV) | Energy mean (eV) | Energy max (eV) "
            "| Force RMS mean | Force RMS median | Force RMS max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            table.append(
                f"| {row['group']} | {row['n_frames']} | {row['energy_ev_min']:.4f} | "
                f"{row['energy_ev_mean']:.4f} | {row['energy_ev_max']:.4f} | "
                f"{row['force_rms_ev_ang_mean']:.4f} | {row['force_rms_ev_ang_median']:.4f} | "
                f"{row['force_rms_ev_ang_max']:.4f} |"
            )
        return table

    lines = [
        "# Label quality report",
        "",
        f"- Frames: {summary['n_frames']}",
        f"- Charge/multiplicity groups: {summary['n_groups']}",
        f"- Force-RMS outlier threshold: {force_outlier_threshold_ev_ang} eV/Angstrom",
        f"- Outlier frames: {len(summary['outliers'])}",
        "",
        "## Per charge/multiplicity group",
        "",
        *stats_table(summary["groups"]),
    ]

    for field in stratify_by:
        rows = summary["by_stratum"].get(field, [])
        if not rows:
            continue
        lines += ["", f"## By {field}", "", *stats_table(rows)]

    if summary["split_coverage"]:
        lines += [
            "",
            "## Split coverage (groups per stratum, not frames)",
            "",
            "A stratum with 0 in valid or test did not get enough groups to round "
            "up to one at the requested split fractions -- see dataset.grouped_split.",
            "",
            "| Stratum | Train | Valid | Test |",
            "|---|---:|---:|---:|",
        ]
        for row in summary["split_coverage"]:
            lines.append(f"| {row['stratum']} | {row['train']} | {row['valid']} | {row['test']} |")

    if summary["outliers"]:
        lines += [
            "",
            "## Force-RMS outliers (likely a non-converged SCF root or a blown-up rattle)",
            "",
            "| record_id | source | group | force RMS (eV/Angstrom) |",
            "|---|---|---|---:|",
        ]
        for outlier_row in summary["outliers"][:200]:
            lines.append(
                f"| {outlier_row['record_id']} | {outlier_row['source']} | {outlier_row['group']} | "
                f"{outlier_row['force_rms_ev_ang']:.4f} |"
            )
        if len(summary["outliers"]) > 200:
            lines.append(f"\n... and {len(summary['outliers']) - 200} more; see label_report.json.")
    (output / "label_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
