from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import TypedDict

from .gaussian import rms_force
from .models import LabeledFrame


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


def _group_key(frame: LabeledFrame) -> str:
    rec = frame.record
    return f"q={rec.charge}, mult={rec.multiplicity}"


def summarize_labels(
    frames: list[LabeledFrame], force_outlier_threshold_ev_ang: float = 5.0
) -> LabelSummary:
    """Per charge/multiplicity energy/force statistics plus outlier frames.

    This is a first, concrete piece of the README's "Validation priorities"
    section: "energy and force errors by charge and multiplicity" needs a
    trained model (see `cluster-mlip evaluate`), but *label* quality can be
    checked the moment `collect` finishes -- a non-converged SCF root or a
    blown-up rattled geometry shows up as an outsized force norm, and today
    nothing between `collect` and `train_from_scratch.sh` looks for that.
    """
    groups: dict[str, list[LabeledFrame]] = {}
    for frame in frames:
        groups.setdefault(_group_key(frame), []).append(frame)

    group_stats: list[GroupStats] = []
    for key in sorted(groups):
        group = groups[key]
        energies = [f.energy_ev for f in group]
        force_rms = [rms_force(f) for f in group]
        group_stats.append(
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
        "n_groups": len(groups),
        "force_outlier_threshold_ev_ang": force_outlier_threshold_ev_ang,
        "groups": group_stats,
        "outliers": outliers,
    }


def write_label_report(
    frames: list[LabeledFrame], output: Path, force_outlier_threshold_ev_ang: float = 5.0
) -> LabelSummary:
    summary = summarize_labels(frames, force_outlier_threshold_ev_ang)
    output.mkdir(parents=True, exist_ok=True)
    (output / "label_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

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
        "| Group | Frames | Energy min (eV) | Energy mean (eV) | Energy max (eV) "
        "| Force RMS mean | Force RMS median | Force RMS max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            f"| {row['group']} | {row['n_frames']} | {row['energy_ev_min']:.4f} | "
            f"{row['energy_ev_mean']:.4f} | {row['energy_ev_max']:.4f} | "
            f"{row['force_rms_ev_ang_mean']:.4f} | {row['force_rms_ev_ang_median']:.4f} | "
            f"{row['force_rms_ev_ang_max']:.4f} |"
        )
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
