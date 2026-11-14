from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, TypedDict

from .mace_glue import load_calculator, predict_forces_and_energy, require_mace
from .models import LabeledFrame
from .stratify import STRATA_FIELDS, classify_record, strata_value

# (energy_ev, forces_ev_ang) aligned index-for-index with a list of frames.
Prediction = tuple[float, list[tuple[float, float, float]]]


class GroupError(TypedDict):
    group: str
    n_frames: int
    energy_mae_ev_per_atom: float
    energy_rmse_ev_per_atom: float
    force_mae_ev_ang: float
    force_rmse_ev_ang: float


class EvaluationSummary(TypedDict):
    n_frames: int
    overall: GroupError
    by_charge_multiplicity: list[GroupError]
    by_stratum: dict[str, list[GroupError]]


def _group_key(frame: LabeledFrame) -> str:
    rec = frame.record
    return f"q={rec.charge}, mult={rec.multiplicity}"


def _group_errors(
    pairs: list[tuple[LabeledFrame, Prediction]],
    group_key: Callable[[LabeledFrame], str],
) -> list[GroupError]:
    groups: dict[str, list[tuple[LabeledFrame, Prediction]]] = {}
    for frame, prediction in pairs:
        groups.setdefault(group_key(frame), []).append((frame, prediction))

    rows: list[GroupError] = []
    for key in sorted(groups):
        group = groups[key]
        energy_abs_err_per_atom: list[float] = []
        force_abs_err: list[float] = []
        for frame, (pred_energy, pred_forces) in group:
            n_atoms = len(frame.record.atoms)
            energy_abs_err_per_atom.append(abs(pred_energy - frame.energy_ev) / n_atoms)
            if len(pred_forces) != len(frame.forces_ev_ang):
                raise ValueError(
                    f"prediction/reference force-array length mismatch for {frame.record.record_id}"
                )
            for (rx, ry, rz), (px, py, pz) in zip(frame.forces_ev_ang, pred_forces):
                force_abs_err.extend((abs(rx - px), abs(ry - py), abs(rz - pz)))
        rows.append(
            {
                "group": key,
                "n_frames": len(group),
                "energy_mae_ev_per_atom": statistics.mean(energy_abs_err_per_atom),
                "energy_rmse_ev_per_atom": math.sqrt(
                    sum(e * e for e in energy_abs_err_per_atom) / len(energy_abs_err_per_atom)
                ),
                "force_mae_ev_ang": statistics.mean(force_abs_err),
                "force_rmse_ev_ang": math.sqrt(sum(e * e for e in force_abs_err) / len(force_abs_err)),
            }
        )
    return rows


def _stratum_key_fn(field: str) -> Callable[[LabeledFrame], str]:
    def key_fn(frame: LabeledFrame) -> str:
        return strata_value(classify_record(frame.record), field)
    return key_fn


def summarize_evaluation(
    frames: list[LabeledFrame],
    predictions: list[Prediction],
    stratify_by: tuple[str, ...] = STRATA_FIELDS,
) -> EvaluationSummary:
    """Implements the README's "energy and force errors by charge and
    multiplicity" validation priority, plus a breakdown across every
    stratify.classify_record axis (pes_region, displacement_class,
    coordination_class, compactness_class, charge_spin_class,
    provenance_tier) -- so a model that's fine on average but bad at, say,
    saddle points or under-coordinated atoms doesn't hide behind one number.

    `predictions` must be aligned index-for-index with `frames`. This
    function itself has no model-backend dependency -- it takes whatever
    (energy, forces) pairs you hand it, so it's fully testable without MACE
    or torch installed. `predict_with_mace` below is the one piece that
    needs the optional training stack.

    charge_spin_class is a coarser view of the same charge/multiplicity axis
    already reported in by_charge_multiplicity; both are kept since the finer
    one is still useful once by_stratum's grouping has enough per-bucket
    frames to be meaningful. Isomer ordering, held-out reaction-family/
    cluster-size splits, and barrier errors from the same README section
    still need reaction-family/isomer-group labels and TS/IRC pairing this
    pipeline does not track yet, and are still future work.
    """
    if len(frames) != len(predictions):
        raise ValueError("frames and predictions must be the same length")
    pairs = list(zip(frames, predictions))
    overall = _group_errors(pairs, lambda _frame: "overall")[0]
    by_group = _group_errors(pairs, _group_key)
    by_stratum = {field: _group_errors(pairs, _stratum_key_fn(field)) for field in stratify_by}
    return {
        "n_frames": len(frames),
        "overall": overall,
        "by_charge_multiplicity": by_group,
        "by_stratum": by_stratum,
    }


def _error_table(rows: list[GroupError]) -> list[str]:
    lines = [
        "| Group | Frames | Energy MAE (eV/atom) | Energy RMSE (eV/atom) "
        "| Force MAE (eV/Ang) | Force RMSE (eV/Ang) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['n_frames']} | {row['energy_mae_ev_per_atom']:.4f} | "
            f"{row['energy_rmse_ev_per_atom']:.4f} | {row['force_mae_ev_ang']:.4f} | "
            f"{row['force_rmse_ev_ang']:.4f} |"
        )
    return lines


PREDICTION_COLUMNS = [
    "frame_id", "split_group", "parent_record_id", "job_id", "stage_frame_position",
    "charge", "multiplicity", "n_atoms", "ref_energy_ev", "pred_energy_ev", "energy_error_ev",
    "energy_error_ev_per_atom", "force_mae_ev_ang", "force_rmse_ev_ang",
    "force_max_abs_error_ev_ang", "ref_force_rms_ev_ang", "fe_moment_pattern",
    "mulliken_spin_sum", "spin_pair_ids",
]
PAIR_ERROR_COLUMNS = [
    "pair_id", "parent_frame_id", "child_frame_id", "parent_multiplicity",
    "child_multiplicity", "ref_gap_ev", "pred_gap_ev", "gap_error_ev",
]


def frame_prediction_rows(
    frames: list[LabeledFrame], predictions: list[Prediction]
) -> list[dict[str, object]]:
    """Per-frame errors with the provenance needed to see *where* a model fails."""
    rows = []
    for frame, (energy, forces) in zip(frames, predictions):
        record = frame.record
        metadata = record.metadata
        errors = [abs(r - p) for ref, pred in zip(frame.forces_ev_ang, forces)
                  for r, p in zip(ref, pred)]
        reference = [value for force in frame.forces_ev_ang for value in force]
        n_atoms = len(record.atoms)
        rows.append({
            "frame_id": record.record_id,
            "split_group": metadata.get("split_group") or metadata.get("parent_record_id", ""),
            "parent_record_id": metadata.get("parent_record_id", ""),
            "job_id": metadata.get("job_id", ""),
            "stage_frame_position": metadata.get("stage_frame_position", ""),
            "charge": record.charge,
            "multiplicity": record.multiplicity,
            "n_atoms": n_atoms,
            "ref_energy_ev": f"{frame.energy_ev:.8f}",
            "pred_energy_ev": f"{energy:.8f}",
            "energy_error_ev": f"{energy - frame.energy_ev:.8f}",
            "energy_error_ev_per_atom": f"{(energy - frame.energy_ev) / n_atoms:.8f}",
            "force_mae_ev_ang": f"{statistics.mean(errors):.6f}" if errors else "",
            "force_rmse_ev_ang": (
                f"{math.sqrt(sum(e * e for e in errors) / len(errors)):.6f}" if errors else ""
            ),
            "force_max_abs_error_ev_ang": f"{max(errors, default=0.0):.6f}",
            "ref_force_rms_ev_ang": (
                f"{math.sqrt(sum(v * v for v in reference) / len(reference)):.6f}"
                if reference else ""
            ),
            "fe_moment_pattern": metadata.get("fe_moment_pattern", ""),
            "mulliken_spin_sum": metadata.get("mulliken_spin_sum", ""),
            "spin_pair_ids": ";".join(
                str(pair.get("pair_id", "")) for pair in metadata.get("spin_pairs") or []
            ),
        })
    return rows


def spin_pair_errors(
    frames: list[LabeledFrame], predictions: list[Prediction]
) -> list[dict[str, Any]]:
    """Vertical spin-flip gaps (child minus parent, same geometry) predicted
    vs DFT, for every verified pair whose two frames were both evaluated."""
    members: dict[str, dict[str, tuple[LabeledFrame, Prediction]]] = {}
    for frame, prediction in zip(frames, predictions):
        for pair in frame.record.metadata.get("spin_pairs") or []:
            members.setdefault(str(pair["pair_id"]), {})[str(pair["role"])] = (frame, prediction)
    rows = []
    for pair_id, roles in sorted(members.items()):
        if {"high_spin_parent", "low_spin_child"} - set(roles):
            continue
        (parent, parent_pred), (child, child_pred) = (
            roles["high_spin_parent"], roles["low_spin_child"]
        )
        ref_gap = child.energy_ev - parent.energy_ev
        pred_gap = child_pred[0] - parent_pred[0]
        rows.append({
            "pair_id": pair_id,
            "parent_frame_id": parent.record.record_id,
            "child_frame_id": child.record.record_id,
            "parent_multiplicity": parent.record.multiplicity,
            "child_multiplicity": child.record.multiplicity,
            "ref_gap_ev": f"{ref_gap:.6f}",
            "pred_gap_ev": f"{pred_gap:.6f}",
            "gap_error_ev": f"{pred_gap - ref_gap:.6f}",
        })
    return rows


def write_evaluation_report(
    frames: list[LabeledFrame],
    predictions: list[Prediction],
    output: Path,
    stratify_by: tuple[str, ...] = STRATA_FIELDS,
) -> EvaluationSummary:
    summary = summarize_evaluation(frames, predictions, stratify_by)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(frame_prediction_rows(frames, predictions))
    gap_rows = spin_pair_errors(frames, predictions)
    with (output / "spin_pair_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_ERROR_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(gap_rows)
    (output / "evaluation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = [
        "group", "n_frames", "energy_mae_ev_per_atom", "energy_rmse_ev_per_atom",
        "force_mae_ev_ang", "force_rmse_ev_ang",
    ]
    with (output / "by_charge_multiplicity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary["by_charge_multiplicity"])
    for field, rows in summary["by_stratum"].items():
        with (output / f"by_{field}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    overall = summary["overall"]
    lines = [
        "# Model evaluation",
        "",
        f"- Frames: {summary['n_frames']}",
        f"- Overall energy MAE: {overall['energy_mae_ev_per_atom']:.4f} eV/atom "
        f"(RMSE {overall['energy_rmse_ev_per_atom']:.4f})",
        f"- Overall force MAE: {overall['force_mae_ev_ang']:.4f} eV/Angstrom "
        f"(RMSE {overall['force_rmse_ev_ang']:.4f})",
        "",
        "## By charge/multiplicity",
        "",
        *_error_table(summary["by_charge_multiplicity"]),
    ]
    if gap_rows:
        gap_errors = [abs(float(row["gap_error_ev"])) for row in gap_rows]
        lines += [
            "",
            "## Vertical spin-flip gaps on verified ladder pairs",
            "",
            f"- Pairs with both frames evaluated: {len(gap_rows)}",
            f"- Gap MAE: {statistics.mean(gap_errors):.4f} eV; max |error|: "
            f"{max(gap_errors):.4f} eV (per pair in spin_pair_errors.csv)",
        ]
    for field in stratify_by:
        rows = summary["by_stratum"].get(field, [])
        if not rows:
            continue
        lines += ["", f"## By {field}", "", *_error_table(rows)]
    lines += [
        "",
        "\"Energy and force errors by charge and multiplicity\" from the README's "
        "Validation priorities is computed here (charge_spin_class above is a coarser "
        "view of the same axis; by_charge_multiplicity is the finer one). Isomer "
        "ordering, held-out reaction families/cluster sizes, and barrier errors still "
        "need reaction-family/isomer-group labels and TS/IRC pairing this pipeline "
        "does not track yet.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def predict_with_mace(model_path: Path, frames: list[LabeledFrame], device: str = "cpu") -> list[Prediction]:
    """Run a trained MACE model over `frames` and return aligned predictions.

    Requires the optional training stack (see mace_glue.TRAIN_EXTRA_HINT).
    This is the one function in the module not unit-testable without a real
    checkpoint -- treat it like the rest of the spin workflow: inspect its
    output before trusting it, especially against a mace-torch version other
    than the one this was written against (>=0.3.16).
    """
    require_mace()
    calculator = load_calculator([model_path], device=device)
    return [predict_forces_and_energy(calculator, frame.record) for frame in frames]
