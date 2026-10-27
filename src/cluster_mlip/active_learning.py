from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import TypedDict

from .io import write_extxyz
from .mace_glue import load_calculator, record_to_atoms, require_mace
from .models import LabeledFrame, Record
from .stratify import pes_region

# One committee member's predicted forces for one structure: one (fx, fy, fz) per atom.
ForceSet = list[tuple[float, float, float]]


class RankedCandidate(TypedDict):
    record_id: str
    source: str
    formula: str
    disagreement_ev_ang: str


def committee_force_disagreement(committee_forces: list[ForceSet]) -> float:
    """Max-atom force disagreement across a committee, for one structure.

    For each atom, take the per-component standard deviation of the force
    across committee members, combine the three components into one vector
    norm, then report the worst (max) atom. This is a standard, simple
    committee-disagreement uncertainty proxy for active learning (the same
    idea used by query-by-committee / deep-ensemble MLIP selection): a
    structure where committee members strongly disagree on the force is one
    the current dataset does not constrain well, and is worth prioritizing
    for expensive DFT labeling over one the committee already agrees on.
    """
    n_models = len(committee_forces)
    if n_models < 2:
        raise ValueError("need at least two committee members to measure disagreement")
    n_atoms = len(committee_forces[0])
    if any(len(forces) != n_atoms for forces in committee_forces):
        raise ValueError("all committee members must predict the same number of atoms")
    if n_atoms == 0:
        return 0.0
    worst = 0.0
    for atom_index in range(n_atoms):
        component_stdevs = []
        for component in range(3):
            values = [committee_forces[m][atom_index][component] for m in range(n_models)]
            component_stdevs.append(statistics.pstdev(values))
        atom_score = math.sqrt(sum(s * s for s in component_stdevs))
        worst = max(worst, atom_score)
    return worst


def committee_mean_force_error(committee_forces: list[ForceSet], reference: ForceSet) -> float:
    """Worst-atom |committee-mean force - reference force| (eV/Angstrom), the
    true-error counterpart of committee_force_disagreement."""
    n_models = len(committee_forces)
    if any(len(forces) != len(reference) for forces in committee_forces):
        raise ValueError("committee and reference must have the same number of atoms")
    worst = 0.0
    for atom_index, ref in enumerate(reference):
        mean = [sum(committee_forces[m][atom_index][c] for m in range(n_models)) / n_models
                for c in range(3)]
        worst = max(worst, math.sqrt(sum((mean[c] - ref[c]) ** 2 for c in range(3))))
    return worst


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman_rank_correlation(left: list[float], right: list[float]) -> float | None:
    """Spearman's rho with average ranks for ties; None when undefined."""
    if len(left) != len(right):
        raise ValueError("rank correlation needs two equally long lists")
    if len(left) < 3:
        return None
    rank_left, rank_right = _average_ranks(left), _average_ranks(right)
    mean_left, mean_right = statistics.mean(rank_left), statistics.mean(rank_right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(rank_left, rank_right))
    spread = math.sqrt(
        sum((a - mean_left) ** 2 for a in rank_left) * sum((b - mean_right) ** 2 for b in rank_right)
    )
    return covariance / spread if spread else None


def committee_error_calibration(
    committee_forces: list[list[ForceSet]],
    reference_forces: list[ForceSet],
    top_fraction: float = 0.1,
) -> dict[str, object]:
    """Does committee disagreement find the frames the committee actually gets wrong?

    committee_forces[m][i] is member m's prediction for labeled frame i and
    reference_forces[i] its DFT forces. Compares, per frame, the worst-atom
    disagreement with the worst-atom error of the committee mean, and reports:

    - Spearman rho between the two;
    - top-error recall: of the `top_fraction` frames with the largest true
      error, the share that are also in the `top_fraction` of disagreement --
      what a selector picking that many frames would catch (chance level is
      `top_fraction`);
    - confident failures: frames in the top true-error set whose disagreement
      is at or below the median. A committee trained on the same data can
      share a mistake, and these frames are where it did.

    This calibrates the selector on frames that have DFT labels the committee
    never trained on; it does not certify accuracy of any unlabeled frame.
    """
    n_models = len(committee_forces)
    if n_models < 2:
        raise ValueError("need at least two committee members to measure disagreement")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    for m, member_forces in enumerate(committee_forces):
        if len(member_forces) != len(reference_forces):
            raise ValueError(
                f"committee member {m} predicted {len(member_forces)} frames, "
                f"expected {len(reference_forces)}"
            )
    disagreement, error = [], []
    for i, reference in enumerate(reference_forces):
        members = [committee_forces[m][i] for m in range(n_models)]
        disagreement.append(committee_force_disagreement(members))
        error.append(committee_mean_force_error(members, reference))
    n_frames = len(reference_forces)
    n_top = max(1, math.ceil(top_fraction * n_frames)) if n_frames else 0
    top_error = set(sorted(range(n_frames), key=lambda i: error[i], reverse=True)[:n_top])
    top_disagreement = set(
        sorted(range(n_frames), key=lambda i: disagreement[i], reverse=True)[:n_top]
    )
    median_disagreement = statistics.median(disagreement) if disagreement else 0.0
    return {
        "n_frames": n_frames,
        "n_models": n_models,
        "top_fraction": top_fraction,
        "n_top": n_top,
        "spearman_rho": spearman_rank_correlation(disagreement, error),
        "top_error_recall": len(top_error & top_disagreement) / n_top if n_top else None,
        "confident_failures": sorted(
            i for i in top_error if disagreement[i] <= median_disagreement
        ),
        "disagreement_ev_ang": disagreement,
        "committee_error_ev_ang": error,
    }


def rank_candidates_by_disagreement(
    candidates: list[Record], committee_forces: list[list[ForceSet]]
) -> list[tuple[Record, float]]:
    """committee_forces[m][i] is committee member m's predicted forces for
    candidates[i]. Returns (record, score) pairs sorted by descending
    disagreement -- the structures most worth labeling next come first.
    """
    n_models = len(committee_forces)
    if n_models < 2:
        raise ValueError("need at least two committee members to measure disagreement")
    for m, member_forces in enumerate(committee_forces):
        if len(member_forces) != len(candidates):
            raise ValueError(
                f"committee member {m} predicted {len(member_forces)} structures, "
                f"expected {len(candidates)}"
            )
    scored = [
        (record, committee_force_disagreement([committee_forces[m][i] for m in range(n_models)]))
        for i, record in enumerate(candidates)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def write_next_batch(
    candidates: list[Record],
    committee_forces: list[list[ForceSet]],
    output: Path,
    top_k: int,
) -> list[tuple[Record, float]]:
    ranked = rank_candidates_by_disagreement(candidates, committee_forces)
    selected = ranked[:top_k]
    output.mkdir(parents=True, exist_ok=True)
    write_extxyz([record for record, _ in selected], output / "next_batch.extxyz")
    with (output / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["record_id", "source", "formula", "disagreement_ev_ang"]
        )
        writer.writeheader()
        for record, score in selected:
            row: RankedCandidate = {
                "record_id": record.record_id,
                "source": record.source,
                "formula": record.formula,
                "disagreement_ev_ang": f"{score:.6f}",
            }
            writer.writerow(row)
    return selected


def write_committee_calibration(
    frames: list[LabeledFrame],
    committee_forces: list[list[ForceSet]],
    output: Path,
    top_fraction: float = 0.1,
) -> dict[str, object]:
    calibration = committee_error_calibration(
        committee_forces, [frame.forces_ev_ang for frame in frames], top_fraction
    )
    output.mkdir(parents=True, exist_ok=True)
    confident = set(calibration["confident_failures"])
    with (output / "calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "record_id", "formula", "charge", "multiplicity", "pes_region",
            "disagreement_ev_ang", "committee_error_ev_ang", "confident_failure",
        ], lineterminator="\n")
        writer.writeheader()
        for i, frame in enumerate(frames):
            record = frame.record
            writer.writerow({
                "record_id": record.record_id,
                "formula": record.formula,
                "charge": record.charge,
                "multiplicity": record.multiplicity,
                "pes_region": pes_region(record.config_type),
                "disagreement_ev_ang": f"{calibration['disagreement_ev_ang'][i]:.6f}",
                "committee_error_ev_ang": f"{calibration['committee_error_ev_ang'][i]:.6f}",
                "confident_failure": int(i in confident),
            })
    summary = {key: value for key, value in calibration.items()
               if key not in ("disagreement_ev_ang", "committee_error_ev_ang")}
    summary["confident_failures"] = [frames[i].record.record_id for i in calibration["confident_failures"]]
    (output / "calibration.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def predict_committee_forces(
    model_paths: list[Path], candidates: list[Record], device: str = "cpu"
) -> list[list[ForceSet]]:
    """Run each checkpoint in `model_paths` over every candidate.

    Requires the optional training stack (see mace_glue.TRAIN_EXTRA_HINT).
    Loads one calculator per checkpoint rather than MACECalculator's own
    multi-model averaging mode, because per-member forces (not just the
    ensemble mean) are what committee_force_disagreement needs.
    """
    require_mace()
    all_forces: list[list[ForceSet]] = []
    for model_path in model_paths:
        calculator = load_calculator([model_path], device=device)
        member_forces: list[ForceSet] = []
        for record in candidates:
            atoms = record_to_atoms(record)
            atoms.calc = calculator
            forces = [(float(fx), float(fy), float(fz)) for fx, fy, fz in atoms.get_forces()]
            member_forces.append(forces)
        all_forces.append(member_forces)
    return all_forces
