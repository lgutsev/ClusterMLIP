from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import TypedDict

from .io import write_extxyz
from .mace_glue import load_calculator, record_to_atoms, require_mace
from .models import Record

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
