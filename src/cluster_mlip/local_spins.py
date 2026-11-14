"""Atom-resolved (local) spin provenance for collected force frames.

`gaussian.parse_force_frames` attaches the Mulliken charges and spin
densities computed from each frame's own SCF density. This module turns them
into the per-frame summary `collect` stores with the label, and the long
per-atom table it writes next to the dataset, so the magnetic state a label
was computed in survives into training data.

The total multiplicity alone does not identify that state: the same 2S+1 is
reached by ferro- and ferrimagnetic Fe arrangements whose energies and forces
differ. Mulliken moments are a basis-dependent partition, so they are kept as
provenance and diagnostics, not as training targets.
"""
from __future__ import annotations

from .models import Record

# Same threshold as spin.fe_spin_summary: below it an Fe moment is "weak".
RESOLVED_MOMENT = 0.25
# Mulliken spin densities sum to N(alpha) - N(beta) = 2S exactly. A larger
# gap means the table was attached to the wrong charge/multiplicity section.
SPIN_SUM_TOLERANCE = 0.05

LOCAL_SPIN_COLUMNS = [
    "frame_id", "parent_record_id", "job_id", "gaussian_output", "stage_index",
    "force_frame_index", "charge", "multiplicity", "atom_index", "symbol",
    "mulliken_charge", "mulliken_spin", "atomic_spin_association",
]


def atomic_spin_values(record: Record) -> list[float] | None:
    """Per-atom Mulliken spins in atom order, or None if absent or mismatched."""
    rows = record.metadata.get("atomic_spins")
    if not rows or len(rows) != len(record.atoms):
        return None
    if [str(row[1]) for row in rows] != [atom.symbol for atom in record.atoms]:
        return None
    return [float(row[2]) for row in rows]


def atomic_charge_values(record: Record) -> list[float] | None:
    rows = record.metadata.get("atomic_charges")
    if not rows or len(rows) != len(record.atoms):
        return None
    if [str(row[1]) for row in rows] != [atom.symbol for atom in record.atoms]:
        return None
    return [float(row[2]) for row in rows]


def _fe_pattern(fe_spins: list[float]) -> str:
    resolved = [spin for spin in fe_spins if abs(spin) >= RESOLVED_MOMENT]
    if not resolved:
        return "weak_or_unresolved"
    if any(spin > 0 for spin in resolved) and any(spin < 0 for spin in resolved):
        return "ferrimagnetic_like"
    if len(resolved) < len(fe_spins):
        return "ferro_like_with_weak_sites"
    return "ferro_like"


def local_spin_summary(record: Record) -> dict[str, object]:
    """Per-frame magnetic provenance derived from the frame's own spin table."""
    spins = atomic_spin_values(record)
    expected_2s = record.multiplicity - 1
    spin = expected_2s / 2
    summary: dict[str, object] = {
        "local_spin_available": spins is not None,
        "s2_expected": spin * (spin + 1),
    }
    for key in ("s2_before", "s2_after"):
        if isinstance(record.metadata.get(key), (int, float)):
            summary[f"{key}_excess"] = round(record.metadata[key] - spin * (spin + 1), 6)
    if spins is None:
        return summary
    total = sum(spins)
    fe = [value for value, atom in zip(spins, record.atoms) if atom.symbol == "Fe"]
    summary.update({
        "mulliken_spin_sum": round(total, 6),
        "mulliken_abs_spin_sum": round(sum(abs(value) for value in spins), 6),
        "mulliken_spin_sum_error": round(total - expected_2s, 6),
        "local_spin_consistent": abs(total - expected_2s) <= SPIN_SUM_TOLERANCE,
    })
    if fe:
        summary.update({
            "fe_count": len(fe),
            "fe_spin_sum": round(sum(fe), 6),
            "non_fe_spin_sum": round(total - sum(fe), 6),
            "fe_spin_min": round(min(fe), 6),
            "fe_spin_max": round(max(fe), 6),
            "fe_spin_mean": round(sum(fe) / len(fe), 6),
            # Moments pointing against the net magnetization.
            "fe_minority_count": sum(
                1 for value in fe
                if abs(value) >= RESOLVED_MOMENT and (value < 0) == (expected_2s >= 0)
            ),
            "fe_weak_count": sum(1 for value in fe if abs(value) < RESOLVED_MOMENT),
            "fe_moment_pattern": _fe_pattern(fe),
        })
    return summary


def local_spin_rows(frame_id: str, record: Record) -> list[dict[str, object]]:
    """One row per atom for local_spins.csv; empty when the frame has no table."""
    spins = atomic_spin_values(record)
    if spins is None:
        return []
    known_charges = atomic_charge_values(record)
    charges: list[float | None] = (
        list(known_charges) if known_charges else [None] * len(spins)
    )
    metadata = record.metadata
    return [
        {
            "frame_id": frame_id,
            "parent_record_id": metadata.get("parent_record_id", ""),
            "job_id": metadata.get("job_id", ""),
            "gaussian_output": metadata.get("gaussian_output", ""),
            "stage_index": metadata.get("stage_index", ""),
            "force_frame_index": metadata.get("force_frame_index", ""),
            "charge": record.charge,
            "multiplicity": record.multiplicity,
            "atom_index": index,
            "symbol": atom.symbol,
            "mulliken_charge": "" if charge is None else f"{charge:.6f}",
            "mulliken_spin": f"{spin:.6f}",
            "atomic_spin_association": metadata.get("atomic_spin_association", ""),
        }
        for index, (atom, spin, charge) in enumerate(zip(record.atoms, spins, charges), start=1)
    ]
