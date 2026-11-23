"""Verified M -> M-2 spin-ladder pairs, and split groups by source geometry.

A spin ladder runs one optimization per multiplicity, each Link1 stage
reading its predecessor's checkpoint (``Geom=Checkpoint Guess=Read``). When
that works, the child's first force frame sits exactly on the parent's final
geometry, and the two frames are a paired electronic-state calculation: two
spin states of the same nuclear configuration, whose energy difference is a
vertical spin-flip gap.

Job order and manifest predecessors say which pairs *should* exist; they do
not show that the geometry was actually carried over (a stale checkpoint, a
restart seeded from a different file, or a reoriented frame all break it).
So a pair is marked only when the coordinates agree within a tolerance, and
every manifest claim is reported whether or not it verifies. The search is
also run without the manifest, which finds pairs the manifest cannot name --
a restart seeded from the predecessor checkpoint starts a new chain whose
first stage records no predecessor at all.

Frames within one optimization are strongly correlated, and one warehouse
structure can appear under several record IDs, so the split unit is a group
of source records whose starting geometries coincide.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from .models import Atom, LabeledFrame
from .spin import _pair_fingerprint

DEFAULT_PAIR_TOLERANCE_ANG = 1e-4
DEFAULT_GROUP_TOLERANCE_ANG = 0.01
GROUP_TOLERANCE_PROFILE = (0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3)

PAIR_COLUMNS = [
    "pair_id", "paired", "pair_basis", "manifest_claims_pair", "pair_kind",
    "formula", "charge",
    "parent_job_id", "parent_output", "parent_multiplicity", "parent_stage_index",
    "parent_stage_completed", "parent_final_frame_index", "parent_energy_ev",
    "parent_s2_before", "parent_mulliken_spin_sum", "parent_fe_moment_pattern",
    "child_job_id", "child_output", "child_multiplicity", "child_stage_index",
    "child_stage_completed", "child_first_frame_index", "child_energy_ev",
    "child_s2_before", "child_mulliken_spin_sum", "child_fe_moment_pattern",
    "max_atom_displacement_ang", "fingerprint_rms_ang", "vertical_gap_ev",
    "parent_record_id", "child_parent_record_id",
    "parent_frame_in_dataset", "child_frame_in_dataset",
]


@dataclass
class StageEndpoint:
    """The first and final force frame of one stage in one output."""

    job_id: str
    output: str
    row: dict[str, str]
    first: LabeledFrame
    final: LabeledFrame
    completed: bool
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.output}::{self.job_id}"

    @property
    def charge(self) -> int:
        return self.final.record.charge

    @property
    def multiplicity(self) -> int:
        return self.final.record.multiplicity

    @property
    def formula(self) -> str:
        return self.final.record.formula


def max_displacement(left: list[Atom], right: list[Atom]) -> float:
    """Largest Cartesian component difference, atom for atom (no alignment)."""
    if len(left) != len(right) or any(a.symbol != b.symbol for a, b in zip(left, right)):
        return math.inf
    return max(
        (max(abs(a.x - b.x), abs(a.y - b.y), abs(a.z - b.z)) for a, b in zip(left, right)),
        default=0.0,
    )


def fingerprint_distance(left: dict[str, list[float]], right: dict[str, list[float]]) -> float:
    """RMS difference of sorted element-pair distances: rotation-,
    translation- and permutation-invariant."""
    if left.keys() != right.keys() or any(len(left[key]) != len(right[key]) for key in left):
        return math.inf
    differences = [a - b for key in left for a, b in zip(left[key], right[key])]
    if not differences:
        return 0.0
    return math.sqrt(sum(value * value for value in differences) / len(differences))


def _bucket(atoms: list[Atom], width: float) -> tuple[int, int, int]:
    first = atoms[0]
    return (math.floor(first.x / width), math.floor(first.y / width), math.floor(first.z / width))


def _metadata(frame: LabeledFrame, key: str) -> str:
    value = frame.record.metadata.get(key, "")
    return "" if value is None else str(value)


def verify_spin_pairs(
    endpoints: list[StageEndpoint],
    tolerance: float = DEFAULT_PAIR_TOLERANCE_ANG,
    dataset_frames: set[int] | None = None,
    aliases: dict[int, LabeledFrame] | None = None,
) -> list[dict[str, object]]:
    """Every claimed or geometrically found parent-final/child-first pair.

    A pair is ``paired`` only when the parent's multiplicity is the child's
    plus two and the two geometries agree atom for atom within ``tolerance``
    Angstrom. Manifest claims that fail are kept with ``paired=false`` so a
    broken hand-off is visible rather than silently absent.

    ``aliases`` maps a frame dropped as an exact duplicate to the identical
    frame kept in the dataset; pair tags go to the kept frame, once per
    distinct (parent, child) frame pair, so twin ladders are not counted twice.
    """
    aliases = aliases or {}

    def resolve(frame: LabeledFrame) -> LabeledFrame:
        return aliases.get(id(frame), frame)

    tagged: set[tuple[int, int]] = set()
    by_job: dict[str, list[StageEndpoint]] = {}
    width = max(tolerance * 10, 1e-3)
    finals: dict[tuple, list[StageEndpoint]] = {}
    for endpoint in endpoints:
        by_job.setdefault(endpoint.job_id, []).append(endpoint)
        atoms = endpoint.final.record.atoms
        if atoms:
            key = (endpoint.formula, endpoint.charge, endpoint.multiplicity, *_bucket(atoms, width))
            finals.setdefault(key, []).append(endpoint)
    fingerprints: dict[int, dict[str, list[float]]] = {}

    def fingerprint(frame: LabeledFrame) -> dict[str, list[float]]:
        if id(frame) not in fingerprints:
            fingerprints[id(frame)] = _pair_fingerprint(frame.record.atoms)
        return fingerprints[id(frame)]

    rows: list[dict[str, object]] = []
    for child in endpoints:
        atoms = child.first.record.atoms
        if not atoms:
            continue
        claimed_job = (child.row.get("predecessor_job_id") or "").strip()
        claimed = [parent for parent in by_job.get(claimed_job, []) if parent is not child]
        bx, by, bz = _bucket(atoms, width)
        found = [
            parent
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            for parent in finals.get(
                (child.formula, child.charge, child.multiplicity + 2, bx + dx, by + dy, bz + dz), []
            )
            if parent is not child
            and max_displacement(parent.final.record.atoms, atoms) <= tolerance
        ]
        seen: set[str] = set()
        for parent in claimed + found:
            if parent.key in seen:
                continue
            seen.add(parent.key)
            displacement = max_displacement(parent.final.record.atoms, atoms)
            identical = displacement <= tolerance
            ladder_step = parent.multiplicity == child.multiplicity + 2
            is_claimed = parent.job_id == claimed_job and bool(claimed_job)
            paired = identical and ladder_step
            if is_claimed and paired:
                basis = "manifest+geometry"
            elif paired:
                basis = "geometry_only"
            elif is_claimed:
                basis = "manifest_claim_not_verified"
            else:
                continue
            if parent.output == child.output:
                kind = "link1_checkpoint_handoff"
            elif (child.row.get("restart_of_job_id") or "").strip():
                kind = "restart_seeded_from_parent"
            else:
                kind = "cross_output"
            pair_id = hashlib.sha1(f"{parent.key}|{child.key}".encode()).hexdigest()[:16]
            parent_frame, child_frame = parent.final, child.first
            kept_parent, kept_child = resolve(parent_frame), resolve(child_frame)
            rows.append({
                "pair_id": pair_id,
                "paired": paired,
                "pair_basis": basis,
                "manifest_claims_pair": is_claimed,
                "pair_kind": kind,
                "formula": child.formula,
                "charge": child.charge,
                "parent_job_id": parent.job_id,
                "parent_output": parent.output,
                "parent_multiplicity": parent.multiplicity,
                "parent_stage_index": parent.row.get("stage_index", ""),
                "parent_stage_completed": parent.completed,
                "parent_final_frame_index": _metadata(parent_frame, "force_frame_index"),
                "parent_energy_ev": f"{parent_frame.energy_ev:.8f}",
                "parent_s2_before": _metadata(parent_frame, "s2_before"),
                "parent_mulliken_spin_sum": _metadata(parent_frame, "mulliken_spin_sum"),
                "parent_fe_moment_pattern": _metadata(parent_frame, "fe_moment_pattern"),
                "child_job_id": child.job_id,
                "child_output": child.output,
                "child_multiplicity": child.multiplicity,
                "child_stage_index": child.row.get("stage_index", ""),
                "child_stage_completed": child.completed,
                "child_first_frame_index": _metadata(child_frame, "force_frame_index"),
                "child_energy_ev": f"{child_frame.energy_ev:.8f}",
                "child_s2_before": _metadata(child_frame, "s2_before"),
                "child_mulliken_spin_sum": _metadata(child_frame, "mulliken_spin_sum"),
                "child_fe_moment_pattern": _metadata(child_frame, "fe_moment_pattern"),
                "max_atom_displacement_ang": (
                    "inf" if math.isinf(displacement) else f"{displacement:.3e}"
                ),
                "fingerprint_rms_ang": f"{fingerprint_distance(fingerprint(parent_frame), fingerprint(child_frame)):.3e}",
                "vertical_gap_ev": (
                    f"{child_frame.energy_ev - parent_frame.energy_ev:.8f}" if paired else ""
                ),
                "parent_record_id": _metadata(parent_frame, "parent_record_id"),
                "child_parent_record_id": _metadata(child_frame, "parent_record_id"),
                "parent_frame_in_dataset": (
                    "" if dataset_frames is None else id(kept_parent) in dataset_frames
                ),
                "child_frame_in_dataset": (
                    "" if dataset_frames is None else id(kept_child) in dataset_frames
                ),
            })
            if paired and (id(kept_parent), id(kept_child)) not in tagged:
                tagged.add((id(kept_parent), id(kept_child)))
                for frame, role, partner in (
                    (kept_parent, "high_spin_parent", child.multiplicity),
                    (kept_child, "low_spin_child", parent.multiplicity),
                ):
                    frame.record.metadata.setdefault("spin_pairs", []).append({
                        "pair_id": pair_id, "role": role, "partner_multiplicity": partner,
                    })
    return rows


class _UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def source_geometry_groups(
    geometries: dict[str, list[Atom]],
    exact_keys: dict[str, str] | None = None,
    tolerance: float = DEFAULT_GROUP_TOLERANCE_ANG,
) -> dict[str, str]:
    """Map each source record ID to a split group.

    Records are merged when they share an exact source-geometry key or when
    their starting geometries' pair-distance fingerprints differ by at most
    ``tolerance`` Angstrom RMS (single linkage). Records without a known
    geometry stay in their own group.
    """
    records = sorted(set(geometries) | set(exact_keys or {}))
    union = _UnionFind(records)
    by_exact: dict[str, str] = {}
    for record_id, key in sorted((exact_keys or {}).items()):
        if key:
            if key in by_exact:
                union.union(by_exact[key], record_id)
            else:
                by_exact[key] = record_id
    by_formula: dict[str, dict[str, dict[str, list[float]]]] = {}
    for record_id, atoms in sorted(geometries.items()):
        if atoms:
            by_formula.setdefault(_formula(atoms), {})[record_id] = _pair_fingerprint(atoms)
    for fingerprints in by_formula.values():
        ids = list(fingerprints)
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                if fingerprint_distance(fingerprints[left], fingerprints[right]) <= tolerance:
                    union.union(left, right)
    return {record_id: f"geom:{union.find(record_id)}" for record_id in records}


def group_count_profile(
    geometries: dict[str, list[Atom]],
    exact_keys: dict[str, str] | None = None,
    tolerances: tuple[float, ...] = GROUP_TOLERANCE_PROFILE,
) -> list[tuple[float, int]]:
    """How many independent groups the source geometries form at each
    tolerance -- a plateau shows where twins end and distinct isomers begin."""
    return [
        (tolerance, len(set(source_geometry_groups(geometries, exact_keys, tolerance).values())))
        for tolerance in tolerances
    ]


def _formula(atoms: list[Atom]) -> str:
    counts: dict[str, int] = {}
    for atom in atoms:
        counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
    return "".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))
