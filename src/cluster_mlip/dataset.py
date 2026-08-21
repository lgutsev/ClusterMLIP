from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .io import parse_extxyz_info_line, quote_extxyz
from .models import Atom, LabeledFrame, Record
from .stratify import classify_record, stratum_key


def write_labeled_extxyz(frames: list[LabeledFrame], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            rec = frame.record
            fields = [
                "Properties=species:S:1:pos:R:3:REF_forces:R:3",
                f"record_id={quote_extxyz(rec.record_id)}",
                f"source={quote_extxyz(rec.source)}",
                f"formula={quote_extxyz(rec.formula)}",
                f"config_type={quote_extxyz(rec.config_type)}",
                f"charge={rec.charge}",
                f"spin={rec.multiplicity}",
                f"multiplicity={rec.multiplicity}",
                f"total_spin={rec.total_spin}",
                f"REF_energy={frame.energy_ev:.14g}",
                'pbc="F F F"',
            ]
            parent = rec.metadata.get("parent_record_id")
            if parent:
                fields.append(f"parent_record_id={quote_extxyz(parent)}")
            if rec.metadata:
                fields.append(f"metadata={quote_extxyz(json.dumps(rec.metadata, sort_keys=True))}")
            handle.write(f"{len(rec.atoms)}\n{' '.join(fields)}\n")
            for atom, force in zip(rec.atoms, frame.forces_ev_ang):
                handle.write(
                    f"{atom.symbol:3s} {atom.x: .12f} {atom.y: .12f} {atom.z: .12f} "
                    f"{force[0]: .12f} {force[1]: .12f} {force[2]: .12f}\n"
                )


def read_labeled_extxyz(path: Path) -> list[LabeledFrame]:
    """Read a labeled extxyz written by write_labeled_extxyz (all/train/
    valid/test.extxyz from `collect`) back into LabeledFrame objects, for
    `evaluate` and any other consumer of the final training data.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[LabeledFrame] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].strip())
        info = parse_extxyz_info_line(lines[i + 1])
        atoms: list[Atom] = []
        forces: list[tuple[float, float, float]] = []
        for line in lines[i + 2:i + 2 + n_atoms]:
            parts = line.split()
            atoms.append(Atom(parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
            forces.append((float(parts[4]), float(parts[5]), float(parts[6])))
        record = Record(
            record_id=info["record_id"],
            source=info.get("source", ""),
            atoms=atoms,
            charge=int(info.get("charge", 0)),
            multiplicity=int(info.get("multiplicity", info.get("spin", 1))),
            config_type=info.get("config_type", "unknown"),
            metadata=json.loads(info["metadata"]) if "metadata" in info else {},
        )
        frames.append(LabeledFrame(record, float(info["REF_energy"]), forces, path))
        i += n_atoms + 2
    return frames


def _parent_group(frame: LabeledFrame) -> str:
    return frame.record.metadata.get("parent_record_id", frame.record.record_id)


def grouped_split(
    frames: list[LabeledFrame],
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    stratify_by: tuple[str, ...] | None = None,
) -> dict[str, list[LabeledFrame]]:
    """Split frames into train/valid/test, keeping every parent-record's
    rattled siblings in one split.

    `stratify_by=None` (default) is the original algorithm, unchanged: each
    parent-group gets one SHA-256-derived value in [0, 1) and is thresholded
    against the requested fractions. That's unbiased in expectation but has
    high variance for a *small* group of groups -- e.g. a stratum with 3
    groups and test_fraction=0.1 has roughly a 73% chance none of them land
    in "test" by chance alone, silently vanishing that stratum from
    evaluation.

    `stratify_by=("pes_region", "charge_spin_class", ...)` classifies one
    representative frame per parent-group (stratify.classify_record) and,
    *within each resulting stratum*, deterministically ranks its groups by a
    seeded hash and cuts them at the requested proportions (rounded) --
    proportional allocation by rank, not another per-group random draw, so a
    stratum's split composition no longer depends on chance. See
    split_coverage() to see the result per stratum.
    """
    result: dict[str, list[LabeledFrame]] = {"train": [], "valid": [], "test": []}
    if stratify_by is None:
        # Original algorithm, byte-for-byte: per-frame (not per-group) so
        # relative frame order within each split bucket is preserved exactly
        # as before, even though the hash value only ever depends on the
        # frame's parent group.
        for frame in frames:
            digest = hashlib.sha256(f"{seed}|{_parent_group(frame)}".encode()).digest()
            value = int.from_bytes(digest[:8], "big") / 2**64
            if value < test_fraction:
                result["test"].append(frame)
            elif value < test_fraction + valid_fraction:
                result["valid"].append(frame)
            else:
                result["train"].append(frame)
        return result

    groups: dict[str, list[LabeledFrame]] = {}
    for frame in frames:
        groups.setdefault(_parent_group(frame), []).append(frame)

    strata_groups: dict[str, list[str]] = {}
    for group_id, members in groups.items():
        representative = next(
            (member for member in members if member.record.record_id == group_id),
            members[0],
        )
        key = stratum_key(classify_record(representative.record), stratify_by)
        strata_groups.setdefault(key, []).append(group_id)

    for key, group_ids in strata_groups.items():
        ordered = sorted(
            group_ids,
            key=lambda gid: hashlib.sha256(f"{seed}|{key}|{gid}".encode()).digest(),
        )
        n = len(ordered)
        n_test = min(round(n * test_fraction), n)
        n_valid = min(round(n * valid_fraction), n - n_test)
        test_ids = set(ordered[:n_test])
        valid_ids = set(ordered[n_test:n_test + n_valid])
        for group_id in ordered:
            bucket = "test" if group_id in test_ids else "valid" if group_id in valid_ids else "train"
            result[bucket].extend(groups[group_id])
    return result


def split_coverage(
    splits: dict[str, list[LabeledFrame]], stratify_by: tuple[str, ...]
) -> list[dict[str, object]]:
    """Per-stratum *group* counts (not frame counts) in each split, so a
    stratum with zero groups in valid/test is visible instead of silently
    absent. `stratify_by` should match what was passed to grouped_split for
    the breakdown to correspond to how the split was actually made.
    """
    counts: dict[str, dict[str, set[str]]] = {}
    for split_name, members in splits.items():
        for frame in members:
            key = stratum_key(classify_record(frame.record), stratify_by)
            counts.setdefault(key, {"train": set(), "valid": set(), "test": set()})
            counts[key][split_name].add(_parent_group(frame))
    rows: list[dict[str, object]] = []
    for key in sorted(counts):
        row: dict[str, object] = {"stratum": key}
        for split_name in ("train", "valid", "test"):
            row[split_name] = len(counts[key][split_name])
        rows.append(row)
    return rows


def read_jobs_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["job_id"]: row for row in csv.DictReader(handle)}
