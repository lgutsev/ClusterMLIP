from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .io import quote_extxyz
from .models import LabeledFrame


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


def grouped_split(
    frames: list[LabeledFrame], valid_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[LabeledFrame]]:
    result: dict[str, list[LabeledFrame]] = {"train": [], "valid": [], "test": []}
    for frame in frames:
        group = frame.record.metadata.get("parent_record_id", frame.record.record_id)
        digest = hashlib.sha256(f"{seed}|{group}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        if value < test_fraction:
            result["test"].append(frame)
        elif value < test_fraction + valid_fraction:
            result["valid"].append(frame)
        else:
            result["train"].append(frame)
    return result


def read_jobs_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["job_id"]: row for row in csv.DictReader(handle)}
