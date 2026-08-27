from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import replace
from pathlib import Path

from .basis import render_gen_basis
from .models import Atom, Record


_LEGACY_SCF = "SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)"
_LEGACY_IOP = "IOP(5/13=1,5/36=1,8/11=1)"

# Keep the historical Gaussian 09-era BPW91 protocol used for the source
# calculations.  The unperturbed records may be reoptimized and checked by a
# frequency calculation, whereas rattled records must retain their displaced
# geometry so that they contribute useful nonzero forces.
#
# The basis is Gen (see basis.py): 6-31G* for light elements, an explicit
# def2-TZVP-without-f contraction for Fe. Every route below reads it from
# the input body rather than naming a basis keyword.
DEFAULT_ROUTE = (
    f"#p UBPW91/Gen {_LEGACY_SCF} NoSymm Opt Freq {_LEGACY_IOP} Int=UltraFine"
)
DEFAULT_RATTLE_ROUTE = (
    f"#p UBPW91/Gen SP {_LEGACY_SCF} NoSymm {_LEGACY_IOP} Int=UltraFine"
)
DEFAULT_LINK1_ROUTE = (
    f"#p UBPW91/Gen Force {_LEGACY_SCF} NoSymm "
    f"Guess=Read Geom=Checkpoint {_LEGACY_IOP} Int=UltraFine"
)


def _geometry_sha256(record: Record) -> str:
    from .models import geometry_signature

    return hashlib.sha256(geometry_signature(record.atoms).encode()).hexdigest()


def _rattle(record: Record, sigma: float, seed: int, variant: int) -> Record:
    seed_text = f"{seed}|{record.record_id}|rattle|{variant}|{sigma:.12g}"
    resolved_seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
    rng = random.Random(resolved_seed)
    atoms = [
        Atom(a.symbol, a.x + rng.gauss(0, sigma), a.y + rng.gauss(0, sigma), a.z + rng.gauss(0, sigma))
        for a in record.atoms
    ]
    suffix = hashlib.sha1(seed_text.encode()).hexdigest()[:8]
    return Record(
        record_id=f"{record.record_id}-r{variant:02d}-{suffix}",
        source=record.source,
        atoms=atoms,
        charge=record.charge,
        multiplicity=record.multiplicity,
        config_type=f"{record.config_type}_rattled",
        route=record.route,
        legacy_energy_hartree=record.legacy_energy_hartree,
        imaginary_frequencies=record.imaginary_frequencies,
        irc_path=record.irc_path,
        irc_point=record.irc_point,
        electronic_state=record.electronic_state,
        metadata={
            **record.metadata,
            "parent_record_id": record.record_id,
            "source_record_id": record.record_id,
            "parent_geometry_sha256": _geometry_sha256(record),
            "campaign_seed": str(seed),
            "rattle_index": str(variant),
            "rattle_sigma": str(sigma),
            "resolved_rattle_seed": str(resolved_seed),
            "variant": f"r{variant:02d}",
        },
    )


def expanded_records(records: list[Record], rattles_per_seed: int, sigma: float, seed: int) -> list[Record]:
    expanded: list[Record] = []
    for record in records:
        expanded.append(
            replace(
                record,
                metadata={
                    **record.metadata,
                    "source_record_id": record.record_id,
                    "campaign_seed": str(seed),
                    "variant": "reference",
                },
            )
        )
        for variant in range(1, rattles_per_seed + 1):
            expanded.append(_rattle(record, sigma, seed, variant))
    return expanded


def _slug(value: str, limit: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return (cleaned or "unknown")[:limit].rstrip("-")


def human_job_stem(record: Record) -> str:
    """Readable, collision-safe filename stem without replacing machine identity."""
    source = _slug(Path(record.source).stem, 64)
    config_type = _slug(record.config_type.removesuffix("_rattled"), 32)
    variant = _slug(record.metadata.get("variant", "reference"), 16)
    charge = "0" if record.charge == 0 else f"{record.charge:+d}"
    state = f"q{charge}-m{record.multiplicity}"
    identity = hashlib.sha1(record.record_id.encode()).hexdigest()[:10]
    return "__".join(
        (source, _slug(record.formula, 32), config_type, state, variant, identity)
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_gaussian_jobs(
    records: list[Record],
    output: Path,
    route: str = DEFAULT_ROUTE,
    memory: str = "16GB",
    nproc: int = 16,
    rattle_route: str = DEFAULT_RATTLE_ROUTE,
    link1_route: str = DEFAULT_LINK1_ROUTE,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    protected_suffixes = {".log", ".out", ".chk", ".status", ".rc", ".started", ".finished"}
    protected = [path for path in output.rglob("*") if path.is_file() and path.suffix.lower() in protected_suffixes]
    if protected:
        preview = ", ".join(str(path.relative_to(output)) for path in protected[:5])
        raise RuntimeError(
            "refusing to regenerate a campaign that already has calculation state; "
            f"use a fresh output directory or archive the existing campaign first: {preview}"
        )
    manifest = output / "jobs.csv"
    rows: list[dict[str, object]] = []
    used_stems: set[str] = set()
    for record in records:
        stem = human_job_stem(record)
        if stem in used_stems:
            raise ValueError(f"human-readable job filename collision: {stem}")
        used_stems.add(stem)
        filename = f"{stem}.gjf"
        path = output / filename
        parent = record.metadata.get("parent_record_id", record.record_id)
        first_route = rattle_route if "rattle_index" in record.metadata else route
        gen_basis = render_gen_basis({a.symbol for a in record.atoms})
        lines = [
            f"%chk={stem}.chk",
            f"%mem={memory}",
            f"%nprocshared={nproc}",
            first_route,
            "",
            (
                f"MLIP label human_id={stem}; job_id={record.record_id}; "
                f"parent={parent}; source={record.source}; type={record.config_type}"
            ),
            "",
            f"{record.charge} {record.multiplicity}",
        ]
        lines.extend(f"{a.symbol:3s} {a.x: .12f} {a.y: .12f} {a.z: .12f}" for a in record.atoms)
        lines.append(gen_basis.rstrip("\n"))
        lines.extend(
            [
                "",
                "--Link1--",
                f"%chk={stem}.chk",
                f"%mem={memory}",
                f"%nprocshared={nproc}",
                link1_route,
                "",
                f"MLIP diffuse-basis force label human_id={stem}; job_id={record.record_id}",
                "",
                f"{record.charge} {record.multiplicity}",
                gen_basis.rstrip("\n"),
                "",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        rows.append(
            {
                "human_id": stem,
                "job_id": record.record_id,
                "source_record_id": record.metadata.get("source_record_id", parent),
                "parent_record_id": parent,
                "source": record.source,
                "source_basename": Path(record.source).name,
                "config_type": record.config_type,
                "formula": record.formula,
                "n_atoms": len(record.atoms),
                "charge": record.charge,
                "multiplicity": record.multiplicity,
                "variant": record.metadata.get("variant", "reference"),
                "rattle_index": record.metadata.get("rattle_index", ""),
                "rattle_sigma_angstrom": record.metadata.get("rattle_sigma", ""),
                "campaign_seed": record.metadata.get("campaign_seed", ""),
                "resolved_rattle_seed": record.metadata.get("resolved_rattle_seed", ""),
                "parent_geometry_sha256": record.metadata.get(
                    "parent_geometry_sha256", _geometry_sha256(record)
                ),
                "input_geometry_sha256": _geometry_sha256(record),
                "legacy_energy_hartree": (
                    "" if record.legacy_energy_hartree is None else record.legacy_energy_hartree
                ),
                "legacy_route": record.route,
                "state_inference": record.metadata.get("state_inference", ""),
                "first_route": first_route,
                "link1_route": link1_route,
                "input": filename,
                "input_sha256": _file_sha256(path),
                "output": f"{stem}.log",
            }
        )

    with manifest.open("w", newline="", encoding="utf-8") as table:
        columns = list(rows[0]) if rows else ["human_id", "job_id", "source"]
        writer = csv.DictWriter(table, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    campaign = {
        "schema_version": 1,
        "job_count": len(rows),
        "source_record_count": len({str(row["source_record_id"]) for row in rows}),
        "campaign_seeds": sorted({str(row["campaign_seed"]) for row in rows}),
        "memory": memory,
        "nprocshared": nproc,
        "seed_route": route,
        "rattle_route": rattle_route,
        "link1_route": link1_route,
        "jobs_csv_sha256": _file_sha256(manifest),
    }
    (output / "campaign_manifest.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    runner = output / "run_one.sh"
    runner.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ninput=$1\noutput=${input%.gjf}.log\ng16 \"$input\" > \"$output\"\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
