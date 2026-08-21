from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

from .models import Atom, Record


_LEGACY_SCF = "SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)"
_LEGACY_IOP = "IOP(5/13=1,5/36=1,8/11=1)"

# Keep the historical Gaussian 09-era BPW91 protocol used for the source
# calculations.  The unperturbed records may be reoptimized and checked by a
# frequency calculation, whereas rattled records must retain their displaced
# geometry so that they contribute useful nonzero forces.
DEFAULT_ROUTE = (
    f"#p UBPW91/6-311G* {_LEGACY_SCF} NoSymm Opt Freq {_LEGACY_IOP} Int=UltraFine"
)
DEFAULT_RATTLE_ROUTE = (
    f"#p UBPW91/6-311G* SP {_LEGACY_SCF} NoSymm {_LEGACY_IOP} Int=UltraFine"
)
DEFAULT_LINK1_ROUTE = (
    f"#p UBPW91/6-311++G* Force {_LEGACY_SCF} NoSymm "
    f"Guess=Read Geom=Checkpoint {_LEGACY_IOP} Int=UltraFine"
)


def _rattle(record: Record, sigma: float, rng: random.Random, variant: int) -> Record:
    atoms = [
        Atom(a.symbol, a.x + rng.gauss(0, sigma), a.y + rng.gauss(0, sigma), a.z + rng.gauss(0, sigma))
        for a in record.atoms
    ]
    suffix = hashlib.sha1(f"{record.record_id}|rattle|{variant}|{sigma}".encode()).hexdigest()[:8]
    return Record(
        record_id=f"{record.record_id}-r{variant:02d}-{suffix}",
        source=record.source,
        atoms=atoms,
        charge=record.charge,
        multiplicity=record.multiplicity,
        config_type=f"{record.config_type}_rattled",
        route=record.route,
        imaginary_frequencies=record.imaginary_frequencies,
        irc_path=record.irc_path,
        irc_point=record.irc_point,
        electronic_state=record.electronic_state,
        metadata={**record.metadata, "parent_record_id": record.record_id, "rattle_sigma": str(sigma)},
    )


def expanded_records(records: list[Record], rattles_per_seed: int, sigma: float, seed: int) -> list[Record]:
    rng = random.Random(seed)
    expanded: list[Record] = []
    for record in records:
        expanded.append(record)
        for variant in range(1, rattles_per_seed + 1):
            expanded.append(_rattle(record, sigma, rng, variant))
    return expanded


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
    manifest = output / "jobs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as table:
        columns = ["job_id", "parent_record_id", "source", "config_type", "formula", "charge", "multiplicity", "input", "output"]
        writer = csv.DictWriter(table, fieldnames=columns)
        writer.writeheader()
        for record in records:
            filename = f"{record.record_id}.gjf"
            path = output / filename
            parent = record.metadata.get("parent_record_id", record.record_id)
            first_route = rattle_route if "parent_record_id" in record.metadata else route
            lines = [
                f"%chk={record.record_id}.chk",
                f"%mem={memory}",
                f"%nprocshared={nproc}",
                first_route,
                "",
                f"MLIP label {record.record_id}; parent={parent}; type={record.config_type}",
                "",
                f"{record.charge} {record.multiplicity}",
            ]
            lines.extend(f"{a.symbol:3s} {a.x: .12f} {a.y: .12f} {a.z: .12f}" for a in record.atoms)
            lines.extend(
                [
                    "",
                    "--Link1--",
                    f"%chk={record.record_id}.chk",
                    f"%mem={memory}",
                    f"%nprocshared={nproc}",
                    link1_route,
                    "",
                    f"MLIP diffuse-basis force label {record.record_id}",
                    "",
                    f"{record.charge} {record.multiplicity}",
                    "",
                ]
            )
            path.write_text("\n".join(lines), encoding="utf-8")
            writer.writerow({
                "job_id": record.record_id,
                "parent_record_id": parent,
                "source": record.source,
                "config_type": record.config_type,
                "formula": record.formula,
                "charge": record.charge,
                "multiplicity": record.multiplicity,
                "input": filename,
                "output": f"{record.record_id}.log",
            })

    runner = output / "run_one.sh"
    runner.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ninput=$1\noutput=${input%.gjf}.log\ng16 \"$input\" > \"$output\"\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
