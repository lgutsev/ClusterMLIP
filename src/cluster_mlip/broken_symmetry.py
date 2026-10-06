"""Fixed-geometry broken-symmetry (BS) pilot inputs for Fe clusters.

Each job holds geometry, charge and multiplicity fixed and asks whether the SCF
converges to a different stable solution when started from a different site
pattern:

  stage 0  SP Stable=Opt Pop=Hirshfeld, from the default guess (P0) or a
           per-Fe fragment guess with chosen sites flipped (beta)
  stage 1  --Link1-- Force Guess=Read Geom=Checkpoint Pop=Hirshfeld

The level matches the archived spin campaign (UBPW91/6-311++G*), except that
IOP(5/13=1) is dropped: an unconverged SCF must fail, because the SCF solution
is the label.

Fragment parity: a neutral Fe fragment has 26 electrons, so its multiplicity
must be odd. Only m=5 (4 unpaired, the Fe atomic ground state) and m=3 (2
unpaired), plus m=7 when needed, are used; each fragment is validated by
spin._validated_fragments, as is the signed total sum = M - 1.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

from .io import write_text_lf
from .jobs import human_job_stem
from .models import Record
from .routes import route_search_kind
from .spin import (
    SPIN_MANIFEST_COLUMNS, _canonical_sha256, _coordinates, _link_header, _route_with,
    _source_geometry_sha256, _validated_fragments, validate_multiplicity,
)

BS_SCF = "SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)"
BS_STAGE0_ROUTE = (
    f"#p UBPW91/6-311++G* {BS_SCF} NoSymm SP Stable=Opt Pop=Hirshfeld "
    "IOP(5/36=1,8/11=1) Int=UltraFine"
)
BS_FORCE_ROUTE = (
    f"#p UBPW91/6-311++G* {BS_SCF} NoSymm Force Pop=Hirshfeld "
    "IOP(5/36=1,8/11=1) Int=UltraFine Geom=Checkpoint Guess=Read"
)
BS_EXTRA_COLUMNS = ["bs_pattern", "flipped_sites", "fragment_unpaired"]
PATTERNS = ("P0", "P1", "P2", "P3")
BOND_CUTOFF = 3.0  # Å, Fe-Fe neighbour cutoff for coordination numbers
FLIP_UNPAIRED = 4  # a flipped site starts as a beta quintet Fe


def _distances(record: Record) -> list[list[float]]:
    xyz = [(a.x, a.y, a.z) for a in record.atoms]
    return [[math.dist(p, q) for q in xyz] for p in xyz]


def coordination(record: Record, cutoff: float = BOND_CUTOFF) -> list[int]:
    d = _distances(record)
    return [sum(1 for j, r in enumerate(row) if j != i and r < cutoff) for i, row in enumerate(d)]


def choose_sites(record: Record) -> dict[str, list[int]]:
    """0-based flipped sites per pattern: P1 central, P2/P3 two non-equivalent surface sites."""
    n = len(record.atoms)
    cn = coordination(record)
    cx = [sum(getattr(a, k) for a in record.atoms) / n for k in "xyz"]
    r = [math.dist((a.x, a.y, a.z), cx) for a in record.atoms]
    central = min(range(n), key=lambda i: (-cn[i], r[i]))
    surface = sorted((i for i in range(n) if i != central), key=lambda i: (cn[i], -r[i]))
    a = surface[0]
    d = _distances(record)
    fingerprint = [sorted(row) for row in d]

    def fp_gap(i: int) -> float:
        return max(abs(x - y) for x, y in zip(fingerprint[i], fingerprint[a]))

    # Second surface site: not bonded to A, and geometrically distinct from A
    # (largest sorted-distance fingerprint difference among low-CN candidates).
    low = [i for i in surface[1:] if cn[i] <= cn[a] + 1 and d[a][i] >= BOND_CUTOFF] or surface[1:]
    b = max(low, key=fp_gap)
    return {"P0": [], "P1": [central], "P2": [a], "P3": [b]}


def fragment_assignment(record: Record, multiplicity: int, flipped: list[int]) -> list[int]:
    """Signed unpaired electrons per atom (one Fe fragment per atom).

    Flipped sites start at -4; the others start at +4 and are lowered to +2
    (highest coordination first, where moments are smallest) or raised to +6
    (lowest coordination first) until the signed sum equals M - 1.
    """
    n = len(record.atoms)
    if any(a.symbol != "Fe" for a in record.atoms):
        raise ValueError("the BS pilot generator handles pure Fe clusters only")
    up = [i for i in range(n) if i not in flipped]
    target = multiplicity - 1 + FLIP_UNPAIRED * len(flipped)
    values = {i: 4 for i in up}
    diff = target - 4 * len(up)
    if diff % 2:
        raise ValueError(f"M={multiplicity} cannot be reached with even per-Fe unpaired counts")
    cn = coordination(record)
    if diff < 0:
        order = sorted(up, key=lambda i: -cn[i])
        steps = -diff // 2
        if steps > len(up):
            raise ValueError(f"M={multiplicity} too low for {len(flipped)} flipped site(s) with m>=3 fragments")
        for i in order[:steps]:
            values[i] = 2
    elif diff > 0:
        order = sorted(up, key=lambda i: cn[i])
        steps = diff // 2
        if steps > len(up):
            raise ValueError(f"M={multiplicity} too high for m<=7 fragments")
        for i in order[:steps]:
            values[i] = 6
    signed = [(-FLIP_UNPAIRED if i in flipped else values[i]) for i in range(n)]
    assert sum(signed) == multiplicity - 1
    return signed


def fragment_specification(record: Record, multiplicity: int, signed: list[int], name: str) -> dict:
    return {
        "name": name,
        "target_multiplicity": multiplicity,
        "fragments": [
            {"atoms": [i + 1], "charge": 0, "multiplicity": abs(s) + 1,
             "orientation": "alpha" if s >= 0 else "beta"}
            for i, s in enumerate(signed)
        ],
    }


def render_bs_input(
    record: Record, multiplicity: int, pattern: str, flipped: list[int],
    memory: str = "24GB", nproc: int = 12,
) -> tuple[str, list[dict[str, str]], list[int] | None]:
    validate_multiplicity(record, multiplicity)
    job = f"{record.record_id}-bs-m{multiplicity}-{pattern.lower()}"
    chk = f"{job}.chk"
    signed: list[int] | None = None
    if pattern == "P0":
        route0 = BS_STAGE0_ROUTE
        cm_line = f"{record.charge} {multiplicity}"
        coords = _coordinates(record.atoms)
        spec_sha = ""
    else:
        signed = fragment_assignment(record, multiplicity, flipped)
        spec = fragment_specification(record, multiplicity, signed, job)
        atom_map, states = _validated_fragments(record, spec)  # per-fragment parity + total
        route0 = _route_with(BS_STAGE0_ROUTE, f"Guess=(Fragment={len(states)},Always)")
        cm_line = f"{record.charge} {multiplicity} " + " ".join(f"{q} {m}" for q, m in states)
        coords = _coordinates(record.atoms, atom_map)
        spec_sha = _canonical_sha256(spec)
    sites = ";".join(str(i + 1) for i in flipped)
    title = (f"ClusterMLIP BS pilot; record={record.record_id}; pattern={pattern}; "
             f"multiplicity={multiplicity}; flipped_atoms={sites or 'none'}")
    lines = _link_header(chk, memory, nproc)
    lines += [route0, "", title + "; stage=0 stable", "", cm_line, *coords, ""]
    lines += ["--Link1--", *_link_header(chk, memory, nproc),
              BS_FORCE_ROUTE, "", title + "; stage=1 force", "",
              f"{record.charge} {multiplicity}", ""]
    common = {
        "chain_id": job, "pathway": "broken_symmetry_pilot",
        "parent_record_id": record.record_id, "source": record.source,
        "formula": record.formula, "config_type": record.config_type,
        "state_inference": str(record.metadata.get("state_inference", "")),
        "source_geometry_sha256": _source_geometry_sha256(record),
        "high_spin_multiplicity": "", "final_target_multiplicity": str(multiplicity),
        "intended_charge": str(record.charge), "intended_multiplicity": str(multiplicity),
        "spin_flip_index": "", "checkpoint": chk,
        "fragment_label": pattern, "fragment_count": "" if signed is None else str(len(signed)),
        "fragment_spec_sha256": spec_sha,
        "bs_pattern": pattern, "flipped_sites": sites,
        "fragment_unpaired": "" if signed is None else " ".join(str(s) for s in signed),
    }
    rows = [
        {**common, "job_id": f"{job}-s00", "stage_index": "0",
         "initialization": "default_guess" if signed is None else "per_fe_fragment_guess",
         "audit_classification": "bs_pilot_stable_sp", "route_search_kind": route_search_kind(route0),
         "first_route": route0, "predecessor_job_id": "", "predecessor_multiplicity": "",
         "predecessor_checkpoint": "", "checkpoint_lineage": f"bs:{pattern}:{chk}"},
        {**common, "job_id": f"{job}-s01", "stage_index": "1",
         "initialization": "checkpoint_read", "audit_classification": "bs_pilot_force",
         "route_search_kind": route_search_kind(BS_FORCE_ROUTE), "first_route": BS_FORCE_ROUTE,
         "predecessor_job_id": f"{job}-s00", "predecessor_multiplicity": str(multiplicity),
         "predecessor_checkpoint": chk, "checkpoint_lineage": f"bs:{pattern}:{chk}>force"},
    ]
    return "\n".join(lines) + "\n", rows, signed


def write_bs_jobs(
    records: list[Record], output: Path, multiplicity: int,
    patterns: tuple[str, ...] = PATTERNS, memory: str = "24GB", nproc: int = 12,
) -> int:
    unknown = sorted(set(patterns) - set(PATTERNS))
    if unknown:
        raise ValueError(f"unknown BS patterns {unknown}; choose from {PATTERNS}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing campaign {output}")
    files: list[tuple[str, str]] = []
    rows: list[dict[str, str]] = []
    for record in records:
        sites = choose_sites(record)
        for pattern in patterns:
            text, job_rows, _ = render_bs_input(record, multiplicity, pattern, sites[pattern], memory, nproc)
            name = f"{human_job_stem(record)}__bs-m{multiplicity}-{pattern.lower()}.gjf"
            files.append((name, text))
            for row in job_rows:
                row["input"] = f"inputs/{name}"
                row["output"] = f"{Path(name).stem}.log"
            rows.extend(job_rows)
    names = [n for n, _ in files]
    if len(set(names)) != len(names):
        raise ValueError("duplicate BS input filenames")
    (output / "inputs").mkdir(parents=True)
    for name, text in files:
        write_text_lf(output / "inputs" / name, text)
    for row in rows:
        row["input_sha256"] = hashlib.sha256((output / row["input"]).read_bytes()).hexdigest()
    with (output / "spin_jobs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPIN_MANIFEST_COLUMNS + BS_EXTRA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(files)
