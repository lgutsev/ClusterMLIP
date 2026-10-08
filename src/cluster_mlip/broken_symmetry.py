"""Fixed-geometry broken-symmetry (BS) pilot inputs for Fe clusters.

Every job holds geometry, total charge and multiplicity fixed and starts the SCF
from a different per-Fe fragment pattern, using the archived tandem structure
(see fragment_tandem.py and docs/fragment-tandem-method.md):

  link 0  SP Guess=(Fragment=N)                 one fragment per Fe atom
  link 1  Stable=Opt Geom=Checkpoint Guess=Read  fixed geometry
  link 2  Force Geom=Checkpoint Guess=Read       forces on the stable state

Patterns (0-based site indices from choose_sites, printed 1-based in sites.csv):
  P0   every Fe alpha (aligned reference)
  P1   central (highest-coordination) Fe beta
  P2   surface site A (lowest coordination) beta
  P3   surface site B (not bonded to A, geometrically distinct) beta
  P0Q, P1Q  as P0/P1 but charge-separated initialization: central Fe(-1), site A Fe(+1)
  C0   control: default (Harris) guess, Stable=Opt then Force, no fragment link

Fragment charges only set the electron count of each atomic guess (26 - q for
Fe). They are not oxidation states, and the SCF decides the final charges.

Per-atom unpaired counts must match the fragment parity: neutral Fe (26 e) takes
2, 4 or 6 unpaired (m = 3, 5, 7); Fe(+1) and Fe(-1) (25, 27 e) take 3 or 5
(m = 4, 6). The non-flipped sites are lowered (highest coordination first) or
raised (lowest coordination first) in steps of two until the signed sum is M - 1.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from .fragment_tandem import (
    ARCHIVED_LEVEL, Fragment, FragmentPlan, LevelOfTheory, TandemJob, render_tandem,
    validate_fragment_plan,
)
from .io import write_text_lf
from .jobs import human_job_stem
from .models import Record
from .routes import route_search_kind
from .spin import SPIN_MANIFEST_COLUMNS, _canonical_sha256, _source_geometry_sha256, validate_multiplicity

BS_EXTRA_COLUMNS = [
    "bs_pattern", "flipped_sites", "charged_sites", "fragment_charges", "fragment_multiplicities",
    "ideal_guess_s2", "stage_kind", "tandem_stages",
]
PATTERNS = ("P0", "P1", "P2", "P3")
ALL_PATTERNS = PATTERNS + ("P0Q", "P1Q", "C0")
BOND_CUTOFF = 3.0  # Å, Fe-Fe neighbour cutoff for coordination numbers
FLIP_UNPAIRED = 4  # a flipped neutral site starts as a beta quintet Fe
# Allowed unpaired counts and the starting value, by Fe fragment charge.
UNPAIRED_CHOICES = {0: (2, 4, 6), 1: (3, 5), -1: (3, 5)}
UNPAIRED_START = {0: 4, 1: 5, -1: 3}


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
    return {"P0": [], "P1": [central], "P2": [a], "P3": [b], "P0Q": [], "P1Q": [central], "C0": [],
            "_central": [central], "_surface_a": [a]}


def pattern_charges(record: Record, pattern: str, sites: dict[str, list[int]]) -> dict[int, int]:
    """Charge-separated patterns: central Fe(-1), surface site A Fe(+1); others neutral."""
    if not pattern.endswith("Q"):
        return {}
    central, surface_a = sites["_central"][0], sites["_surface_a"][0]
    return {central: -1, surface_a: +1}


def fragment_assignment(
    record: Record, multiplicity: int, flipped: list[int], charges: dict[int, int] | None = None,
) -> list[int]:
    """Signed unpaired electrons per Fe atom (one fragment per atom)."""
    n = len(record.atoms)
    if any(a.symbol != "Fe" for a in record.atoms):
        raise ValueError("the BS pilot generator handles pure Fe clusters only")
    charges = charges or {}
    if any(q not in UNPAIRED_CHOICES for q in charges.values()):
        raise ValueError(f"Fe fragment charges must be in {sorted(UNPAIRED_CHOICES)}")
    values = {i: UNPAIRED_START[charges.get(i, 0)] for i in range(n)}  # neutral start = FLIP_UNPAIRED
    up =[i for i in range(n) if i not in flipped]
    target = multiplicity - 1 + sum(values[i] for i in flipped)
    diff = target - sum(values[i] for i in up)
    if diff % 2:
        raise ValueError(f"M={multiplicity} cannot be reached: the unpaired-count parity is wrong")
    cn = coordination(record)
    step = -2 if diff < 0 else 2
    order = sorted(up, key=lambda i: -cn[i]) if diff < 0 else sorted(up, key=lambda i: cn[i])
    remaining = abs(diff) // 2
    # Several passes so one site can move twice when every site has been moved once.
    while remaining:
        moved = False
        for i in order:
            choices = UNPAIRED_CHOICES[charges.get(i, 0)]
            if remaining and values[i] + step in choices:
                values[i] += step
                remaining -= 1
                moved = True
        if not moved:
            raise ValueError(f"M={multiplicity} is out of range for {len(flipped)} flipped site(s) "
                             f"with the allowed per-Fe moments")
    signed = [(-values[i] if i in flipped else values[i]) for i in range(n)]
    if sum(signed) != multiplicity - 1:
        raise AssertionError("internal error: signed unpaired sum")
    return signed


def build_plan(record: Record, multiplicity: int, signed: list[int], charges: dict[int, int]) -> FragmentPlan:
    fragments = [
        Fragment((i + 1,), charges.get(i, 0), (abs(s) + 1) * (1 if s >= 0 else -1))
        for i, s in enumerate(signed)
    ]
    return validate_fragment_plan([a.symbol for a in record.atoms], record.charge, multiplicity, fragments)


def make_job(
    record: Record, multiplicity: int, pattern: str, sites: dict[str, list[int]],
    level: LevelOfTheory = ARCHIVED_LEVEL, memory: str = "24GB", nproc: int = 12, force_stage: bool = True,
) -> tuple[TandemJob, list[int] | None, dict[int, int]]:
    validate_multiplicity(record, multiplicity)
    if pattern not in ALL_PATTERNS:
        raise ValueError(f"unknown BS pattern {pattern}; choose from {ALL_PATTERNS}")
    name = f"{record.record_id}-bs-m{multiplicity}-{pattern.lower()}"
    flipped = sites[pattern]
    title = (f"ClusterMLIP BS pilot; record={record.record_id}; pattern={pattern}; multiplicity={multiplicity}; "
             f"flipped_atoms={';'.join(str(i + 1) for i in flipped) or 'none'}")
    coords = tuple((a.x, a.y, a.z) for a in record.atoms)
    symbols = tuple(a.symbol for a in record.atoms)
    if pattern == "C0":
        job = TandemJob(name, symbols, coords, record.charge, multiplicity, None, level, force_stage,
                        title=title, memory=memory, nproc=nproc)
        return job, None, {}
    charges = pattern_charges(record, pattern, sites)
    signed = fragment_assignment(record, multiplicity, flipped, charges)
    plan = build_plan(record, multiplicity, signed, charges)
    job = TandemJob(name, symbols, coords, record.charge, multiplicity, plan, level, force_stage,
                    title=title, memory=memory, nproc=nproc)
    return job, signed, charges


def manifest_rows(record: Record, job: TandemJob, pattern: str, sites: dict[str, list[int]],
                  charges: dict[int, int]) -> list[dict[str, str]]:
    kinds, routes = job.stage_kinds(), job.routes()
    plan = job.plan
    common = {
        "chain_id": job.name, "pathway": "broken_symmetry_pilot",
        "parent_record_id": record.record_id, "source": record.source,
        "formula": record.formula, "config_type": record.config_type,
        "state_inference": str(record.metadata.get("state_inference", "")),
        "source_geometry_sha256": _source_geometry_sha256(record),
        "high_spin_multiplicity": "", "final_target_multiplicity": str(job.multiplicity),
        "intended_charge": str(job.charge), "intended_multiplicity": str(job.multiplicity),
        "spin_flip_index": "", "checkpoint": job.checkpoint,
        "fragment_label": pattern, "fragment_count": "" if plan is None else str(len(plan.fragments)),
        "fragment_spec_sha256": "" if plan is None else _canonical_sha256(
            [[list(f.atoms), f.charge, f.multiplicity] for f in plan.fragments]),
        "bs_pattern": pattern,
        "flipped_sites": ";".join(str(i + 1) for i in sites[pattern]),
        "charged_sites": ";".join(f"{i + 1}:{q:+d}" for i, q in sorted(charges.items())),
        "fragment_charges": "" if plan is None else " ".join(str(f.charge) for f in plan.fragments),
        "fragment_multiplicities": "" if plan is None else " ".join(str(f.multiplicity) for f in plan.fragments),
        "ideal_guess_s2": "" if plan is None else f"{plan.ideal_guess_s2:.4f}",
        "tandem_stages": ">".join(kinds),
    }
    rows = []
    for index, (kind, route) in enumerate(zip(kinds, routes)):
        rows.append({
            **common, "job_id": f"{job.name}-s{index:02d}", "stage_index": str(index), "stage_kind": kind,
            "initialization": {"fragment_init": "per_fe_fragment_guess", "stable_opt": "checkpoint_read",
                               "force": "checkpoint_read", "control_stable_opt": "default_guess"}[kind],
            "audit_classification": f"bs_pilot_{kind}",
            "route_search_kind": route_search_kind(route), "first_route": route,
            "predecessor_job_id": "" if index == 0 else f"{job.name}-s{index - 1:02d}",
            "predecessor_multiplicity": "" if index == 0 else str(job.multiplicity),
            "predecessor_checkpoint": "" if index == 0 else job.checkpoint,
            "checkpoint_lineage": f"bs:{pattern}:{job.checkpoint}" + "".join(f">{k}" for k in kinds[1:index + 1]),
        })
    return rows


def site_table(record: Record, sites: dict[str, list[int]], reference_spins: dict[int, float] | None = None) -> list[dict]:
    """Per-atom identity used to assign patterns: 1-based index, CN, radius, roles, reference spin."""
    n = len(record.atoms)
    cn = coordination(record)
    cx = [sum(getattr(a, k) for a in record.atoms) / n for k in "xyz"]
    roles: dict[int, list[str]] = {}
    for pattern in ("P1", "P2", "P3"):
        for i in sites[pattern]:
            roles.setdefault(i, []).append(pattern)
    rows = []
    for i, a in enumerate(record.atoms):
        rows.append({
            "record": record.record_id, "atom": i + 1, "symbol": a.symbol, "cn_3.0A": cn[i],
            "r_centroid_A": f"{math.dist((a.x, a.y, a.z), cx):.3f}",
            "flipped_in": "+".join(roles.get(i, [])),
            "reference_mulliken_spin": "" if not reference_spins or (i + 1) not in reference_spins
            else f"{reference_spins[i + 1]:.3f}",
        })
    return rows


def write_bs_jobs(
    records: list[Record], output: Path, multiplicity: int,
    patterns: tuple[str, ...] = PATTERNS, memory: str = "24GB", nproc: int = 12,
    level: LevelOfTheory = ARCHIVED_LEVEL, reference_spins: dict[str, dict[int, float]] | None = None,
) -> int:
    unknown = sorted(set(patterns) - set(ALL_PATTERNS))
    if unknown:
        raise ValueError(f"unknown BS patterns {unknown}; choose from {ALL_PATTERNS}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing campaign {output}")
    files: list[tuple[str, str]] = []
    rows: list[dict[str, str]] = []
    sites_rows: list[dict] = []
    for record in records:
        sites = choose_sites(record)
        sites_rows += site_table(record, sites, (reference_spins or {}).get(record.record_id))
        for pattern in patterns:
            job, _, charges = make_job(record, multiplicity, pattern, sites, level, memory, nproc)
            text = render_tandem(job)  # raises if the independent inspection finds any problem
            name = f"{human_job_stem(record)}__bs-m{multiplicity}-{pattern.lower()}.gjf"
            files.append((name, text))
            for row in manifest_rows(record, job, pattern, sites, charges):
                row["input"] = f"inputs/{name}"
                row["output"] = f"{Path(name).stem}.log"
                rows.append(row)
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
    with (output / "sites.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sites_rows[0]))
        writer.writeheader()
        writer.writerows(sites_rows)
    write_text_lf(output / "level_of_theory.json", json.dumps(level.__dict__, indent=1) + "\n")
    return len(files)
