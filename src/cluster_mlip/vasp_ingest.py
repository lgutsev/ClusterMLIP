"""Ingest finished periodic VASP runs as labeled frames.

`vasp.collect_vasp_campaign` assembles the Delta-model interaction labels of a
campaign this package prepared. This module reads *any* finished VASP job
directory -- relaxations, MD, single points, slabs, bulk, gas references --
and writes every ionic step as a periodic extxyz frame:

* ``REF_energy`` -- the free energy (``e_fr_energy``), the energy the forces
  are the derivative of; ``energy_sigma0`` is kept alongside.
* ``REF_forces`` -- for every atom, including frozen ones (VASP reports full
  forces for selective-dynamics atoms; ``fixed`` marks them).
* ``vasp_magmom`` -- the per-atom ``tot`` moment of that ionic step's
  ``magnetization (x)`` table in OUTCAR (needs LORBIT; absent for ISPIN = 1).
* ``REF_stress`` -- only for variable-cell runs (ISIF >= 3). A fixed-cell slab
  stress is not a training target.

The electronic state is recorded, never inferred. A run with NUPDOWN set has
a declared multiplicity M = NUPDOWN + 1. A released-spin run reports the
total moment it converged to (OSZICAR ``mag=``) and, when a cluster is named,
the sign pattern of the cluster's local moments; it gets a multiplicity only
if the moment is within ``integer_tolerance`` of an integer, and
``spin_constraint`` says which case applies. The total moment alone does not
identify a state of a supported cluster: two runs at the same M can differ in
their local-moment pattern, and that pattern is what `moment_pattern` keeps.

Frames whose SCF hit NELM are not written (they are counted in jobs.csv).
The final step of a relaxation that reached EDIFFG is ``relaxed_minimum``;
every other step is ``relaxation_step`` (or ``md_step`` / ``single_point``).
Nothing is deduplicated: consecutive relaxation steps are strongly
correlated, so split by ``job`` (the parent group), not by frame.

Standard library only; vasprun.xml and OUTCAR are read, never modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import write_text_lf
from .periodic import Structure, write_structures

# 1 kBar in eV/A^3, and VASP's sign convention (positive = compressive) flipped
# to ASE's (positive = tensile).
_KBAR_TO_EV_A3 = -0.1 / 160.21766208

# |m| below this (mu_B) counts as a quenched site in `moment_pattern`.
RESOLVED_MOMENT = 0.5

_RELAX_IBRION = {1, 2, 3}
_MD_IBRION = {0}


@dataclass
class IonicStep:
    cell: list[tuple[float, float, float]]
    positions: list[tuple[float, float, float]]
    forces: list[tuple[float, float, float]]
    energy_free: float
    energy_sigma0: float | None
    stress_kbar: list[float] | None
    n_scsteps: int


@dataclass
class VaspRun:
    job: str
    path: Path
    symbols: list[str]
    fixed: list[bool] | None
    partially_fixed: int
    params: dict[str, Any]
    incar: dict[str, Any]
    version: str
    potcars: list[str]
    kpoints: str
    charge: float
    steps: list[IonicStep]
    total_moments: list[float] = field(default_factory=list)
    atomic_moments: list[list[float]] | None = None
    ionic_converged: bool | None = None
    complete: bool = False
    sha256: str = ""


def _value(item: ET.Element) -> Any:
    text = (item.text or "").strip()
    kind = item.get("type")
    if kind == "logical":
        return text.upper().startswith("T")
    if kind == "string":
        return text
    if item.tag == "v":
        return [float(t) for t in text.split()]
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if kind == "int" else number


def _flat_params(block: ET.Element | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if block is None:
        return out
    for item in block.iter():
        name = item.get("name")
        if item.tag in {"i", "v"} and name and name not in out:
            out[name] = _value(item)
    return out


def _vectors(varray: ET.Element | None) -> list[tuple[float, ...]]:
    if varray is None:
        return []
    return [tuple(float(t) for t in (v.text or "").split()) for v in varray.findall("v")]


def _named(parent: ET.Element, tag: str, name: str) -> ET.Element | None:
    for child in parent.findall(tag):
        if child.get("name") == name:
            return child
    return None


def _cartesian(frac: list[tuple[float, ...]], basis: list[tuple[float, ...]]) -> list[tuple[float, float, float]]:
    return [
        tuple(sum(f[k] * basis[k][j] for k in range(3)) for j in range(3))  # type: ignore[misc]
        for f in frac
    ]


def _tail(path: Path, n_bytes: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(handle.tell() - n_bytes, 0))
        return handle.read()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_vasprun_trajectory(path: Path, job: str = "") -> VaspRun:
    """Every ionic step of a vasprun.xml, with the run's settings and species.

    Raises ValueError for a vasprun.xml that is truncated or has no ionic step.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{path}: truncated or malformed vasprun.xml ({exc})") from exc
    generator = _flat_params(root.find("generator"))
    params = _flat_params(root.find("parameters"))
    incar = _flat_params(root.find("incar"))

    atominfo = root.find("atominfo")
    if atominfo is None:
        raise ValueError(f"{path}: no <atominfo>")
    symbols: list[str] = []
    potcars: list[str] = []
    valence = 0.0
    for array in atominfo.findall("array"):
        rows = [[(c.text or "").strip() for c in rc.findall("c")] for rc in array.iter("rc")]
        if array.get("name") == "atoms":
            symbols = [row[0] for row in rows]
        elif array.get("name") == "atomtypes":
            for count, _element, _mass, zval, pseudo in rows:
                valence += int(count) * float(zval)
                potcars.append(" ".join(pseudo.split()))

    fixed: list[bool] | None = None
    partial = 0
    initial = _named(root, "structure", "initialpos")
    selective = _named(initial, "varray", "selective") if initial is not None else None
    if selective is not None:
        flags = [(v.text or "").split() for v in selective.findall("v")]
        fixed = [all(f == "F" for f in row) for row in flags]
        partial = sum(1 for row in flags if "T" in row and "F" in row)

    kpoints = ""
    generation = root.find("kpoints/generation")
    if generation is not None:
        divisions = _named(generation, "v", "divisions")
        grid = " ".join((divisions.text or "").split()) if divisions is not None else ""
        kpoints = f"{generation.get('param', '')} {grid}".strip()
    elif root.find("kpoints") is not None:
        kpoints = f"{len(_vectors(_named(root.find('kpoints'), 'varray', 'kpointlist')))} explicit"  # type: ignore[arg-type]

    steps: list[IonicStep] = []
    for calc in root.findall("calculation"):
        structure = calc.find("structure")
        energy = {i.get("name"): float(i.text) for i in calc.findall("energy/i") if i.text}
        forces = _vectors(_named(calc, "varray", "forces"))
        if structure is None or "e_fr_energy" not in energy or len(forces) != len(symbols):
            continue  # an interrupted last step
        basis = _vectors(_named(structure.find("crystal"), "varray", "basis"))  # type: ignore[arg-type]
        frac = _vectors(_named(structure, "varray", "positions"))
        stress = _vectors(_named(calc, "varray", "stress"))
        steps.append(IonicStep(
            cell=[tuple(row) for row in basis],  # type: ignore[misc]
            positions=_cartesian(frac, basis),
            forces=[tuple(f) for f in forces],  # type: ignore[misc]
            energy_free=energy["e_fr_energy"],
            energy_sigma0=energy.get("e_0_energy"),
            stress_kbar=[x for row in stress for x in row] if len(stress) == 3 else None,
            n_scsteps=len(calc.findall("scstep")),
        ))
    if not steps:
        raise ValueError(f"{path}: no complete ionic step")
    return VaspRun(
        job=job or path.parent.name, path=path, symbols=symbols, fixed=fixed, partially_fixed=partial,
        params=params, incar=incar, version=str(generator.get("version", "")).strip(), potcars=potcars,
        kpoints=kpoints, charge=valence - float(params.get("NELECT", valence)), steps=steps,
        complete=_tail(path, 256).rstrip().endswith(b"</modeling>"),
    )


def parse_outcar_moment_tables(path: Path, n_atoms: int) -> list[list[float]]:
    """The ``tot`` column of every ``magnetization (x)`` table in OUTCAR, in order.

    VASP prints one table per ionic step and repeats the last one at the end of
    the run, so a finished run of N steps has N + 1 tables.
    """
    tables: list[list[float]] = []
    current: list[float] | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(" magnetization (x)"):
                current = []
                continue
            if current is None:
                continue
            tokens = line.split()
            if tokens and tokens[0].isdigit():
                current.append(float(tokens[-1]))
                if len(current) == n_atoms:
                    tables.append(current)
                    current = None
            elif current and (not tokens or tokens[0].startswith("tot") or tokens[0].startswith("---")):
                current = None  # a table for fewer atoms than expected; drop it
    return tables


_OSZICAR_MAG = re.compile(r"mag=\s*(-?[\d.Ee+-]+)")


def parse_oszicar_moments(path: Path) -> list[float | None]:
    """Total moment (``mag=``) of every ionic step in OSZICAR; None for ISPIN = 1."""
    out: list[float | None] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.split()
            if " F=" not in line or not tokens or not tokens[0].isdigit():
                continue  # SCF iteration lines; ionic lines start with the step number
            match = _OSZICAR_MAG.search(line)
            out.append(float(match.group(1)) if match else None)
    return out


def read_vasp_job(job_dir: Path, job: str = "") -> VaspRun:
    """vasprun.xml plus the per-step moments from OUTCAR/OSZICAR of one job directory."""
    run = parse_vasprun_trajectory(job_dir / "vasprun.xml", job or job_dir.name)
    run.sha256 = _sha256(job_dir / "vasprun.xml")
    n_steps = len(run.steps)
    oszicar = job_dir / "OSZICAR"
    if oszicar.is_file():
        moments = parse_oszicar_moments(oszicar)
        if len(moments) >= n_steps and all(m is not None for m in moments[:n_steps]):
            run.total_moments = [float(m) for m in moments[:n_steps]]  # type: ignore[arg-type]
    outcar = job_dir / "OUTCAR"
    if outcar.is_file():
        tables = parse_outcar_moment_tables(outcar, len(run.symbols))
        # One table per step (+ the final repeat). Anything else cannot be
        # aligned with the steps, and a misaligned moment is worse than none.
        if n_steps <= len(tables) <= n_steps + 1:
            run.atomic_moments = tables[:n_steps]
        text_tail = _tail(outcar, 200_000).decode("utf-8", errors="replace")
        run.complete = run.complete and "General timing and accounting" in text_tail
        if run.params.get("IBRION") in _RELAX_IBRION and int(run.params.get("NSW", 0)) > 0:
            with outcar.open(encoding="utf-8", errors="replace") as handle:
                run.ionic_converged = any("reached required accuracy" in line for line in handle)
    return run


def level_of_theory(run: VaspRun) -> str:
    """A normalized description of the settings that change the energy surface."""
    p, i = run.params, run.incar
    functional = str(i.get("METAGGA") or i.get("GGA") or "")
    if not functional:
        families = {pc.split()[0] for pc in run.potcars if pc}
        functional = "PBE" if families <= {"PAW_PBE"} else "POTCAR-default"
    if p.get("LHFCALC") or i.get("LHFCALC"):
        functional += f"+HF(AEXX={p.get('AEXX', i.get('AEXX'))},HFSCREEN={p.get('HFSCREEN', i.get('HFSCREEN'))})"
    # INCAR as written wins: <parameters> truncates strings (PREC "accura")
    # and reports the cutoff as ENMAX, not ENCUT.
    parts = [
        f"VASP {run.version}".strip(), functional,
        f"IVDW={i.get('IVDW', p.get('IVDW', 0)) or 0}",
        f"LDAU={'T' if (p.get('LDAU') or i.get('LDAU')) else 'F'}",
        f"ENCUT={float(i.get('ENCUT', p.get('ENMAX', 0))):g}",
        f"PREC={str(i.get('PREC', p.get('PREC', ''))).strip().lower()}",
        f"ISMEAR={p.get('ISMEAR')}/SIGMA={p.get('SIGMA')}",
        f"LREAL={_flag(i.get('LREAL', p.get('LREAL', 'F')))}",
        f"LASPH={_flag(i.get('LASPH', p.get('LASPH', False)))}",
        # A dipole correction changes slab energies and forces; corrected and
        # uncorrected frames are different labels.
        f"dipole={_dipole(i, p)}",
        f"k={run.kpoints}",
    ]
    return " | ".join(parts)


def _flag(value: Any) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    text = str(value).strip().strip(".").upper()
    return {"TRUE": "T", "FALSE": "F", ".TRUE": "T", ".FALSE": "F"}.get(text, text or "F")


def _dipole(incar: dict[str, Any], params: dict[str, Any]) -> str:
    idipol = int(incar.get("IDIPOL", params.get("IDIPOL", 0)) or 0)
    if idipol == 0:
        return "none"
    ldipol = _flag(incar.get("LDIPOL", params.get("LDIPOL", False)))
    return f"IDIPOL={idipol}/LDIPOL={ldipol}"


def _potcar_by_element(run: VaspRun) -> dict[str, str]:
    """Element -> PAW dataset title. ISPIN and the POTCAR set are not part of
    `level_of_theory` (a closed-shell reference and a slab differ in both but
    share the surface); POTCAR consistency is checked across jobs instead."""
    out: dict[str, str] = {}
    for title in run.potcars:
        tokens = title.split()
        if len(tokens) >= 2:
            out[tokens[1].split("_")[0]] = title
    return out


def _spin_label(run: VaspRun, moment: float | None, tolerance: float) -> dict[str, Any]:
    if int(run.params.get("ISPIN", 1)) == 1:
        return {"spin_constraint": "ISPIN=1", "multiplicity": 1}
    nupdown = float(run.params.get("NUPDOWN", -1))
    if nupdown >= 0:
        return {"spin_constraint": "NUPDOWN", "multiplicity": round(nupdown) + 1}
    if moment is None:
        return {"spin_constraint": "free"}
    if abs(moment - round(moment)) <= tolerance:
        return {"spin_constraint": "free", "multiplicity": abs(round(moment)) + 1}
    return {"spin_constraint": "free_fractional"}


def moment_pattern(symbols: list[str], moments: list[float], mask: list[bool]) -> str:
    """Sign pattern of the cluster's local moments, e.g. ``Fe:++-`` or ``Fe:+0+``."""
    by_element: dict[str, str] = {}
    for symbol, m, inside in zip(symbols, moments, mask):
        if inside:
            by_element.setdefault(symbol, "")
            by_element[symbol] += "0" if abs(m) < RESOLVED_MOMENT else ("+" if m > 0 else "-")
    return " ".join(f"{el}:{signs}" for el, signs in by_element.items())


def _config_type(run: VaspRun, index: int) -> str:
    ibrion = run.params.get("IBRION")
    nsw = int(run.params.get("NSW", 0))
    if nsw == 0 or ibrion not in _RELAX_IBRION | _MD_IBRION:
        return "single_point"
    if ibrion in _MD_IBRION:
        return "md_step"
    if index == len(run.steps) - 1 and run.ionic_converged:
        return "relaxed_minimum"
    return "relaxation_step"


def run_frames(
    run: VaspRun,
    *,
    every: int = 1,
    final_only: bool = False,
    cluster_elements: set[str] | None = None,
    integer_tolerance: float = 0.05,
) -> tuple[list[Structure], int]:
    """Frames for the selected ionic steps, and how many steps were dropped for SCF failure."""
    nelm = int(run.params.get("NELM", 60))
    level = level_of_theory(run)
    variable_cell = int(run.params.get("ISIF", 2)) >= 3
    last = len(run.steps) - 1
    selected = [last] if final_only else sorted({*range(0, last + 1, max(every, 1)), last})
    mask = [s in cluster_elements for s in run.symbols] if cluster_elements else None
    frames: list[Structure] = []
    scf_failed = 0
    for index in selected:
        step = run.steps[index]
        if step.n_scsteps >= nelm:
            scf_failed += 1
            continue
        moment = run.total_moments[index] if run.total_moments else None
        info: dict[str, Any] = {
            "structure_id": f"{run.job}:{index:04d}",
            "job": run.job,
            "ionic_step": index,
            "n_ionic_steps": len(run.steps),
            "config_type": _config_type(run, index),
            "charge": round(run.charge, 6) if abs(run.charge - round(run.charge)) > 1e-6 else round(run.charge),
            **_spin_label(run, moment, integer_tolerance),
            "REF_energy": step.energy_free,
            "method": "VASP",
            "label_level": level,
            "potcars": ",".join(run.potcars),
            "scf_steps": step.n_scsteps,
            "source": str(run.path),
            "vasprun_sha256": run.sha256,
        }
        if "multiplicity" in info:
            info["spin"] = info["multiplicity"]
            info["total_spin"] = info["multiplicity"] - 1
        if step.energy_sigma0 is not None:
            info["energy_sigma0"] = step.energy_sigma0
        if moment is not None:
            info["total_magnetization"] = moment
        if variable_cell and step.stress_kbar is not None:
            info["REF_stress"] = " ".join(f"{_KBAR_TO_EV_A3 * x:.10g}" for x in step.stress_kbar)
        arrays: dict[str, list[Any]] = {"REF_forces": step.forces}
        if run.fixed is not None:
            arrays["fixed"] = run.fixed
        if mask is not None and any(mask):
            arrays["cluster"] = [int(m) for m in mask]
        if run.atomic_moments is not None:
            moments = run.atomic_moments[index]
            arrays["vasp_magmom"] = moments
            if mask is not None and any(mask):
                info["cluster_moment"] = round(sum(m for m, inside in zip(moments, mask) if inside), 4)
                info["moment_pattern"] = moment_pattern(run.symbols, moments, mask)
        frames.append(Structure(list(run.symbols), step.positions, step.cell, (True, True, True), info, arrays))
    return frames, scf_failed


def find_job_dirs(paths: list[Path]) -> list[tuple[str, Path]]:
    """(job name, directory) for every directory holding a vasprun.xml under `paths`."""
    found: list[tuple[str, Path]] = []
    for base in paths:
        if (base / "vasprun.xml").is_file():
            found.append((base.name, base))
            continue
        for vasprun in sorted(base.rglob("vasprun.xml")):
            rel = vasprun.parent.relative_to(base).as_posix()
            found.append((rel if rel != "." else base.name, vasprun.parent))
    names = [name for name, _ in found]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"job names are not unique: {duplicates}; pass the parent directories separately")
    return found


JOB_COLUMNS = [
    "job", "status", "complete", "ionic_converged", "ionic_steps", "frames_written", "scf_failed_steps",
    "fractional_spin_frames", "fractional_spin_dropped",
    "config_type_final", "E_free_final", "spin_constraint", "multiplicity", "total_magnetization_final",
    "cluster_moment_final", "moment_pattern_final", "charge", "n_atoms", "formula", "fixed_atoms",
    "partially_fixed_atoms", "label_level", "potcars", "vasprun_sha256", "path",
]


def _formula(symbols: list[str]) -> str:
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    return "".join(f"{el}{n}" for el, n in counts.items())


@dataclass
class JobSplit:
    """Train/valid/test by whole job. Consecutive ionic steps of one relaxation
    are nearly identical, so a frame-level split would put copies of the test
    frames in the training set."""

    valid_jobs: tuple[str, ...] = ()
    test_jobs: tuple[str, ...] = ()
    valid_fraction: float = 0.0
    test_fraction: float = 0.0
    seed: int = 20260811

    @property
    def requested(self) -> bool:
        return bool(self.valid_jobs or self.test_jobs or self.valid_fraction or self.test_fraction)


def _job_rank(job: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{job}".encode()).hexdigest()


def split_jobs(jobs: list[str], split: JobSplit) -> dict[str, list[str]]:
    """Assign each job to train/valid/test: named jobs first, then a seeded draw."""
    unknown = sorted(j for j in {*split.valid_jobs, *split.test_jobs} if j not in jobs)
    if unknown:
        raise ValueError(f"split names jobs that were not ingested: {unknown}")
    both = sorted(set(split.valid_jobs) & set(split.test_jobs))
    if both:
        raise ValueError(f"jobs named for both valid and test: {both}")
    out = {"train": [], "valid": list(split.valid_jobs), "test": list(split.test_jobs)}
    rest = sorted((j for j in jobs if j not in out["valid"] and j not in out["test"]),
                  key=lambda j: _job_rank(j, split.seed))
    for name, fraction in (("test", split.test_fraction), ("valid", split.valid_fraction)):
        if fraction and not out[name]:
            take = max(1, round(len(jobs) * fraction))
            out[name], rest = rest[:take], rest[take:]
    out["train"] = sorted(rest)
    empty = [name for name, members in out.items() if not members]
    if empty:
        raise ValueError(f"split leaves {', '.join(empty)} without any job ({len(jobs)} jobs ingested)")
    return out


def ingest_vasp_runs(
    paths: list[Path],
    output: Path,
    *,
    every: int = 1,
    final_only: bool = False,
    cluster_elements: set[str] | None = None,
    integer_tolerance: float = 0.05,
    exclude_jobs: tuple[str, ...] = (),
    split: JobSplit | None = None,
    drop_fractional_spin: bool = False,
) -> dict[str, Any]:
    """Write frames.extxyz, jobs.csv and ingest_summary.json for every job found under `paths`.

    With `split`, also write all/train/valid/test.extxyz split by job, the
    layout `cluster-mlip train` reads. `drop_fractional_spin` leaves out
    released-spin frames whose moment is not near an integer (`train` refuses
    them); jobs.csv counts them per job.
    """
    jobs = find_job_dirs(paths)
    missing = sorted(set(exclude_jobs) - {name for name, _ in jobs})
    if missing:
        raise ValueError(f"--exclude-jobs names jobs that were not found: {missing}")
    jobs = [(name, d) for name, d in jobs if name not in exclude_jobs]
    if not jobs:
        raise ValueError(f"no vasprun.xml under {', '.join(str(p) for p in paths)}")
    output.mkdir(parents=True, exist_ok=True)
    frames: list[Structure] = []
    rows: list[dict[str, Any]] = []
    levels: dict[str, list[str]] = {}
    potcars: dict[str, dict[str, list[str]]] = {}
    for name, job_dir in jobs:
        row: dict[str, Any] = {"job": name, "path": str(job_dir)}
        try:
            run = read_vasp_job(job_dir, name)
        except (ValueError, OSError) as exc:
            row["status"] = f"unreadable: {exc}"
            rows.append(row)
            continue
        job_frames, scf_failed = run_frames(
            run, every=every, final_only=final_only,
            cluster_elements=cluster_elements, integer_tolerance=integer_tolerance,
        )
        fractional = sum(1 for f in job_frames if f.info["spin_constraint"] == "free_fractional")
        if drop_fractional_spin:
            job_frames = [f for f in job_frames if f.info["spin_constraint"] != "free_fractional"]
        frames.extend(job_frames)
        level = level_of_theory(run)
        levels.setdefault(level, []).append(name)
        for element, title in _potcar_by_element(run).items():
            potcars.setdefault(element, {}).setdefault(title, []).append(name)
        final = job_frames[-1].info if job_frames and job_frames[-1].info["ionic_step"] == len(run.steps) - 1 else {}
        problems = []
        if not run.complete:
            problems.append("run did not finish")
        if run.ionic_converged is False:
            problems.append("relaxation did not reach EDIFFG")
        if run.steps[-1].n_scsteps >= int(run.params.get("NELM", 60)):
            problems.append("final SCF hit NELM")
        if run.partially_fixed:
            problems.append(f"{run.partially_fixed} atoms partially fixed")
        if int(run.params.get("ISPIN", 1)) == 2 and run.atomic_moments is None:
            problems.append("no per-step moment tables (LORBIT unset or OUTCAR misaligned)")
        row.update(
            status="; ".join(problems) or "ok",
            complete=run.complete,
            ionic_converged="" if run.ionic_converged is None else run.ionic_converged,
            ionic_steps=len(run.steps),
            frames_written=len(job_frames),
            scf_failed_steps=scf_failed,
            fractional_spin_frames=fractional,
            fractional_spin_dropped=fractional if drop_fractional_spin else 0,
            config_type_final=final.get("config_type", ""),
            E_free_final=run.steps[-1].energy_free,
            spin_constraint=final.get("spin_constraint", ""),
            multiplicity=final.get("multiplicity", ""),
            total_magnetization_final=run.total_moments[-1] if run.total_moments else "",
            cluster_moment_final=final.get("cluster_moment", ""),
            moment_pattern_final=final.get("moment_pattern", ""),
            charge=round(run.charge, 6),
            n_atoms=len(run.symbols),
            formula=_formula(run.symbols),
            fixed_atoms=sum(run.fixed) if run.fixed else 0,
            partially_fixed_atoms=run.partially_fixed,
            label_level=level,
            potcars=",".join(run.potcars),
            vasprun_sha256=run.sha256,
        )
        rows.append(row)

    write_structures(frames, output / "frames.extxyz")
    splits: dict[str, list[str]] | None = None
    if split is not None and split.requested:
        usable = sorted({f.info["job"] for f in frames})
        splits = split_jobs(usable, split)
        write_structures(frames, output / "all.extxyz")
        for name, members in splits.items():
            chosen = set(members)
            write_structures([f for f in frames if f.info["job"] in chosen], output / f"{name}.extxyz")
    with (output / "jobs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOB_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    config_types: dict[str, int] = {}
    for frame in frames:
        config_types[frame.info["config_type"]] = config_types.get(frame.info["config_type"], 0) + 1
    summary = {
        "inputs": [str(p) for p in paths],
        "jobs": len(rows),
        "jobs_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "jobs_flagged": {r["job"]: r["status"] for r in rows if r.get("status") != "ok"},
        "frames": len(frames),
        "config_types": config_types,
        "levels_of_theory": levels,
        "mixed_levels": len(levels) > 1,
        "potcar_conflicts": {el: titles for el, titles in potcars.items() if len(titles) > 1},
        "selection": {"every": every, "final_only": final_only},
        "excluded_jobs": sorted(exclude_jobs),
        "drop_fractional_spin": drop_fractional_spin,
        "splits": splits,
        "cluster_elements": sorted(cluster_elements) if cluster_elements else None,
        "energy": "free (e_fr_energy); energy_sigma0 kept in info",
    }
    write_text_lf(output / "ingest_summary.json", json.dumps(summary, indent=1) + "\n")
    return summary
