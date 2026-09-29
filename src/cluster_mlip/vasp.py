"""VASP interaction-energy campaigns for supported clusters.

The supported-cluster Delta-model is

    E(cluster on support) = E_gas(cluster; q, M) + E_support + dE_int,

with the gas-phase term from the UBPW91 cluster MACE and dE_int fitted to
periodic DFT. For every supported structure this module prepares three
single points at *identical* numerical settings in the *same* cell:

    AB  the whole supported system, total spin fixed at NUPDOWN = M - 1
    A   the cluster alone, frozen at its supported geometry, same NUPDOWN
    B   the support alone, frozen, NUPDOWN = 0 (closed-shell support)

and `collect_vasp_campaign` turns them into interaction labels

    dE_int = E_AB - E_A - E_B,    dF_i = F_AB,i - F_A,i  (cluster atoms)
                                         F_AB,i - F_B,i  (support atoms).

A plane-wave basis has no basis-set superposition error, so the frozen
fragments need no ghost atoms. B does not depend on M and is run once per
structure. Because the fragments are frozen, dE_int is not an adsorption
energy (that would also contain the cluster and support strain); the
strain terms come from the gas-phase and support models.

POTCAR files are licensed and never written here: each job gets a
``POTCAR.spec`` (one PAW dataset name per species, in POSCAR order) and the
generated Slurm script concatenates them from ``$VASP_PP_PATH`` on the
cluster. Nothing in this module imports ASE.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shlex
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .io import write_text_lf
from .periodic import Structure, read_structures, write_structures

ROLES = ("AB", "A", "B")

# Materials-Project-style PAW choices; override per element with --potcar.
DEFAULT_POTCARS = {
    "H": "H", "C": "C", "N": "N", "O": "O", "Mg": "Mg_pv", "Al": "Al", "Si": "Si",
    "Ti": "Ti_pv", "V": "V_pv", "Cr": "Cr_pv", "Mn": "Mn_pv", "Fe": "Fe_pv",
    "Co": "Co", "Ni": "Ni_pv", "Cu": "Cu_pv", "Zn": "Zn", "Ce": "Ce",
}
# Elements that receive a share of the starting moment when no explicit moments are given.
MAGNETIC_ELEMENTS = {"Cr", "Mn", "Fe", "Co", "Ni"}


@dataclass(frozen=True)
class VaspSettings:
    gga: str = "PE"
    encut: float = 450.0
    ediff: float = 1e-6
    prec: str = "Accurate"
    ismear: int = 0
    sigma: float = 0.05
    ivdw: int = 12  # D3(BJ); 0 disables dispersion
    lreal: str = "Auto"
    algo: str = "Normal"
    nelm: int = 300
    kpoints: tuple[int, int, int] = (1, 1, 1)
    dipole: bool = True  # IDIPOL = 3: slab normal along the third lattice vector
    magnetic_mixing: bool = True
    support_nupdown: int | None = 0  # None leaves the support's moment free
    potcars: dict[str, str] = field(default_factory=dict)
    incar_overrides: dict[str, str] = field(default_factory=dict)

    def potcar(self, element: str) -> str:
        name = self.potcars.get(element) or DEFAULT_POTCARS.get(element)
        if not name:
            raise ValueError(f"no default POTCAR for {element}; pass --potcar {element}=<name>")
        return name

    def level(self) -> str:
        """A normalized level-of-theory string, recorded on every collected frame."""
        disp = f"+IVDW{self.ivdw}" if self.ivdw else ""
        pots = ",".join(f"{el}:{name}" for el, name in sorted(self.potcars.items()))
        return f"VASP GGA={self.gga}{disp} ENCUT={self.encut:g} ISMEAR={self.ismear} SIGMA={self.sigma:g}" + (
            f" POTCAR[{pots}]" if pots else ""
        )


@dataclass(frozen=True)
class VaspSlurmConfig:
    partition: str = "checkpt"
    account: str = "loni_perovsk27"
    nodes: int = 1
    ntasks_per_node: int = 48
    time_limit: str = "24:00:00"
    modules: str = "vasp/6.4.2"
    vasp_command: str = "srun vasp_std"
    potcar_root: str = "$VASP_PP_PATH/potpaw_PBE"
    job_name: str = "cluster_mlip_vasp"
    concurrent_jobs: int = 10


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "structure"


def _species_order(symbols: list[str], indices: list[int]) -> list[int]:
    """Stable-group `indices` by species in order of first appearance (POSCAR needs blocks)."""
    seen: list[str] = []
    for i in indices:
        if symbols[i] not in seen:
            seen.append(symbols[i])
    return [i for el in seen for i in indices if symbols[i] == el]


def _poscar(structure: Structure, order: list[int], title: str) -> str:
    assert structure.cell is not None
    species: list[str] = []
    counts: list[int] = []
    for i in order:
        el = structure.symbols[i]
        if species and species[-1] == el:
            counts[-1] += 1
        else:
            species.append(el)
            counts.append(1)
    lines = [title, "1.0"]
    lines += ["  " + " ".join(f"{v:18.10f}" for v in row) for row in structure.cell]
    lines += ["  " + " ".join(species), "  " + " ".join(str(c) for c in counts), "Cartesian"]
    lines += ["  " + " ".join(f"{v:18.10f}" for v in structure.positions[i]) for i in order]
    return "\n".join(lines) + "\n"


def _initial_moments(structure: Structure, mask: list[bool], multiplicity: int) -> list[float]:
    """Starting moments for every atom: explicit column if present, else M-1 spread over the cluster."""
    for key in ("initial_magmoms", "atomic_spins"):
        if key in structure.arrays:
            values = [float(v) for v in structure.arrays[key]]
            return [v if m else 0.0 for v, m in zip(values, mask)]
    carriers = [i for i, m in enumerate(mask) if m and structure.symbols[i] in MAGNETIC_ELEMENTS]
    if not carriers:
        carriers = [i for i, m in enumerate(mask) if m]
    moments = [0.0] * len(structure)
    share = (multiplicity - 1) / len(carriers) if carriers else 0.0
    for i in carriers:
        moments[i] = share
    return moments


def _magmom_line(moments: list[float]) -> str:
    """Run-length encode MAGMOM (VASP's N*value syntax)."""
    parts: list[str] = []
    k = 0
    while k < len(moments):
        j = k
        while j + 1 < len(moments) and abs(moments[j + 1] - moments[k]) < 1e-9:
            j += 1
        n = j - k + 1
        value = f"{moments[k]:.4f}".rstrip("0").rstrip(".")
        if value in ("-0", ""):
            value = "0"
        parts.append(f"{n}*{value}" if n > 1 else value)
        k = j + 1
    return " ".join(parts)


def _incar(settings: VaspSettings, *, title: str, moments: list[float], nupdown: int | None) -> str:
    tags: dict[str, str] = {
        "SYSTEM": title,
        "PREC": settings.prec,
        "ENCUT": f"{settings.encut:g}",
        "EDIFF": f"{settings.ediff:.1E}",
        "GGA": settings.gga,
        "ISMEAR": str(settings.ismear),
        "SIGMA": f"{settings.sigma:g}",
        "ISPIN": "2",
        "MAGMOM": _magmom_line(moments),
        "ISYM": "0",
        "LASPH": ".TRUE.",
        "ALGO": settings.algo,
        "NELM": str(settings.nelm),
        "LREAL": settings.lreal,
        "IBRION": "-1",
        "NSW": "0",
        "LORBIT": "11",
        "LWAVE": ".FALSE.",
        "LCHARG": ".FALSE.",
    }
    if nupdown is not None:
        tags["NUPDOWN"] = str(nupdown)
    if settings.ivdw:
        tags["IVDW"] = str(settings.ivdw)
    if settings.dipole:
        tags["LDIPOL"] = ".TRUE."
        tags["IDIPOL"] = "3"
    if settings.magnetic_mixing:
        tags.update(AMIX="0.2", BMIX="0.0001", AMIX_MAG="0.8", BMIX_MAG="0.0001")
    for key, value in settings.incar_overrides.items():
        tags[key.upper()] = value
    return "\n".join(f"{k} = {v}" for k, v in tags.items()) + "\n"


def _kpoints(settings: VaspSettings) -> str:
    grid = " ".join(str(k) for k in settings.kpoints)
    return f"Gamma-centred grid\n0\nGamma\n  {grid}\n  0 0 0\n"


def _structure_multiplicities(structure: Structure, override: list[int] | None) -> list[int]:
    if override:
        return override
    for key in ("multiplicity", "spin"):
        if key in structure.info:
            return [int(structure.info[key])]
    raise ValueError(
        f"{structure.structure_id or 'structure'}: no multiplicity in the extxyz info; "
        "pass --multiplicities"
    )


def _write_job(
    job_dir: Path, structure: Structure, order: list[int], settings: VaspSettings, *,
    title: str, moments: list[float], nupdown: int | None, meta: dict[str, Any],
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    species: list[str] = []
    for i in order:
        if not species or species[-1] != structure.symbols[i]:
            species.append(structure.symbols[i])
    write_text_lf(job_dir / "POSCAR", _poscar(structure, order, title))
    write_text_lf(job_dir / "INCAR", _incar(settings, title=title, moments=[moments[i] for i in order], nupdown=nupdown))
    write_text_lf(job_dir / "KPOINTS", _kpoints(settings))
    write_text_lf(job_dir / "POTCAR.spec", "\n".join(settings.potcar(el) for el in species) + "\n")
    write_text_lf(job_dir / "job.json", json.dumps({**meta, "order": order, "species": species}, indent=1) + "\n")


def prepare_vasp_campaign(
    structures_path: Path,
    output: Path,
    *,
    settings: VaspSettings | None = None,
    slurm: VaspSlurmConfig | None = None,
    multiplicities: list[int] | None = None,
    cluster_elements: set[str] | None = None,
) -> dict[str, Any]:
    """Write AB / A / B single-point inputs for every structure in `structures_path`."""
    settings = settings or VaspSettings()
    slurm = slurm or VaspSlurmConfig()
    structures = read_structures(structures_path)
    if not structures:
        raise ValueError(f"{structures_path}: no structures")
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    jobs: list[str] = []
    used: set[str] = set()
    clean: list[Structure] = []
    for index, s in enumerate(structures):
        if s.cell is None or not any(s.pbc):
            raise ValueError(f"structure {index}: a supported structure needs a periodic cell")
        charge = int(s.info.get("charge", 0))
        if charge:
            raise ValueError(
                f"structure {index}: charge {charge} -- charged periodic cells need NELECT and a "
                "compensating background; not supported by this generator"
            )
        mask = s.cluster_mask(cluster_elements)
        sid = _safe_name(s.structure_id or f"s{index:04d}")
        if sid in used:
            sid = f"{sid}_{index:04d}"
        used.add(sid)
        s.info["structure_id"] = sid
        s.arrays["cluster"] = [1 if m else 0 for m in mask]
        clean.append(s)
        cluster = [i for i, m in enumerate(mask) if m]
        support = [i for i, m in enumerate(mask) if not m]
        mults = _structure_multiplicities(s, multiplicities)
        root = output / "structures" / sid
        entry: dict[str, Any] = {
            "structure_id": sid, "source_index": index, "n_atoms": len(s),
            "cluster_indices": cluster, "support_indices": support,
            "multiplicities": mults, "charge": 0, "jobs": {},
        }
        support_job = root / "B"
        _write_job(
            support_job, s, _species_order(s.symbols, support), settings,
            title=f"{sid} B", moments=[0.0] * len(s), nupdown=settings.support_nupdown,
            meta={"structure_id": sid, "role": "B", "multiplicity": None},
        )
        entry["jobs"]["B"] = support_job.relative_to(output).as_posix()
        jobs.append(entry["jobs"]["B"])
        for m in mults:
            if m < 1:
                raise ValueError(f"{sid}: multiplicity {m} < 1")
            moments = _initial_moments(s, mask, m)
            for role, indices in (("AB", list(range(len(s)))), ("A", cluster)):
                job = root / f"m{m}" / role
                _write_job(
                    job, s, _species_order(s.symbols, indices), settings,
                    title=f"{sid} {role} M={m}", moments=moments, nupdown=m - 1,
                    meta={"structure_id": sid, "role": role, "multiplicity": m},
                )
                rel = job.relative_to(output).as_posix()
                entry["jobs"][f"{role}_m{m}"] = rel
                jobs.append(rel)
        manifest.append(entry)
    write_structures(clean, output / "structures.extxyz")
    settings_dict = asdict(settings)
    plan = {
        "structures": str(structures_path),
        "structures_sha256": hashlib.sha256(structures_path.read_bytes()).hexdigest(),
        "settings": settings_dict,
        "level": settings.level(),
        "slurm": asdict(slurm),
        "n_structures": len(manifest),
        "n_jobs": len(jobs),
        "entries": manifest,
    }
    write_text_lf(output / "vasp_manifest.json", json.dumps(plan, indent=1) + "\n")
    write_text_lf(output / "jobs.txt", "\n".join(jobs) + "\n")
    write_text_lf(output / "run_vasp.sbatch", _sbatch_script(slurm))
    write_text_lf(output / "submit.sh", _submit_script(len(jobs), slurm.concurrent_jobs))
    return plan


def _sbatch_script(cfg: VaspSlurmConfig) -> str:
    for name in ("partition", "account", "time_limit", "job_name"):
        value = getattr(cfg, name)
        if not value or any(c in value for c in "\r\n"):
            raise ValueError(f"{name} must be a non-empty single line")
    modules = "\n".join(f"set +u; module load {shlex.quote(m)}; set -u" for m in shlex.split(cfg.modules))
    return f"""#!/bin/bash
#SBATCH --job-name={cfg.job_name}
#SBATCH --partition={cfg.partition}
#SBATCH --account={cfg.account}
#SBATCH --nodes={cfg.nodes}
#SBATCH --ntasks-per-node={cfg.ntasks_per_node}
#SBATCH --time={cfg.time_limit}
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Generated by `cluster-mlip vasp-prepare`. One array task = one single point (jobs.txt line).
set -euo pipefail
{modules}
CAMPAIGN="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
[ -n "${{SLURM_SUBMIT_DIR:-}}" ] && [ -f "$SLURM_SUBMIT_DIR/jobs.txt" ] && CAMPAIGN="$SLURM_SUBMIT_DIR"
POTCAR_ROOT="${{POTCAR_ROOT:-{cfg.potcar_root}}}"
VASP_CMD="${{VASP_CMD:-{cfg.vasp_command}}}"
mapfile -t JOBS < "$CAMPAIGN/jobs.txt"
JOB="${{JOBS[$SLURM_ARRAY_TASK_ID]}}"
cd "$CAMPAIGN/$JOB"
if [ -f DONE ]; then echo "$JOB already done"; exit 0; fi
rm -f FAILED POTCAR
while read -r name; do
  [ -z "$name" ] && continue
  if [ ! -f "$POTCAR_ROOT/$name/POTCAR" ]; then
    echo "missing $POTCAR_ROOT/$name/POTCAR" > FAILED; exit 1
  fi
  cat "$POTCAR_ROOT/$name/POTCAR" >> POTCAR
done < POTCAR.spec
set +e
$VASP_CMD > vasp.out 2>&1
status=$?
set -e
if [ $status -eq 0 ] && grep -q "</modeling>" vasprun.xml 2>/dev/null; then
  touch DONE
else
  echo "exit $status" > FAILED
  exit 1
fi
"""


def _submit_script(n_jobs: int, concurrent: int) -> str:
    return f"""#!/bin/bash
# Submit every VASP single point as one Slurm array ({n_jobs} tasks, at most {concurrent} at once).
# Finished tasks (DONE marker) exit immediately, so re-submitting after failures is safe.
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
mkdir -p logs
sbatch --array=0-{n_jobs - 1}%{concurrent} run_vasp.sbatch
"""


# --------------------------------------------------------------------------- collect
@dataclass
class VaspResult:
    completed: bool
    converged: bool
    energy_free: float | None = None
    energy_sigma0: float | None = None
    forces: list[tuple[float, float, float]] = field(default_factory=list)
    n_scsteps: int = 0
    nelm: int | None = None
    magnetization: float | None = None
    atomic_magnetizations: list[float] | None = None
    reason: str = ""


def parse_vasprun(path: Path) -> VaspResult:
    """Energy, forces and electronic convergence of the last ionic step of a vasprun.xml."""
    if not path.is_file():
        return VaspResult(False, False, reason="no vasprun.xml")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return VaspResult(False, False, reason="truncated vasprun.xml")
    calcs = root.findall("calculation")
    if not calcs:
        return VaspResult(False, False, reason="no <calculation> in vasprun.xml")
    last = calcs[-1]
    nelm = None
    params = root.find("parameters")
    if params is not None:
        for item in params.iter("i"):
            if item.get("name") == "NELM" and item.text:
                nelm = int(float(item.text))
                break
    n_sc = len(last.findall("scstep"))
    energies = {}
    block = last.find("energy")
    if block is not None:
        for item in block.findall("i"):
            if item.text:
                energies[item.get("name")] = float(item.text)
    forces: list[tuple[float, float, float]] = []
    for varray in last.findall("varray"):
        if varray.get("name") == "forces":
            for v in varray.findall("v"):
                x, y, z = (float(t) for t in (v.text or "").split())
                forces.append((x, y, z))
    if "e_fr_energy" not in energies or not forces:
        return VaspResult(True, False, n_scsteps=n_sc, nelm=nelm, reason="no energy/forces")
    converged = nelm is None or n_sc < nelm
    return VaspResult(
        True, converged,
        energy_free=energies["e_fr_energy"],
        energy_sigma0=energies.get("e_0_energy", energies.get("e_wo_entrp")),
        forces=forces, n_scsteps=n_sc, nelm=nelm,
        reason="" if converged else f"SCF hit NELM={nelm}",
    )


_MAG_RE = re.compile(r"mag=\s*(-?[\d.Ee+-]+)")


def parse_oszicar_magnetization(path: Path) -> float | None:
    if not path.is_file():
        return None
    value = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _MAG_RE.search(line)
        if match and " F=" in line:
            value = float(match.group(1))
    return value


def parse_outcar_magnetization(path: Path, n_atoms: int) -> list[float] | None:
    """The 'tot' column of the last 'magnetization (x)' table in OUTCAR."""
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = None
    for k, line in enumerate(lines):
        if line.strip().startswith("magnetization (x)"):
            start = k
    if start is None:
        return None
    values: list[float] = []
    for line in lines[start + 1:]:
        tokens = line.split()
        if not tokens or not tokens[0].isdigit():
            if values:
                break
            continue
        values.append(float(tokens[-1]))
        if len(values) == n_atoms:
            break
    return values if len(values) == n_atoms else None


def _read_job(job_dir: Path) -> tuple[dict[str, Any], VaspResult]:
    meta = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    result = parse_vasprun(job_dir / "vasprun.xml")
    if (job_dir / "FAILED").is_file() and not result.completed:
        result.reason = "FAILED: " + (job_dir / "FAILED").read_text(encoding="utf-8").strip()
    if result.completed:
        order = meta["order"]
        if len(result.forces) != len(order):
            result.converged = False
            result.reason = f"{len(result.forces)} forces for {len(order)} atoms"
        result.magnetization = parse_oszicar_magnetization(job_dir / "OSZICAR")
        result.atomic_magnetizations = parse_outcar_magnetization(job_dir / "OUTCAR", len(order))
    return meta, result


def _scatter(n: int, order: list[int], values: list[Any], fill: Any) -> list[Any]:
    out = [fill] * n
    for k, i in enumerate(order):
        out[i] = values[k]
    return out


def collect_vasp_campaign(
    campaign: Path, output: Path, *, energy: str = "free", magnetization_tolerance: float = 0.1,
) -> dict[str, Any]:
    """Assemble interaction (dE_int), total and fragment frames from a finished campaign."""
    if energy not in {"free", "sigma0"}:
        raise ValueError("energy must be 'free' or 'sigma0'")
    plan = json.loads((campaign / "vasp_manifest.json").read_text(encoding="utf-8"))
    structures = {s.info["structure_id"]: s for s in read_structures(campaign / "structures.extxyz")}
    level = plan["level"]
    output.mkdir(parents=True, exist_ok=True)
    interaction: list[Structure] = []
    totals: list[Structure] = []
    clusters: list[Structure] = []
    supports: list[Structure] = []
    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, str]] = []

    def pick(r: VaspResult) -> float:
        value = r.energy_free if energy == "free" else r.energy_sigma0
        assert value is not None
        return value

    def frame(s: Structure, subset: list[int] | None, e: float, forces: list[Any], config_type: str,
              m: int, extra: dict[str, Any]) -> Structure:
        idx = subset if subset is not None else list(range(len(s)))
        info = {
            "structure_id": s.info["structure_id"], "config_type": config_type,
            "charge": 0, "spin": m, "multiplicity": m, "total_spin": m - 1,
            "REF_energy": e, "label_level": level, "method": "VASP",
        }
        for key in ("support", "isomer", "source"):
            if key in s.info:
                info[key] = s.info[key]
        info.update(extra)
        arrays = {"REF_forces": [forces[i] for i in idx], "cluster": [s.arrays["cluster"][i] for i in idx]}
        return Structure([s.symbols[i] for i in idx], [s.positions[i] for i in idx], s.cell, s.pbc, info, arrays)

    for entry in plan["entries"]:
        sid = entry["structure_id"]
        s = structures[sid]
        n = len(s)
        cl, sp = entry["cluster_indices"], entry["support_indices"]
        b_meta, b_res = _read_job(campaign / entry["jobs"]["B"])
        if not (b_res.completed and b_res.converged):
            failures.append((sid, entry["jobs"]["B"], b_res.reason or "not converged"))
        else:
            fb = _scatter(n, b_meta["order"], b_res.forces, (0.0, 0.0, 0.0))
            b_mult = 1 + (plan["settings"]["support_nupdown"] or 0)
            supports.append(frame(s, sp, pick(b_res), fb, "support_frozen", b_mult, {}))
        for m in entry["multiplicities"]:
            row: dict[str, Any] = {"structure_id": sid, "multiplicity": m, "status": "ok"}
            results = {}
            for role in ("AB", "A"):
                meta, res = _read_job(campaign / entry["jobs"][f"{role}_m{m}"])
                results[role] = (meta, res)
                row[f"{role}_scsteps"] = res.n_scsteps
                row[f"{role}_mag"] = res.magnetization
                if not (res.completed and res.converged):
                    failures.append((sid, entry["jobs"][f"{role}_m{m}"], res.reason or "not converged"))
                    row["status"] = f"{role}: {res.reason or 'not converged'}"
                elif res.magnetization is not None and abs(res.magnetization - (m - 1)) > magnetization_tolerance:
                    failures.append((sid, entry["jobs"][f"{role}_m{m}"], f"mag {res.magnetization} != {m - 1}"))
                    row["status"] = f"{role}: magnetization {res.magnetization} != {m - 1}"
            ab_meta, ab = results["AB"]
            a_meta, a = results["A"]
            if row["status"] == "ok" and not (b_res.completed and b_res.converged):
                row["status"] = f"B: {b_res.reason or 'not converged'}"
            if row["status"] != "ok":
                rows.append(row)
                continue
            zero = (0.0, 0.0, 0.0)
            f_ab_full = _scatter(n, ab_meta["order"], ab.forces, zero)
            f_a_full = _scatter(n, a_meta["order"], a.forces, zero)
            f_b_full = _scatter(n, b_meta["order"], b_res.forces, zero)
            total = frame(s, None, pick(ab), f_ab_full, "supported_total", m, {})
            if ab.atomic_magnetizations is not None:
                total.arrays["vasp_magmom"] = _scatter(n, ab_meta["order"], ab.atomic_magnetizations, 0.0)
            totals.append(total)
            clusters.append(frame(s, cl, pick(a), f_a_full, "cluster_frozen", m, {}))
            e_ab, e_a, e_b = pick(ab), pick(a), pick(b_res)
            d_forces = [
                tuple(f_ab_full[i][k] - (f_a_full[i][k] if s.arrays["cluster"][i] else f_b_full[i][k]) for k in range(3))
                for i in range(n)
            ]
            e_int = e_ab - e_a - e_b
            interaction.append(frame(
                s, None, e_int, d_forces, "support_interaction", m,
                {"E_AB": e_ab, "E_A": e_a, "E_B": e_b},
            ))
            row.update(E_AB=e_ab, E_A=e_a, E_B=e_b, E_int=e_int,
                       max_dF=max(math.sqrt(sum(c * c for c in f)) for f in d_forces))
            rows.append(row)

    write_structures(interaction, output / "interaction.extxyz")
    write_structures(totals, output / "supported_total.extxyz")
    write_structures(clusters, output / "cluster_frozen.extxyz")
    write_structures(supports, output / "support_frozen.extxyz")
    columns = ["structure_id", "multiplicity", "status", "E_AB", "E_A", "E_B", "E_int", "max_dF",
               "AB_scsteps", "A_scsteps", "AB_mag", "A_mag"]
    with (output / "interaction_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "failed_jobs.tsv").open("w", newline="", encoding="utf-8") as handle:
        handle.write("structure_id\tjob\treason\n")
        for failure in failures:
            handle.write("\t".join(failure) + "\n")
    summary = {
        "campaign": str(campaign), "level": level, "energy": energy,
        "structures": len(plan["entries"]),
        "interaction_frames": len(interaction), "total_frames": len(totals),
        "cluster_frames": len(clusters), "support_frames": len(supports),
        "failed_jobs": len(failures),
    }
    write_text_lf(output / "collect_summary.json", json.dumps(summary, indent=1) + "\n")
    return summary
