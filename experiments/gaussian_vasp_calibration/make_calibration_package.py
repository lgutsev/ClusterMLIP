"""Stage the Gaussian-vs-VASP calibration package (see docs/gaussian-vasp-calibration.md).

Picks ~30 neutral Fe_n(N/O) geometries (Fe2-Fe16; minima, off-equilibrium optimisation frames,
legacy TS/IRC frames and two seeded rattles), adds vertical spin ladders, and writes

  gaussian/inputs/frame_NNNN.gjf   one Gaussian task per (geometry, M): UBPW91 > UPW91PW91 > UPBEPBE,
                                   Force in every link, Guess=Read after the first, all at
                                   6-311++G* with IOP(5/13=1,5/36=1,8/11=1) in every link
  vasp/inputs/<job>/               INCAR/POSCAR/KPOINTS/POTCAR.spec written by the existing
                                   `cluster_mlip.vasp.prepare_vasp_campaign` (its `A` role = the
                                   cluster alone in the box); only those job directories are kept
  states.csv, calibration_set.extxyz, run_*.slurm, submit.sh, package.json

Nothing is submitted. The VASP generator lives on feat/periodic-vasp-ingest (or
feat/supported-delta-vasp); pass its src directory with --cluster-mlip-src.

    python experiments/gaussian_vasp_calibration/make_calibration_package.py \
        --cluster-mlip-src .claude/worktrees/periodic-vasp-ingest/src \
        --spin-dataset D:/MLIP_Work_Folder/Cluster_MLIP/FenOm_Warehouse2/dataset_spin_v0/all.extxyz \
        --legacy D:/MLIP_Work_Folder/Cluster_MLIP/Gutsev_Legacy_Warehouse/extracted \
        -o D:/MLIP_Work_Folder/Cluster_MLIP/loni_package/56_clustermlip_gaussian_vasp_calibration
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HA_BOHR_TO_EV_A = 51.42208619083232
BASIS = "6-311++G*"
SCF = "SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)"
IOP = "IOP(5/13=1,5/36=1,8/11=1)"  # every link; never drop it
# G09 D.01 by default: the archived labels are G09, and the user keeps these systems on G09 (2026-10-16)
PROFILES = {
    "g09": dict(module="gaussian/g09-d01", command="g09"),
    "g16": dict(module="gaussian/g16-c01", command="g16"),
}
LINKS = (  # (label, Gaussian functional keyword)
    ("BPW91", "UBPW91"),
    ("PW91", "UPW91PW91"),
    ("PBE", "UPBEPBE"),
)

# --- selection ---------------------------------------------------------------------------------
# dataset_spin_v0 (G09 UBPW91/6-311++G*, labels + Mulliken spins); neutral only (vasp-prepare
# refuses charged cells). offeq = also take one mid-optimisation frame at the minimum's M.
SPIN_PICKS = {"Fe2N": False, "Fe2NO": False, "Fe2N2O2": True, "Fe2N2O4": True, "Fe16": True, "Fe16N2": True}
# Gutsev legacy warehouse: lowest-energy record of the formula at its best-covered basis.
LEGACY_MINIMA = ["Fe2", "Fe2O2", "Fe3N2", "Fe4", "Fe4N2", "Fe5N2", "Fe6N", "Fe7N", "Fe8", "Fe8N2",
                 "Fe9N", "Fe10", "Fe10N2", "Fe12", "Fe16NO2"]
LEGACY_OFFEQ = [("Fe2N2O4", "transition_state"), ("Fe5N2", "transition_state"),
                ("Fe4N2", "irc_forward"), ("Fe16N2", "irc_forward")]
RATTLES = [("Fe8", 0.08), ("Fe16N2", 0.08)]  # (formula of a selected minimum, sigma in Angstrom)
LADDERS = ["Fe2N", "Fe2N2O2", "Fe4N2", "Fe8", "Fe16", "Fe16N2"]  # minimum also at M-2 and M+2
# box/ENCUT/PREC convergence and the LIBXC BPW91 probe run on these minima (PBE only for convergence)
CONVERGENCE = ["Fe2N2O2", "Fe8", "Fe16N2"]

PROD_GAP = 12.0  # Angstrom between the cluster and its nearest periodic image
CONV_GAPS = (8.0, 10.0, 14.0)
PROD_ENCUT = 520.0
CONV_ENCUTS = (400.0, 650.0)
FORMULA = re.compile(r"^Fe(\d+)((?:[NO]\d*){0,2})$")


def fe_count(formula: str) -> int:
    m = re.match(r"Fe(\d+)", formula)
    return int(m.group(1)) if m else 0


def force_rms(forces) -> float:
    return math.sqrt(sum(c * c for f in forces for c in f) / (3 * len(forces)))


def fmax(forces) -> float:
    return max(math.sqrt(sum(c * c for c in f)) for f in forces)


def pick_spin_dataset(path: Path, periodic) -> list[dict]:
    frames = periodic.read_structures(path)
    by_formula: dict[str, list] = defaultdict(list)
    for s in frames:
        f = str(s.info.get("formula", ""))
        if f in SPIN_PICKS and int(s.info.get("charge", 0)) == 0:
            by_formula[f].append(s)
    picks = []
    for formula, offeq in SPIN_PICKS.items():
        rows = by_formula[formula]
        spins = lambda s: [float(v[2]) for v in s.info.get("metadata", {}).get("atomic_spins", [])]
        def collinear(s):  # every Fe moment up: the state a default guess and a uniform MAGMOM both find
            sp = spins(s)
            return len(sp) == len(s) and min(v for v, el in zip(sp, s.symbols) if el == "Fe") > -0.5
        def has_spins(s):
            return len(spins(s)) == len(s)

        converged = [s for s in rows if fmax(s.arrays["REF_forces"]) < 0.03]
        # prefer an all-Fe-up state (both codes reach it from an unbiased start), then any state whose
        # spins were recorded, then any converged frame
        minima = ([s for s in converged if has_spins(s) and collinear(s)]
                  or [s for s in converged if has_spins(s)] or converged)
        if not minima:
            raise SystemExit(f"{formula}: no converged frame in {path}")
        best = min(minima, key=lambda s: (float(s.info["REF_energy"]), str(s.info["record_id"])))
        m = int(best.info["multiplicity"])
        picks.append(_from_structure(best, "minimum", "dataset_spin_v0", spins(best), collinear(best) if has_spins(best) else None))
        if offeq:
            cands = [s for s in rows if int(s.info["multiplicity"]) == m and 0.2 <= force_rms(s.arrays["REF_forces"]) <= 1.0]
            cands = [s for s in cands if has_spins(s)] or cands
            if cands:
                mid = min(cands, key=lambda s: (abs(force_rms(s.arrays["REF_forces"]) - 0.5), str(s.info["record_id"])))
                picks.append(_from_structure(mid, "offeq_opt_frame", "dataset_spin_v0", spins(mid),
                                             collinear(mid) if has_spins(mid) else None))
    return picks


def _from_structure(s, role: str, source: str, spins: list[float], collinear: bool | None) -> dict:
    return {
        "source_collinear": "" if collinear is None else collinear,
        "formula": str(s.info["formula"]), "multiplicity": int(s.info["multiplicity"]), "role": role,
        "source": source, "source_record_id": str(s.info["record_id"]),
        "source_file": str(s.info.get("source", "")),
        "symbols": list(s.symbols), "positions": [tuple(p) for p in s.positions],
        "spins": spins, "ref_energy_ev": float(s.info["REF_energy"]),
        "ref_forces": [tuple(f) for f in s.arrays["REF_forces"]],
    }


def pick_legacy(root: Path) -> list[dict]:
    wanted_min = set(LEGACY_MINIMA)
    wanted_off = {f: kind for f, kind in LEGACY_OFFEQ}
    rows: dict[str, list[dict]] = defaultdict(list)
    with (root / "manifest.csv").open(encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            f = r["formula"]
            if r["charge"] != "0" or (f not in wanted_min and f not in wanted_off):
                continue
            route = r["route"].upper()
            b = re.search(r"BPW91/(\S+)", route)
            # 6-311 family only (GEN/CHKBAS records carry unknown bases); M = 1 would need a
            # broken-symmetry guess that neither code's default start reproduces
            if not b or not b.group(1).startswith("6-311") or not r["legacy_energy_hartree"] or int(r["multiplicity"]) < 2:
                continue
            r["basis"] = b.group(1)
            rows[f].append(r)
    chosen: dict[str, tuple[str, dict]] = {}
    for f in wanted_min:
        cands = [r for r in rows[f] if r["config_type"] in ("minimum", "optimized_unverified")]
        chosen[f] = ("minimum", _lowest_at_best_basis(f, cands))
    for f, kind in wanted_off.items():
        cands = [r for r in rows[f] if r["config_type"] == kind]
        chosen[f + ":" + kind] = ("legacy_" + kind, _lowest_at_best_basis(f, cands))
    ids = {r["record_id"]: key for key, (_, r) in chosen.items()}
    geoms = _read_legacy_geometries(root / "seeds.extxyz", set(ids))
    picks = []
    for key, (role, r) in chosen.items():
        symbols, positions = geoms[r["record_id"]]
        picks.append({
            "formula": r["formula"], "multiplicity": int(r["multiplicity"]), "role": role,
            "source": "gutsev_legacy", "source_record_id": r["record_id"], "source_file": r["source"],
            "legacy_basis": r["basis"], "source_collinear": "", "symbols": symbols, "positions": positions, "spins": [],
            "ref_energy_ev": float(r["legacy_energy_hartree"]) * 27.211386245988, "ref_forces": None,
        })
    return picks


def _lowest_at_best_basis(formula: str, cands: list[dict]) -> dict:
    if not cands:
        raise SystemExit(f"legacy: no candidate records for {formula}")
    counts: dict[str, int] = defaultdict(int)
    for r in cands:
        counts[r["basis"]] += 1
    # prefer the production basis, else the best-covered one (energies only compare within a basis)
    basis = BASIS.upper() if BASIS.upper() in counts else max(counts, key=lambda b: (counts[b], b))
    return min((r for r in cands if r["basis"] == basis), key=lambda r: (float(r["legacy_energy_hartree"]), r["record_id"]))


def _read_legacy_geometries(path: Path, ids: set[str]) -> dict[str, tuple[list[str], list[tuple]]]:
    out: dict[str, tuple[list[str], list[tuple]]] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        while len(out) < len(ids):
            head = fh.readline()
            if not head:
                break
            if not head.strip():
                continue
            n = int(head.split()[0])
            info = fh.readline()
            m = re.search(r'record_id="?([0-9a-f]+)', info)
            lines = [fh.readline() for _ in range(n)]
            if m and m.group(1) in ids:
                toks = [ln.split() for ln in lines]
                out[m.group(1)] = ([t[0] for t in toks], [tuple(float(x) for x in t[1:4]) for t in toks])
    missing = ids - set(out)
    if missing:
        raise SystemExit(f"legacy seeds: geometries not found for {sorted(missing)}")
    return out


def rattle(pick: dict, sigma: float, seed: int) -> dict:
    rng = random.Random(f"{pick['source_record_id']}:{sigma}:{seed}")
    pos = [tuple(c + rng.gauss(0.0, sigma) for c in p) for p in pick["positions"]]
    return {**pick, "positions": pos, "role": f"rattle_{sigma:g}A", "ref_energy_ev": None, "ref_forces": None}


def states_from(picks: list[dict]) -> list[dict]:
    states = []
    for g, p in enumerate(picks):
        ladder = [p["multiplicity"]]
        if p["role"] == "minimum" and p["formula"] in LADDERS:
            ladder = [m for m in (p["multiplicity"] - 2, p["multiplicity"], p["multiplicity"] + 2) if m >= 1]
        for m in ladder:
            states.append({**p, "geometry_id": f"g{g:02d}", "multiplicity": m, "vertical_from": p["multiplicity"]
                           if m != p["multiplicity"] else ""})
    for k, s in enumerate(states):
        s["state_id"] = f"s{k:03d}_{s['formula']}_m{s['multiplicity']}_{s['role'].split('_')[0]}"
    return states


# --- writers -----------------------------------------------------------------------------------
def gjf(state: dict, frame: str) -> str:
    coords = "\n".join(f"{el:<2} {x:16.8f} {y:16.8f} {z:16.8f}" for el, (x, y, z) in zip(state["symbols"], state["positions"]))
    blocks = []
    for i, (label, functional) in enumerate(LINKS):
        guess = "" if i == 0 else " Geom=Checkpoint Guess=Read"
        route = f"#p {functional}/{BASIS} {SCF} NoSymm Force {IOP} Int=UltraFine Pop=Hirshfeld{guess}"
        head = f"%chk={frame}.chk\n%nprocshared=12\n%mem=24GB\n{route}\n\n{state['state_id']} {label}\n\n0 {state['multiplicity']}\n"
        blocks.append(head + (coords + "\n" if i == 0 else "") + "\n")
    return "--Link1--\n".join(blocks)


def boxed(state: dict, gap: float, periodic):
    xs = list(zip(*state["positions"]))
    extent = max(max(c) - min(c) for c in xs)
    length = round(extent + gap, 2)
    centre = [(max(c) + min(c)) / 2 for c in xs]
    pos = [tuple(p[k] - centre[k] + length / 2 for k in range(3)) for p in state["positions"]]
    cell = [(length, 0.0, 0.0), (0.0, length + 0.13, 0.0), (0.0, 0.0, length + 0.27)]  # no cubic degeneracy
    arrays = {"cluster": [1] * len(pos)}
    if state["spins"] and state.get("source_collinear") is True:  # else vasp.py's uniform M-1 start
        arrays["initial_magmoms"] = [round(v, 3) for v in _scaled_spins(state)]
    info = {"structure_id": state["state_id"], "multiplicity": state["multiplicity"], "charge": 0,
            "formula": state["formula"], "geometry_id": state["geometry_id"], "role": state["role"],
            "vacuum_gap": gap}
    return periodic.Structure(state["symbols"], pos, cell, (True, True, True), info, arrays)


def _scaled_spins(state: dict) -> list[float]:
    """Gaussian Mulliken spins rescaled to the target M - 1 (vertical ladder members reuse them)."""
    spins = state["spins"]
    total = sum(spins)
    if abs(total) < 1e-6:
        return spins
    return [v * (state["multiplicity"] - 1) / total for v in spins]


def vasp_variants() -> list[tuple[str, float, dict]]:
    """(variant, vacuum gap, VaspSettings kwargs). prod_* run on every state; the rest on CONVERGENCE."""
    base = dict(encut=PROD_ENCUT, ediff=1e-6, prec="Accurate", ismear=0, sigma=0.01, ivdw=0,
                lreal=".FALSE.", dipole=False, incar_overrides={"NCORE": "8"})
    out = [("prod_pw91", PROD_GAP, {**base, "gga": "91"}), ("prod_pbe", PROD_GAP, {**base, "gga": "PE"})]
    for gap in CONV_GAPS:
        out.append((f"conv_pbe_gap{gap:g}", gap, {**base, "gga": "PE"}))
    for encut in CONV_ENCUTS:
        out.append((f"conv_pbe_encut{encut:g}", PROD_GAP, {**base, "gga": "PE", "encut": encut}))
    out.append(("conv_pbe_precnormal", PROD_GAP, {**base, "gga": "PE", "prec": "Normal"}))
    # BPW91 itself through Libxc (B88 exchange + PW91 correlation): a probe; it fails if the module
    # was built without Libxc, which is a result, not a package failure
    out.append(("probe_libxc_bpw91", PROD_GAP, {**base, "gga": "LIBXC",
                                                 "incar_overrides": {"NCORE": "8", "LIBXC1": "GGA_X_B88", "LIBXC2": "GGA_C_PW91"}}))
    return out


def stage_vasp(states: list[dict], out: Path, periodic, vasp) -> list[dict]:
    jobs = []
    conv_states = [s for s in states if s["role"] == "minimum" and s["formula"] in CONVERGENCE and not s["vertical_from"]]
    for variant, gap, kwargs in vasp_variants():
        subset = states if variant.startswith("prod_") else conv_states
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "structures.extxyz"
            periodic.write_structures([boxed(s, gap, periodic) for s in subset], src)
            plan = vasp.prepare_vasp_campaign(src, Path(tmp) / "campaign", settings=vasp.VaspSettings(**kwargs))
            for entry, s in zip(plan["entries"], subset):
                m = s["multiplicity"]
                rel = entry["jobs"][f"A_m{m}"]  # the cluster alone; AB is identical and B is empty here
                job = f"{variant}/{s['state_id']}"
                shutil.copytree(Path(tmp) / "campaign" / rel, out / "vasp" / "inputs" / job)
                jobs.append({"job": job, "variant": variant, "state_id": s["state_id"], "binary": "vasp_gam",
                             "level": plan["level"]})
    return jobs


RUN_GAUSSIAN = """#!/bin/bash
#SBATCH --job-name=gv-calib-@COMMAND@
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Written by ClusterMLIP experiments/gaussian_vasp_calibration/make_calibration_package.py. One array task
# per frame: UBPW91 > UPW91PW91 > UPBEPBE single points (Force) in work/frame_NNNN; log to outputs/.
# Memory comes from each input's %mem (24GB). The module function reads unset variables: -u after it.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the package directory}"
PKG=$PWD
frame=$(printf 'frame_%04d' "$SLURM_ARRAY_TASK_ID")
mkdir -p gaussian/outputs "gaussian/work/$frame"
module load @MODULE@
set -u
scratch="${TMPDIR:-/tmp}/@COMMAND@_${SLURM_JOB_ID}_${frame}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
cd "gaussian/work/$frame"
start=$(date +%s)
status=0
GAUSS_SCRDIR="$scratch" @COMMAND@ < "$PKG/gaussian/inputs/$frame.gjf" > "$PKG/gaussian/outputs/$frame.log" || status=$?
printf '%s\\n' "$(( $(date +%s) - start ))" > "$PKG/gaussian/outputs/$frame.seconds"
exit "$status"
"""

RUN_VASP = """#!/bin/bash
#SBATCH --job-name=gv-calib-vasp
#SBATCH --account=loni_perovsk27
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Written by ClusterMLIP experiments/gaussian_vasp_calibration/make_calibration_package.py. One array task
# per line of vasp/jobs.txt ("<variant>/<state> <binary>"): a single point (IBRION=-1, NSW=0) run in
# vasp/runs/<job>; vasprun.xml, OUTCAR, OSZICAR, INCAR go to vasp/outputs/<job>. Inputs come from
# cluster_mlip.vasp.prepare_vasp_campaign (role A: the cluster alone in a box). The module function
# reads unset variables: -u only after it.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the package directory}"
PKG=$PWD
module purge
module load vasp6/6.6.1-cpu
set -u
export SINGULARITYENV_OMP_NUM_THREADS=1
POTCARS=${POTCARS:-/home/lgutsev/pot/potpaw_PBE}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" vasp/jobs.txt)
job=$(echo "$line" | awk '{print $1}')
bin=$(echo "$line" | awk '{print $2}')
[ -n "$job" ] || { echo "no job for task $SLURM_ARRAY_TASK_ID"; exit 2; }
vasp=$(command -v "$bin" || true)
case "$vasp" in
  "") echo "no $bin after module load vasp6/6.6.1-cpu" >&2; exit 2 ;;
  "$HOME"/bin/*) echo "$bin resolves to $vasp, not the module; check PATH" >&2; exit 2 ;;
esac
work="vasp/runs/$job"; out="vasp/outputs/$job"
mkdir -p "$work" "$out"
if [ -f "$work/DONE" ]; then echo "$job already done"; exit 0; fi
trap 'code=$?; printf "%s\\n" "$code" > "$PKG/$out/exit_code"' EXIT
{ echo "$bin: $vasp"; echo "LOADEDMODULES: ${LOADEDMODULES:-}"; } > "$out/vasp_binary.txt"
while read -r el; do
  [ -z "$el" ] && continue
  [ -s "$POTCARS/$el/POTCAR" ] || { echo "missing $POTCARS/$el/POTCAR" >&2; exit 2; }
done < "vasp/inputs/$job/POTCAR.spec"
: > "$work/POTCAR.tmp"
while read -r el; do [ -n "$el" ] && cat "$POTCARS/$el/POTCAR" >> "$work/POTCAR.tmp"; done < "vasp/inputs/$job/POTCAR.spec"
cp "vasp/inputs/$job/INCAR" "vasp/inputs/$job/POSCAR" "vasp/inputs/$job/KPOINTS" "$work/"
mv "$work/POTCAR.tmp" "$work/POTCAR"
cd "$work"
start=$(date +%s)
status=0
srun "$vasp" > vasp.out 2>&1 || status=$?
printf '%s\\n' "$(( $(date +%s) - start ))" > "$PKG/$out/seconds"
cp vasprun.xml OUTCAR OSZICAR INCAR vasp.out "$PKG/$out/" 2>/dev/null || true
if [ $status -eq 0 ] && grep -q "</modeling>" vasprun.xml 2>/dev/null; then touch DONE; fi
exit "$status"
"""

SUBMIT = """#!/bin/bash
# Submit both arrays (nothing else). Re-running is safe: finished VASP jobs (DONE) exit at once.
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
mkdir -p logs
sbatch --array=0-{n_g}%20 run_gaussian.slurm
sbatch --array=0-{n_v}%8 run_vasp.slurm
"""


README = """# {title}: Gaussian UBPW91 vs VASP calibration (ClusterMLIP)

This package measures how UBPW91 labels differ from VASP labels for the same neutral Fe_n(N/O) cluster
geometries (Fe2-Fe16), to decide whether VASP data can join MACE training as the same head, a separate
head, or only as a delta. Plan, metrics and decision rule: ClusterMLIP `docs/gaussian-vasp-calibration.md`.

| part | tasks | where | per task |
|---|---|---|---|
| Gaussian | {n_g} (`gaussian/inputs/frame_0000..{last:04d}.gjf`) | QB4 `single`, `{module}` (`{command}`) | 12 cores, 24 GB, 12 h |
| VASP | {n_v} (`vasp/jobs.txt`) | QB4 `workq`, `vasp6/6.6.1-cpu`, `vasp_gam` | 64 cores, 12 h, at most 8 at once |

- **Gaussian:** three links per task on one geometry: UBPW91, then UPW91PW91, then UPBEPBE, each Force at
  6-311++G* with IOP(5/13=1,5/36=1,8/11=1). Links 2 and 3 read link 1's guess.
- **VASP:** single points with NUPDOWN = M - 1: `prod_pw91` and `prod_pbe` on every state; box/ENCUT/PREC
  convergence variants and `probe_libxc_bpw91` on 3 states each.

`states.csv` maps each frame to its state and source record; `calibration_set.extxyz` holds the boxed geometries.

## Run

From the package directory: `bash submit.sh`. POTCARs come from `$POTCARS` (default
`/home/lgutsev/pot/potpaw_PBE`; Fe_pv, N, O).

## Done

- every `gaussian/outputs/frame_*.log` has 3 `Normal termination` lines and a `.seconds` file;
- every `vasp/outputs/<variant>/<state>/` has `vasprun.xml` with `</modeling>`, OUTCAR and OSZICAR.

`probe_libxc_bpw91` failing without Libxc, and an unconverged or odd-spin SCF, are results, not package
failures. If a task hits 12 h, report its time and last SCF cycle; do not resubmit with a longer limit
without asking.
"""


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cluster-mlip-src", required=True, help="src/ of a checkout that has cluster_mlip/vasp.py")
    ap.add_argument("--spin-dataset", required=True)
    ap.add_argument("--legacy", required=True, help="Gutsev_Legacy_Warehouse/extracted (manifest.csv + seeds.extxyz)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--gaussian", choices=sorted(PROFILES), default="g09", help="QB4 Gaussian module profile")
    ap.add_argument("--desk-number", type=int, help="the number the dispatch desk assigned (README and package.json); "
                    "unset = a proposal, taken from the output directory name")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(args.cluster_mlip_src).resolve()))
    from cluster_mlip import periodic, vasp  # noqa: E402  (branch-only module)

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    picks = pick_spin_dataset(Path(args.spin_dataset), periodic) + pick_legacy(Path(args.legacy))
    picks.sort(key=lambda p: (fe_count(p["formula"]), p["formula"], p["role"]))
    by_formula = {p["formula"]: p for p in picks if p["role"] == "minimum"}
    picks += [rattle(by_formula[f], sigma, args.seed) for f, sigma in RATTLES]
    states = states_from(picks)

    frames = []
    for k, s in enumerate(states):
        frame = f"frame_{k:04d}"
        write_lf(out / "gaussian" / "inputs" / f"{frame}.gjf", gjf(s, frame))
        frames.append(frame)
    jobs = stage_vasp(states, out, periodic, vasp)
    write_lf(out / "vasp" / "jobs.txt", "".join(f"{j['job']} {j['binary']}\n" for j in jobs))

    fields = ["frame", "state_id", "geometry_id", "formula", "n_fe", "multiplicity", "vertical_from", "role",
              "source", "source_record_id", "source_file", "legacy_basis", "ref_energy_ev", "ref_fmax_ev_a", "has_spins", "source_collinear"]
    with (out / "states.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for frame, s in zip(frames, states):
            w.writerow({**{k: s.get(k, "") for k in fields}, "frame": frame, "n_fe": fe_count(s["formula"]),
                        "ref_energy_ev": "" if s["vertical_from"] or s["ref_energy_ev"] is None else f"{s['ref_energy_ev']:.6f}",
                        "ref_fmax_ev_a": f"{fmax(s['ref_forces']):.4f}" if s["ref_forces"] and not s["vertical_from"] else "",
                        "has_spins": bool(s["spins"])})
    periodic.write_structures([boxed(s, PROD_GAP, periodic) for s in states], out / "calibration_set.extxyz")
    profile = PROFILES[args.gaussian]
    write_lf(out / "run_gaussian.slurm",
             RUN_GAUSSIAN.replace("@MODULE@", profile["module"]).replace("@COMMAND@", profile["command"]))
    write_lf(out / "run_vasp.slurm", RUN_VASP)
    write_lf(out / "submit.sh", SUBMIT.format(n_g=len(frames) - 1, n_v=len(jobs) - 1))
    write_lf(out / "export.list", "gaussian/outputs/*\nvasp/outputs/*\n")
    write_lf(out / "logs" / "README.txt", "Slurm logs land here.\n")
    desk = args.desk_number
    write_lf(out / "README.md", README.format(
        title=f"Package {desk}" if desk is not None else f"Package {out.name.split('_')[0]} (proposed)",
        module=profile["module"], command=profile["command"], n_g=len(frames), last=len(frames) - 1, n_v=len(jobs)))
    with (out / "vasp" / "jobs.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(jobs[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(jobs)
    generator = Path(__file__).resolve()
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob("*")) if p.is_file()}
    package = {
        "package": out.name, "desk_number": desk, "owner": "ClusterMLIP",
        "generator": "experiments/gaussian_vasp_calibration/make_calibration_package.py",
        "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "plan": "docs/gaussian-vasp-calibration.md", "cluster": "LONI QB4",
        "gaussian": {"module": profile["module"], "command": profile["command"], "partition": "single", "cpus_per_task": 12,
                     "memory_per_task": "24GB", "time_hours": 12, "tasks": len(frames)},
        "vasp": {"module": "vasp6/6.6.1-cpu", "partition": "workq", "cores": 64, "time_hours": 12,
                 "tasks": len(jobs), "variants": sorted({j["variant"] for j in jobs})},
        "n_geometries": len(picks), "n_states": len(states), "files_sha256": files, "submitted": False,
    }
    write_lf(out / "package.json", json.dumps(package, indent=1) + "\n")
    print(json.dumps({k: package[k] for k in ("package", "n_geometries", "n_states", "gaussian", "vasp")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
