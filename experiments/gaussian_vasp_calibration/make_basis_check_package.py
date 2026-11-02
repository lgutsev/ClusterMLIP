"""Stage the G09 basis-set check on the package-62 force gap (see docs/gaussian-basis-check.md).

Package 62 found that G09 UPW91PW91/6-311++G* and VASP PW91 (PAW Fe_pv, basis-converged) forces on the
same Fe_n(N/O) geometries point the same way but differ 3-4x: the two codes put the minima at different
bond lengths. This package repeats the G09 PW91 single points on a few package-62 states with larger
Gaussian basis sets, to tell which side of that gap is off.

One G09 D.01 task per state (or Fe2 scan point); its links, all Force, NoSymm, Int=UltraFine, Pop=Hirshfeld,
SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) and IOP(5/13=1,5/36=1,8/11=1) in EVERY link:

  1  UBPW91/6-311++G*            default guess    replays package 62's link 1 (its guess source)
  2  UPW91PW91/6-311++G*         Guess=Read       replays package 62's G_PW91 link (the control, B0)
  3  UPW91PW91/Def2TZVPP         Guess=Read from link 2's checkpoint (%OldChk), own %Chk
  4  UPW91PW91/Def2QZVPP         same, only when n_Fe <= --qz-max-fe

Links 3 and 4 read the 6-311++G* PW91 orbitals (projected) so they stay in the same state. Geometries are
the package-62 Gaussian input coordinates, unchanged; the Fe2 scan points stretch that bond.

--ladder beyond-qz stages the follow-up (package 75 returned ladder_not_converged): the same links 1-2, then
on Fe2 M7 (s000 and its 6 scan points) and Fe8 M25 (s024) the built-in def2-QZVPP and Fe-only Gen sets from the
Basis Set Exchange (basis/*.gbs, verbatim, SHA-256 pinned): def2-QZVPP through Gen (transcription control),
decontracted def2-QZVPP, pcseg-3, pcseg-4 without its i shell, pcseg-4, and the decontracted def2-QZVPP with
Int=(UltraFine,DKH2) (scalar relativity, which VASP's PAW has and G09's default Hamiltonian lacks). Every Gen
link reads link 2's orbitals through %OldChk. Fe8 is split into one task per large basis (12 h cap); on Fe2 the
links that might fail (DKH2, then i functions) come last. It also stages a VASP PW91 Fe2 M7 bond scan
(NUPDOWN=6) on the same 7 bond lengths at Fe_pv and Fe_sv (and Fe_sv at ENCUT 650), plus Fe8 s024 at Fe_sv,
from package 62's prod_pw91 inputs with only POSCAR / POTCAR.spec / ENCUT changed.

Nothing is submitted. Stdlib only.

    python experiments/gaussian_vasp_calibration/make_basis_check_package.py \
        --calibration D:/MLIP_Work_Folder/hpc_smoke_tests/62_clustermlip_gaussian_vasp_calibration \
        -o D:/MLIP_Work_Folder/Cluster_MLIP/loni_package/65_clustermlip_gaussian_basis_check

    python experiments/gaussian_vasp_calibration/make_basis_check_package.py --ladder beyond-qz \
        --calibration D:/MLIP_Work_Folder/hpc_smoke_tests/62_clustermlip_gaussian_vasp_calibration \
        -o D:/MLIP_Work_Folder/Cluster_MLIP/loni_package/80_clustermlip_gaussian_basis_attribution
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_calibration_package as mcp  # noqa: E402  (sibling: package 62's VASP runner)

SCF = "SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)"
IOP = "IOP(5/13=1,5/36=1,8/11=1)"  # every link; never drop it
TAIL = f"{SCF} NoSymm Force {IOP} Int=UltraFine Pop=Hirshfeld"
MODULE, COMMAND = "gaussian/g09-d01", "g09"  # the user keeps these systems on G09 D.01
# (basis label, G09 keyword); B0 is the training basis. def2 sets are all-electron for Fe and built into G09.
B0 = ("6-311++G*", "6-311++G*")
LARGER = (("def2-TZVPP", "Def2TZVPP"), ("def2-QZVPP", "Def2QZVPP"))
# package-62 states: converged G_PW91 and V_PW91 SCF, matching Fe spin pattern (gv_calibration_62/states.csv)
STATES = [
    ("s000_Fe2_m7_minimum", "force gap; Fe2 anchor"),
    ("s005_Fe2N2O2_m9_minimum", "force gap with N/O"),
    ("s007_Fe2N2O2_m9_offeq", "deformation energy vs s005; large forces"),
    ("s023_Fe8_m23_minimum", "spin gap vs s024 (vertical)"),
    ("s024_Fe8_m25_minimum", "force gap"),
    ("s025_Fe8_m27_minimum", "spin gap vs s024 (vertical)"),
    ("s028_Fe10_m29_minimum", "force gap"),
    ("s032_Fe16_m51_minimum", "force gap, Fe16 (pkg-62 link 1 unconverged; link 2 converged)"),
    ("s036_Fe16N2_m51_minimum", "force gap, Fe16N2"),
]
FE2_STATE = "s000_Fe2_m7_minimum"
FE2_SCAN = (-0.08, -0.06, -0.04, -0.02, 0.02, 0.04)  # Angstrom added to the package-62 Fe2 bond

# --- beyond def2-QZVPP (--ladder beyond-qz) ------------------------------------------------------
BUILTIN = dict(LARGER)
BASIS_DIR = Path(__file__).resolve().parent / "basis"
# Fe-only Gen sets, verbatim from the Basis Set Exchange v0.12 API (gaussian94 format, elements=26), 2026-11-01.
# label -> (file, transform). Every system here is pure Fe, so Gen needs no other element.
GEN_SETS = {
    "def2-QZVPP(gen)": ("Fe_def2-QZVPP.gbs", None),  # control: must reproduce the built-in Def2QZVPP link
    "unc-def2-QZVPP": ("Fe_def2-QZVPP.gbs", "uncontract"),
    "pcseg-3": ("Fe_pcseg-3.gbs", None),
    "pcseg-4(-i)": ("Fe_pcseg-4.gbs", "drop_i"),
    "pcseg-4": ("Fe_pcseg-4.gbs", None),
    # the r2 revision's relativity pair: same 8s7p5d3f2g1h shells, one contracted for the non-relativistic
    # Hamiltonian and one recontracted for Douglas-Kroll; no lone tight primitive (package 83's quadrature failure)
    "cc-pVQZ": ("Fe_cc-pVQZ.gbs", None),
    "cc-pVQZ-DK": ("Fe_cc-pVQZ-DK.gbs", None),
}
BASIS_SHA256 = {
    "Fe_def2-QZVPP.gbs": "17c879366d6ea69fc07989ee82c23c525442decd0fdb3064264236a465855823",
    "Fe_pcseg-3.gbs": "169ffae03de8253f9d3fa6dd4a73fdb704324c4b51cb94e8c8e9d5a08b047a11",
    "Fe_pcseg-4.gbs": "29b72ad21fd65c13323b413afe97b6b735bcbbe03b40c88f341ffb719ec43e0e",
    "Fe_cc-pVQZ.gbs": "f1a595e6f9ce9f11e30d83a91c790c2aec4a9074373d3c4759a05813eef3cfce",
    "Fe_cc-pVQZ-DK.gbs": "22f075b89ec0459fc616e4e25637a6cbe67a29ab24aa5ce14db093c54a1ebc74",
}
DKH = "+DKH2"  # label suffix: that Gen set with Int=(UltraFine,DKH2), as a single point (G09 D.01 has no
# analytic DKH gradients: package 83 stopped in l1 with "No analytic first derivatives for this method")
UNC_MERGE = 1.05  # decontracting: drop a primitive within this exponent ratio of one already kept (s 72.8/72.2, p 4.39/4.54)
L_OF = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6}
# Fe2: one task per geometry, every rung; links that may fail on G09 D.01 (DKH2 gradients, i functions) go last,
# because a failed link ends the job
FE2_BEYOND = ("def2-QZVPP", "def2-QZVPP(gen)", "unc-def2-QZVPP", "pcseg-3", "pcseg-4(-i)", "unc-def2-QZVPP" + DKH, "pcseg-4")
FE8_STATE = "s024_Fe8_m25_minimum"
FE8_BEYOND = (("def2-QZVPP", "pcseg-3"), ("unc-def2-QZVPP",), ("pcseg-4(-i)",), ("unc-def2-QZVPP" + DKH,))
# --ladder beyond-qz-r2, the revision after package 83 (docs/gaussian-basis-check.md, "Package 83 results"):
# ONE large-basis link per task, so no failure blocks another basis. unc-def2-QZVPP is dropped (its lone
# 4.3e6 s primitive fails G09's per-function quadrature check under Int=UltraFine). Relativity comes from
# DKH2 single-point energies: the Fe2 scan (re) and an Fe8 breathing stretch about the centroid.
FE2_R2 = ("pcseg-3", "pcseg-4(-i)", "pcseg-4", "cc-pVQZ", "cc-pVQZ-DK" + DKH)
FE8_R2 = ("cc-pVQZ", "cc-pVQZ-DK" + DKH)
BREATHING = (0.99, 1.00, 1.01)  # Fe8 s024 positions scaled about their centroid
# VASP Fe2 M7 scan: (variant, Fe POTCAR, ENCUT or None = package 62's 520). Fe_sv variants also run Fe8 s024.
VASP_SCAN = (("pw91_pv", "Fe_pv", None), ("pw91_sv", "Fe_sv", None), ("pw91_sv_encut650", "Fe_sv", 650))
# The desk's fix (package 83): up to 8 array tasks loading the VASP container module at once can corrupt its
# wrapper (package 77 died with "Exec format error"), so each task waits (index % 8) x 20 s first. Same bytes as
# the desk's run_vasp.slurm.
VASP_STAGGER_AT = "module purge\nmodule load vasp6/6.6.1-cpu\n"
VASP_STAGGER = ("# desk fix: stagger the container module loads (8 tasks at once; 77 died with Exec format error that way)\n"
                "sleep $(( ${SLURM_ARRAY_TASK_ID:-0} % 8 * 20 ))\n")


def parse_gbs(path: Path) -> list[tuple[str, list[tuple[str, str]]]]:
    """One element's gaussian94 block -> [(shell letter, [(exponent token, coefficient token)])]."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASIS_SHA256[path.name]:
        raise SystemExit(f"{path}: SHA-256 does not match the Basis Set Exchange copy")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("!")]
    if lines[0].split() != ["Fe", "0"] or lines[-1] != "****":
        raise SystemExit(f"{path}: expected one Fe block")
    shells, i = [], 1
    while i < len(lines) - 1:
        letter, n, _ = lines[i].split()
        shells.append((letter, [tuple(ln.split()[:2]) for ln in lines[i + 1:i + 1 + int(n)]]))
        i += 1 + int(n)
    return shells


def _num(token: str) -> float:
    return float(token.replace("D", "E"))


def gen_shells(label: str) -> list[tuple[str, list[tuple[str, str]]]]:
    file, how = GEN_SETS[label.removesuffix(DKH)]
    shells = parse_gbs(BASIS_DIR / file)
    if how == "drop_i":
        shells = [s for s in shells if s[0] != "I"]
    elif how == "uncontract":
        kept: dict[str, list[str]] = {}
        for letter, prims in shells:
            for e, _ in prims:
                have = kept.setdefault(letter, [])
                if all(abs(math.log(_num(e) / _num(k))) > math.log(UNC_MERGE) for k in have):
                    have.append(e)
        shells = [(letter, [(e, "1.0000000")]) for letter in sorted(kept, key=L_OF.get)
                  for e in sorted(kept[letter], key=_num, reverse=True)]
    return shells


def gen_block(label: str) -> str:
    out = ["Fe     0"]
    for letter, prims in gen_shells(label):
        out.append(f"{letter}    {len(prims)}   1.00")
        out += [f"  {e:>20}  {c:>20}" for e, c in prims]
    return "\n".join(out + ["****"]) + "\n"


def functions_per_fe(label: str) -> int:
    if label in BUILTIN:
        return {"def2-TZVPP": 64, "def2-QZVPP": 100}[label]  # G09 NBasis / n_Fe in package 75
    return sum(2 * L_OF[letter] + 1 for letter, _ in gen_shells(label))


def read_pkg62(results: Path) -> dict[str, dict]:
    """state_id -> {frame, formula, n_fe, M, symbols, positions} from package 62's states.csv and link-1 inputs."""
    out = {}
    for row in csv.DictReader((results / "states.csv").open(encoding="utf-8")):
        text = (results / "gaussian" / "inputs" / f"{row['frame']}.gjf").read_text(encoding="utf-8")
        link1 = text.split("--Link1--")[0]
        m = re.search(r"\n0 (\d+)\n(.*?)\n\n", link1, re.S)
        if not m or int(m.group(1)) != int(row["multiplicity"]):
            raise SystemExit(f"{row['frame']}: cannot read charge/multiplicity and coordinates")
        symbols, positions = [], []
        for line in m.group(2).splitlines():
            el, *xyz = line.split()
            symbols.append(el)
            positions.append(tuple(float(v) for v in xyz))
        out[row["state_id"]] = {"frame": row["frame"], "formula": row["formula"], "n_fe": int(row["n_fe"]),
                                "M": int(row["multiplicity"]), "symbols": symbols, "positions": positions}
    return out


def stretched(state: dict, delta: float) -> list[tuple]:
    """The Fe2 geometry with its bond lengthened by delta, about the same midpoint and axis."""
    (a, b) = state["positions"]
    mid = [(x + y) / 2 for x, y in zip(a, b)]
    r = math.dist(a, b)
    u = [(x - y) / r for x, y in zip(a, b)]
    h = (r + delta) / 2
    return [tuple(c + h * k for c, k in zip(mid, u)), tuple(c - h * k for c, k in zip(mid, u))]


def bases_for(n_fe: int, qz_max_fe: int) -> list[str]:
    return [LARGER[0][0]] + ([LARGER[1][0]] if n_fe <= qz_max_fe else [])


def gjf(task: str, title: str, M: int, symbols: list[str], positions: list[tuple], larger: list[str]) -> str:
    coords = "".join(f"{el:<2} {x:16.8f} {y:16.8f} {z:16.8f}\n" for el, (x, y, z) in zip(symbols, positions))
    head = "%nprocshared=12\n%mem=24GB\n"
    blocks = [
        f"%chk={task}.chk\n{head}#p UBPW91/{B0[1]} {TAIL}\n\n{title} BPW91 {B0[0]}\n\n0 {M}\n{coords}\n",
        f"%chk={task}.chk\n{head}#p UPW91PW91/{B0[1]} {TAIL} Geom=Checkpoint Guess=Read\n\n{title} PW91 {B0[0]}\n\n0 {M}\n\n",
    ]
    for label in larger:
        if label in BUILTIN:
            tag, route, basis = label.split("-")[1].lower(), f"UPW91PW91/{BUILTIN[label]} {TAIL}", ""
        else:  # Fe-only Gen set after the geometry; 5D 7F stated (G09's def2 built-ins are pure too)
            tag, route, basis = re.sub(r"[^a-z0-9]+", "", label.lower()), f"UPW91PW91/Gen 5D 7F {TAIL}", gen_block(label) + "\n"
            if label.endswith(DKH):
                route = route.replace("Int=UltraFine", "Int=(UltraFine,DKH2)").replace(" Force", "")
        blocks.append(f"%oldchk={task}.chk\n%chk={task}_{tag}.chk\n{head}#p {route} Guess=Read\n\n"
                      f"{title} PW91 {label}\n\n0 {M}\n{coords}\n{basis}")
    return "--Link1--\n".join(blocks)


RUN_GAUSSIAN = """#!/bin/bash
#SBATCH --job-name=@JOB@
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Written by ClusterMLIP experiments/gaussian_vasp_calibration/make_basis_check_package.py. One array task
# per line of tasks.csv: UBPW91/6-311++G* > UPW91PW91/6-311++G* > the larger-basis UPW91PW91 links in its
# `links` column, all Force, in gaussian/work/<task>; log to gaussian/outputs/. Memory comes from each
# input's %mem (24GB).
# The module function reads unset variables: -u after it.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the package directory}"
PKG=$PWD
task=$(awk -F, -v n="$((SLURM_ARRAY_TASK_ID + 2))" 'NR == n {print $1}' tasks.csv)
[ -n "$task" ] || { echo "no task for array index $SLURM_ARRAY_TASK_ID" >&2; exit 2; }
mkdir -p gaussian/outputs "gaussian/work/$task"
module load @MODULE@
set -u
scratch="${TMPDIR:-/tmp}/g09_${SLURM_JOB_ID}_${task}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
cd "gaussian/work/$task"
start=$(date +%s)
status=0
GAUSS_SCRDIR="$scratch" @COMMAND@ < "$PKG/gaussian/inputs/$task.gjf" > "$PKG/gaussian/outputs/$task.log" || status=$?
printf '%s\\n' "$(( $(date +%s) - start ))" > "$PKG/gaussian/outputs/$task.seconds"
exit "$status"
"""

SUBMIT = """#!/bin/bash
# Submit the one Gaussian array (nothing else).
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
mkdir -p logs
sbatch --array=0-{last}%{n} run_gaussian.slurm
"""

README = """# {title}: G09 basis-set check on the package-62 force gap (ClusterMLIP)

Package 62 found that G09 UPW91PW91/6-311++G* and VASP PW91 forces on the same Fe_n(N/O) geometries point
the same way but differ 3-4x (Fe force RMSE ~160 meV/A; VASP is basis-converged). This package reruns the G09
PW91 single points on {n_states} package-62 states plus a {n_scan}-point Fe2 bond scan with larger Gaussian
basis sets, to tell whether the 6-311++G* side is the incomplete one. Plan and decision rule: ClusterMLIP
`docs/gaussian-basis-check.md`.

| part | tasks | where | per task |
|---|---|---|---|
| Gaussian | {n_tasks} (`tasks.csv`, `gaussian/inputs/<task>.gjf`) | QB4 `single`, `{module}` (`{command}`) | 12 cores, 24 GB, 12 h |

Every task runs, on one fixed geometry, with Force and IOP(5/13=1,5/36=1,8/11=1) in every link:

1. UBPW91/6-311++G* (default guess; replays package 62's first link);
2. UPW91PW91/6-311++G*, Guess=Read (replays package 62's PW91 link: the control);
3. UPW91PW91/def2-TZVPP, Guess=Read from link 2 (`%OldChk`);
4. UPW91PW91/def2-QZVPP, the same way, only for clusters with at most {qz_max_fe} Fe.

`tasks.csv` lists each task's state, links and why it is in the set.

## Run

From the package directory: `bash submit.sh`.

## Done

- every `gaussian/outputs/<task>.log` has as many `Normal termination` lines as its `n_links` in
  `tasks.csv`, and a `.seconds` file;
- `export.list` is `gaussian/outputs/*`.

An unconverged SCF is a result, not a package failure (IOP(5/13=1) lets it run on to Normal termination; the
analysis screens every link). If a task hits 12 h, report its time and the last link and SCF cycle reached;
do not resubmit with a longer limit without asking.
"""


SUBMIT_BEYOND = """#!/bin/bash
# Submit both arrays (nothing else). Re-running is safe for VASP: finished jobs (DONE) exit at once.
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
mkdir -p logs
sbatch --array=0-{last_g}%{n_g} run_gaussian.slurm
sbatch --array=0-{last_v}%8 run_vasp.slurm
"""

README_BEYOND = """# {title}: finish attributing the package-62 G09-vs-VASP PW91 force gap (ClusterMLIP)

Package 62 found G09 UPW91PW91/6-311++G* and VASP PW91 (PAW Fe_pv) forces on the same Fe clusters differ by
~160 meV/A for Fe, VASP wanting shorter bonds. Package 75 showed def2-QZVPP closes 69% of the Fe2 gap and
16-23% for Fe8/Fe10, but the TZVPP->QZVPP step was still 0.3-0.5 of the gap. This package (a) pushes the G09
ladder past def2-QZVPP on Fe2 M7 and Fe8 M25 and adds scalar relativity (DKH2), and (b) runs a real VASP Fe2
bond scan at Fe_pv and Fe_sv. Plan and decision rule: ClusterMLIP `docs/gaussian-basis-check.md`
("Beyond def2-QZVPP").

| part | tasks | where | per task |
|---|---|---|---|
| Gaussian | {n_tasks} (`tasks.csv`, `gaussian/inputs/<task>.gjf`) | QB4 `single`, `{module}` (`{command}`) | 12 cores, 24 GB, 12 h |
| VASP | {n_v} (`vasp/jobs.txt`) | QB4 `workq`, `vasp6/6.6.1-cpu`, `vasp_gam` | 64 cores, 1 h, at most 8 at once |

**Gaussian.** Every task runs, on one fixed geometry, Force with
`SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Int=UltraFine` and IOP(5/13=1,5/36=1,8/11=1) in
every link:

1. UBPW91/6-311++G* (replays package 62's first link);
2. UPW91PW91/6-311++G*, Guess=Read (replays package 62's PW91 link: the control);
3. onward: UPW91PW91 with the larger bases in the task's `links` column, each reading link 2's orbitals
   (`%OldChk`). Gen sets are Fe-only, from the Basis Set Exchange, written after the geometry.

| basis | functions per Fe | what it tests |
|---|---:|---|
{basis_rows}

The `+DKH2` link uses `Int=(UltraFine,DKH2)`. On the Fe2 tasks it and the `pcseg-4` link (i functions) are
last, because a failed link ends the job; if either ends in Error termination, that is a result.

**VASP.** Single points (IBRION=-1) from package 62's `prod_pw91` inputs, byte-identical INCAR/KPOINTS (same
box, NUPDOWN=6, MAGMOM 2*3, ENCUT 520) except: POSCAR (the Fe2 bond set to the same 7 lengths as the Gaussian
tasks), POTCAR.spec (`Fe_pv` or `Fe_sv`), and ENCUT 650 in `pw91_sv_encut650`. The Fe_sv variants also run Fe8
s024 (package 62 already has it at Fe_pv). `vasp/jobs.csv` lists every job.

## Run

From the package directory: `bash submit.sh`. POTCARs come from `$POTCARS` (default
`/home/lgutsev/pot/potpaw_PBE`) and need **`Fe_pv` and `Fe_sv`**; check `ls $POTCARS/Fe_sv/POTCAR` first.

## Done

- every `gaussian/outputs/<task>.log` has as many `Normal termination` lines as its `n_links` in `tasks.csv`
  (or fewer only because a `+DKH2` or `pcseg-4` link ended in Error termination: report which), and a
  `.seconds` file;
- every `vasp/outputs/<variant>/<point>/` has `vasprun.xml` with `</modeling>`, OUTCAR and OSZICAR;
- `export.list` is `gaussian/outputs/*` and `vasp/outputs/*`.

An unconverged SCF is a result, not a package failure. If a task hits its time limit, report its time and the
last link and SCF cycle reached; do not resubmit with a longer limit without asking.
"""


README_R2 = """# {title}: revision of package 83, relativity and the Fe2 basis limit (ClusterMLIP)

Package 83 (G09 basis ladder past def2-QZVPP + VASP Fe2 scan) came back partial: its decontracted
def2-QZVPP links failed G09's quadrature check, which blocked every later link on Fe2, and G09 D.01 has no
DKH2 gradients. Its finished links already show the Gaussian basis is converged for Fe8 (def2-QZVPP, pcseg-3
and pcseg-4(-i) agree) and explains only ~20% of the Fe8 force gap, and that VASP Fe_sv moves further away
from G09, not closer. This revision measures the remaining candidate, scalar relativity, and finishes the
Fe2 basis ladder. Plan: ClusterMLIP `docs/gaussian-basis-check.md`, "Package 83 results and the revision".

| part | tasks | where | per task |
|---|---|---|---|
| Gaussian | {n_tasks} (`tasks.csv`, `gaussian/inputs/<task>.gjf`) | QB4 `single`, `{module}` (`{command}`) | 12 cores, 24 GB, 12 h |

Each task has ONE large-basis link, so a failure cannot block another basis. Every link carries
`SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Int=UltraFine` and IOP(5/13=1,5/36=1,8/11=1):

1. UBPW91/6-311++G* (replays package 62's first link);
2. UPW91PW91/6-311++G*, Guess=Read (replays package 62's PW91 link);
3. UPW91PW91 with the task's basis (Fe-only Gen from the Basis Set Exchange), reading link 2's orbitals
   (`%OldChk`).

| basis | functions per Fe | role |
|---|---:|---|
{basis_rows}

- {n_fe8} Fe8 M25 tasks (first in the array; about 1 h each): cc-pVQZ and cc-pVQZ-DK+DKH2 at the package-62
  geometry scaled by 0.99, 1.00 and 1.01 about its centroid.
- {n_fe2} Fe2 M7 tasks (minutes each): every basis above at the 7 bond lengths of package 83.

The DKH2 links use `Int=(UltraFine,DKH2)` and are single points (no Force): G09 D.01 has no analytic DKH2
gradients.

## Run

From the package directory: `bash submit.sh` (at most 20 tasks at once).

## Done

- every `gaussian/outputs/<task>.log` has 3 `Normal termination` lines and a `.seconds` file;
- `export.list` is `gaussian/outputs/*`.

Results, not failures: an unconverged SCF, and an Error termination of a `pcseg-4` link (i functions).
Report which. If a task hits 12 h, report its time and the last link and SCF cycle reached; do not resubmit
with a longer limit without asking.
"""


def read_poscar(path: Path) -> tuple[list[str], int, list[tuple]]:
    """(lines, index of the first coordinate line, Cartesian positions) of a package-62 POSCAR."""
    lines = path.read_text(encoding="utf-8").splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.strip() == "Cartesian") + 1
    n = sum(int(v) for v in lines[k - 2].split())
    return lines, k, [tuple(float(v) for v in ln.split()[:3]) for ln in lines[k:k + n]]


def stage_vasp_scan(cal: Path, out: Path, points: list[tuple[str, float]]) -> list[dict]:
    """Package 62's prod_pw91 Fe2 (and Fe8) inputs with only the bond length, Fe POTCAR and ENCUT changed."""
    src = cal / "vasp" / "inputs" / "prod_pw91"
    jobs = []
    for variant, potcar, encut in VASP_SCAN:
        targets = [(name, FE2_STATE, d) for name, d in points] + ([(FE8_STATE, FE8_STATE, None)] if potcar != "Fe_pv" else [])
        for name, sid, delta in targets:
            dst = out / "vasp" / "inputs" / variant / name
            dst.mkdir(parents=True)
            for f in ("KPOINTS", "job.json"):
                shutil.copyfile(src / sid / f, dst / f)
            incar = (src / sid / "INCAR").read_text(encoding="utf-8")
            if encut is not None:
                incar, n = re.subn(r"^ENCUT = .*$", f"ENCUT = {encut}", incar, flags=re.M)
                assert n == 1, "package-62 INCAR has no ENCUT line"
            write_lf(dst / "INCAR", incar)
            spec = (src / sid / "POTCAR.spec").read_text(encoding="utf-8")
            assert spec.split() == ["Fe_pv"], f"{sid}: expected a Fe_pv-only POTCAR.spec"
            write_lf(dst / "POTCAR.spec", potcar + "\n")
            lines, k, pos = read_poscar(src / sid / "POSCAR")
            r = math.dist(*pos[:2]) if len(pos) == 2 else None
            if delta:
                new = stretched({"positions": pos}, delta)
                lines[k:k + 2] = [f"  {x:18.10f}  {y:18.10f}  {z:18.10f}" for x, y, z in new]
                r = math.dist(*new)
            write_lf(dst / "POSCAR", "\n".join(lines) + "\n")
            jobs.append({"job": f"{variant}/{name}", "variant": variant, "point": name, "state_id": sid,
                         "potcar_fe": potcar, "encut": encut or 520, "fe2_bond_a": f"{r:.5f}" if r else "",
                         "binary": "vasp_gam"})
    return jobs


def beyond_qz_tasks(pkg62: dict) -> list[tuple]:
    """(task, state_id, state, stretch, positions, larger labels, why) for --ladder beyond-qz."""
    fe2, fe8 = pkg62[FE2_STATE], pkg62[FE8_STATE]
    out = [("bq00_" + FE2_STATE, FE2_STATE, fe2, 0.0, fe2["positions"], FE2_BEYOND, "Fe2 anchor (scan point 0)")]
    for d in FE2_SCAN:
        name = f"Fe2_m7_scan{d:+.2f}".replace("+", "p").replace("-", "m")
        out.append((f"bq{len(out):02d}_{name}", FE2_STATE, fe2, d, stretched(fe2, d), FE2_BEYOND, "Fe2 bond scan: re per basis"))
    for larger in FE8_BEYOND:
        tag = "_".join(re.sub(r"[^a-z0-9]+", "", b.lower()) for b in larger)
        out.append((f"bq{len(out):02d}_{FE8_STATE}_{tag}", FE8_STATE, fe8, 0.0, fe8["positions"], larger,
                    "Fe8 force gap; one task per large basis (12 h cap)"))
    return out


def breathed(state: dict, scale: float) -> list[tuple]:
    c = [sum(p[k] for p in state["positions"]) / len(state["positions"]) for k in range(3)]
    return [tuple(c[k] + scale * (p[k] - c[k]) for k in range(3)) for p in state["positions"]]


def r2_tasks(pkg62: dict) -> list[tuple]:
    """(task, state_id, state, stretch, positions, larger labels, why, breathing scale) for --ladder beyond-qz-r2.
    Fe8 first: those are the long tasks."""
    fe2, fe8 = pkg62[FE2_STATE], pkg62[FE8_STATE]
    out = []
    for sc in BREATHING:
        for label in FE8_R2:
            tag = re.sub(r"[^a-z0-9]+", "", label.lower())
            out.append((f"br{len(out):02d}_{FE8_STATE}_b{sc:.2f}_{tag}".replace(".", "p"), FE8_STATE, fe8, 0.0,
                        breathed(fe8, sc), (label,), "Fe8 breathing: relativity (DKH2 - NR) on the radial force", sc))
    geoms = [(FE2_STATE, 0.0, fe2["positions"])] + [
        (f"Fe2_m7_scan{d:+.2f}".replace("+", "p").replace("-", "m"), d, stretched(fe2, d)) for d in FE2_SCAN]
    for label in FE2_R2:
        tag = re.sub(r"[^a-z0-9]+", "", label.lower())
        for name, d, pos in geoms:
            out.append((f"br{len(out):02d}_{name}_{tag}", FE2_STATE, fe2, d, pos, (label,), "Fe2 re per basis", None))
    return out


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibration", required=True, help="package-62 results directory (states.csv, gaussian/inputs)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--ladder", choices=("qz", "beyond-qz", "beyond-qz-r2"), default="qz",
                    help="qz: package 75 (def2-TZVPP/QZVPP on 9 states); beyond-qz: Gen sets past def2-QZVPP + VASP Fe2 "
                         "scan (package 83); beyond-qz-r2: the package-83 revision (one basis per task, DKH2 single points)")
    ap.add_argument("--qz-max-fe", type=int, default=10, help="largest Fe count that also gets def2-QZVPP")
    ap.add_argument("--desk-number", type=int, help="the number the dispatch desk assigned; unset = a proposal")
    args = ap.parse_args(argv)

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    cal = Path(args.calibration)
    pkg62 = read_pkg62(cal)
    rows = []

    def add(task, state_id, s, delta, positions, larger, why, breathing=None):
        write_lf(out / "gaussian" / "inputs" / f"{task}.gjf", gjf(task, task, s["M"], s["symbols"], positions, larger))
        on_pkg62 = delta == 0 and breathing in (None, 1.0)  # link 2 then replays package 62 exactly (the control)
        rows.append({"task": task, "state_id": state_id, "pkg62_frame": s["frame"] if on_pkg62 else "",
                     "formula": s["formula"], "n_fe": s["n_fe"], "multiplicity": s["M"],
                     "fe2_stretch_a": f"{delta:+.3f}" if s["formula"] == "Fe2" else "",
                     **({"breathing_scale": f"{breathing:.2f}" if breathing else ""} if r2 else {}),
                     "links": ";".join(["BPW91/" + B0[0], "PW91/" + B0[0]] + ["PW91/" + b for b in larger]),
                     "n_links": 2 + len(larger), "why": why})

    beyond, r2 = args.ladder == "beyond-qz", args.ladder == "beyond-qz-r2"
    if r2:
        for task, sid, s, d, positions, larger, why, sc in r2_tasks(pkg62):
            add(task, sid, s, d, positions, list(larger), why, sc)
    elif beyond:
        for task, sid, s, d, positions, larger, why in beyond_qz_tasks(pkg62):
            add(task, sid, s, d, positions, list(larger), why)
    else:
        for k, (sid, why) in enumerate(STATES):
            s = pkg62[sid]
            add(f"bc{k:02d}_{sid}", sid, s, 0.0, s["positions"], bases_for(s["n_fe"], args.qz_max_fe), why)
        fe2 = pkg62[FE2_STATE]
        for j, d in enumerate(FE2_SCAN):
            add(f"bc{len(STATES) + j:02d}_Fe2_m7_scan{d:+.2f}".replace("+", "p").replace("-", "m"), FE2_STATE, fe2, d,
                stretched(fe2, d), bases_for(fe2["n_fe"], args.qz_max_fe), "Fe2 bond scan: re per basis vs experiment")

    with (out / "tasks.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    write_lf(out / "run_gaussian.slurm", RUN_GAUSSIAN.replace("@MODULE@", MODULE).replace("@COMMAND@", COMMAND)
             .replace("@JOB@", "g09-basis-attrib" if beyond or r2 else "g09-basis-check"))
    write_lf(out / "logs" / "README.txt", "Slurm logs land here.\n")
    desk = args.desk_number
    title = f"Package {desk}" if desk is not None else f"Package {out.name.split('_')[0]} (proposed)"
    jobs = []
    if beyond:
        points = [(FE2_STATE, 0.0)] + [(r["task"].split("_", 1)[1], float(r["fe2_stretch_a"])) for r in rows
                                       if r["formula"] == "Fe2" and r["fe2_stretch_a"] != "+0.000"]
        jobs = stage_vasp_scan(cal, out, points)
        write_lf(out / "vasp" / "jobs.txt", "".join(f"{j['job']} {j['binary']}\n" for j in jobs))
        with (out / "vasp" / "jobs.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(jobs[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(jobs)
        assert VASP_STAGGER_AT in mcp.RUN_VASP, "package 62's runner changed; re-place the module-load stagger"
        write_lf(out / "run_vasp.slurm", mcp.RUN_VASP.replace("gv-calib-vasp", "basis-attrib-vasp")
                 .replace("#SBATCH --time=12:00:00", "#SBATCH --time=01:00:00")
                 .replace("make_calibration_package.py", "make_basis_check_package.py (package 62's runner)")
                 .replace(VASP_STAGGER_AT, VASP_STAGGER + VASP_STAGGER_AT))
        write_lf(out / "submit.sh", SUBMIT_BEYOND.format(last_g=len(rows) - 1, n_g=len(rows), last_v=len(jobs) - 1))
        write_lf(out / "export.list", "gaussian/outputs/*\nvasp/outputs/*\n")
        what = {"def2-QZVPP": "built-in; reproduces package 75 (anchor)",
                "def2-QZVPP(gen)": "the BSE def2-QZVPP through Gen: must equal the built-in link (transcription control)",
                "unc-def2-QZVPP": f"fully decontracted def2-QZVPP (primitives within {UNC_MERGE}x merged): contraction error",
                "pcseg-3": "Jensen's DFT-optimised QZ family, with an h shell",
                "pcseg-4(-i)": "pcseg-4 without its one i shell (Fe8; i-function gradients untested on G09 D.01)",
                "pcseg-4": "Jensen's 5Z, near the DFT basis limit (Fe2 only, last link)",
                "unc-def2-QZVPP" + DKH: "scalar relativity (DKH2) at a fixed decontracted basis; VASP PAW is scalar-relativistic"}
        labels = list(FE2_BEYOND)
        write_lf(out / "README.md", README_BEYOND.format(
            title=title, n_tasks=len(rows), module=MODULE, command=COMMAND, n_v=len(jobs),
            basis_rows="\n".join(f"| {b} | {functions_per_fe(b.removesuffix(DKH))} | {what[b]} |" for b in labels)))
    elif r2:
        write_lf(out / "submit.sh", SUBMIT.format(last=len(rows) - 1, n=20))
        write_lf(out / "export.list", "gaussian/outputs/*\n")
        what = {"pcseg-3": "Jensen's DFT-optimised QZ, with an h shell",
                "pcseg-4(-i)": "pcseg-4 without its i shell (ran on Fe8 in package 83)",
                "pcseg-4": "Jensen's 5Z, with the i shell (G09 i-function gradients untested; its own task)",
                "cc-pVQZ": "non-relativistic partner of the DKH2 link; Force",
                "cc-pVQZ-DK" + DKH: "Douglas-Kroll recontracted cc-pVQZ with Int=(UltraFine,DKH2); single point"}
        write_lf(out / "README.md", README_R2.format(
            title=title, n_tasks=len(rows), module=MODULE, command=COMMAND,
            n_fe8=len(BREATHING) * len(FE8_R2), n_fe2=len(rows) - len(BREATHING) * len(FE8_R2),
            basis_rows="\n".join(f"| {b} | {functions_per_fe(b.removesuffix(DKH))} | {what[b]} |" for b in FE2_R2)))
    else:
        write_lf(out / "submit.sh", SUBMIT.format(last=len(rows) - 1, n=len(rows)))
        write_lf(out / "export.list", "gaussian/outputs/*\n")
        write_lf(out / "README.md", README.format(
            title=title, n_states=len(STATES), n_scan=len(FE2_SCAN), n_tasks=len(rows), module=MODULE,
            command=COMMAND, qz_max_fe=args.qz_max_fe))
    generator = Path(__file__).resolve()
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob("*")) if p.is_file()}
    package = {
        "package": out.name, "desk_number": desk, "owner": "ClusterMLIP",
        "generator": "experiments/gaussian_vasp_calibration/make_basis_check_package.py",
        "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(), "ladder": args.ladder,
        "plan": "docs/gaussian-basis-check.md", "source_package": "62_clustermlip_gaussian_vasp_calibration",
        "cluster": "LONI QB4",
        "gaussian": {"module": MODULE, "command": COMMAND, "partition": "single", "cpus_per_task": 12,
                     "memory_per_task": "24GB", "time_hours": 12, "tasks": len(rows)},
        "files_sha256": files, "submitted": False,
    }
    if beyond:
        package |= {"bases": [B0[0]] + list(FE2_BEYOND),
                    "functions_per_fe": {b: functions_per_fe(b.removesuffix(DKH)) for b in FE2_BEYOND},
                    "gen_sources": {f: {"sha256": h, "from": "https://www.basissetexchange.org/api/basis/<name>/format/gaussian94/?elements=26",
                                        "bse_version": "0.12"} for f, h in BASIS_SHA256.items()},
                    "vasp": {"module": "vasp6/6.6.1-cpu", "partition": "workq", "cores": 64, "time_hours": 1,
                             "tasks": len(jobs), "variants": [v for v, _, _ in VASP_SCAN],
                             "source": "62_clustermlip_gaussian_vasp_calibration vasp/inputs/prod_pw91"}}
    elif r2:
        package |= {"bases": [B0[0]] + list(FE2_R2), "breathing_scales": list(BREATHING),
                    "functions_per_fe": {b: functions_per_fe(b.removesuffix(DKH)) for b in FE2_R2},
                    "gen_sources": {f: {"sha256": BASIS_SHA256[f], "bse_version": "0.12",
                                        "from": "https://www.basissetexchange.org/api/basis/<name>/format/gaussian94/?elements=26"}
                                    for f in sorted({GEN_SETS[b.removesuffix(DKH)][0] for b in FE2_R2})},
                    "revises": "83_clustermlip_gaussian_basis_attribution"}
    else:
        package |= {"bases": [B0[0]] + [b for b, _ in LARGER], "qz_max_fe": args.qz_max_fe}
    write_lf(out / "package.json", json.dumps(package, indent=1) + "\n")
    print(json.dumps({k: package.get(k) for k in ("package", "gaussian", "bases", "functions_per_fe", "vasp")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
