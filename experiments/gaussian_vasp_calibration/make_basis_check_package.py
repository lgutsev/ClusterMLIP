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

Nothing is submitted. Stdlib only.

    python experiments/gaussian_vasp_calibration/make_basis_check_package.py \
        --calibration D:/MLIP_Work_Folder/hpc_smoke_tests/62_clustermlip_gaussian_vasp_calibration \
        -o D:/MLIP_Work_Folder/Cluster_MLIP/loni_package/65_clustermlip_gaussian_basis_check
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

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


def bases_for(n_fe: int, qz_max_fe: int) -> list[tuple[str, str]]:
    return [LARGER[0]] + ([LARGER[1]] if n_fe <= qz_max_fe else [])


def gjf(task: str, title: str, M: int, symbols: list[str], positions: list[tuple], larger: list[tuple[str, str]]) -> str:
    coords = "".join(f"{el:<2} {x:16.8f} {y:16.8f} {z:16.8f}\n" for el, (x, y, z) in zip(symbols, positions))
    head = "%nprocshared=12\n%mem=24GB\n"
    blocks = [
        f"%chk={task}.chk\n{head}#p UBPW91/{B0[1]} {TAIL}\n\n{title} BPW91 {B0[0]}\n\n0 {M}\n{coords}\n",
        f"%chk={task}.chk\n{head}#p UPW91PW91/{B0[1]} {TAIL} Geom=Checkpoint Guess=Read\n\n{title} PW91 {B0[0]}\n\n0 {M}\n\n",
    ]
    for label, key in larger:
        tag = label.split("-")[1].lower()
        blocks.append(f"%oldchk={task}.chk\n%chk={task}_{tag}.chk\n{head}#p UPW91PW91/{key} {TAIL} Guess=Read\n\n"
                      f"{title} PW91 {label}\n\n0 {M}\n{coords}\n")
    return "--Link1--\n".join(blocks)


RUN_GAUSSIAN = """#!/bin/bash
#SBATCH --job-name=g09-basis-check
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Written by ClusterMLIP experiments/gaussian_vasp_calibration/make_basis_check_package.py. One array task
# per line of tasks.csv: UBPW91/6-311++G* > UPW91PW91/6-311++G* > UPW91PW91/def2-TZVPP [> def2-QZVPP], Force,
# in gaussian/work/<task>; log to gaussian/outputs/. Memory comes from each input's %mem (24GB).
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


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibration", required=True, help="package-62 results directory (states.csv, gaussian/inputs)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--qz-max-fe", type=int, default=10, help="largest Fe count that also gets def2-QZVPP")
    ap.add_argument("--desk-number", type=int, help="the number the dispatch desk assigned; unset = a proposal")
    args = ap.parse_args(argv)

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    pkg62 = read_pkg62(Path(args.calibration))
    rows = []

    def add(task, state_id, s, delta, positions, why):
        larger = bases_for(s["n_fe"], args.qz_max_fe)
        write_lf(out / "gaussian" / "inputs" / f"{task}.gjf", gjf(task, task, s["M"], s["symbols"], positions, larger))
        rows.append({"task": task, "state_id": state_id, "pkg62_frame": s["frame"] if delta == 0 else "",
                     "formula": s["formula"], "n_fe": s["n_fe"], "multiplicity": s["M"],
                     "fe2_stretch_a": f"{delta:+.3f}" if s["formula"] == "Fe2" else "",
                     "links": ";".join(["BPW91/" + B0[0], "PW91/" + B0[0]] + ["PW91/" + b for b, _ in larger]),
                     "n_links": 2 + len(larger), "why": why})

    for k, (sid, why) in enumerate(STATES):
        s = pkg62[sid]
        add(f"bc{k:02d}_{sid}", sid, s, 0.0, s["positions"], why)
    fe2 = pkg62[FE2_STATE]
    for j, d in enumerate(FE2_SCAN):
        add(f"bc{len(STATES) + j:02d}_Fe2_m7_scan{d:+.2f}".replace("+", "p").replace("-", "m"), FE2_STATE, fe2, d,
            stretched(fe2, d), "Fe2 bond scan: re per basis vs experiment")

    with (out / "tasks.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    write_lf(out / "run_gaussian.slurm", RUN_GAUSSIAN.replace("@MODULE@", MODULE).replace("@COMMAND@", COMMAND))
    write_lf(out / "submit.sh", SUBMIT.format(last=len(rows) - 1, n=len(rows)))
    write_lf(out / "export.list", "gaussian/outputs/*\n")
    write_lf(out / "logs" / "README.txt", "Slurm logs land here.\n")
    desk = args.desk_number
    write_lf(out / "README.md", README.format(
        title=f"Package {desk}" if desk is not None else f"Package {out.name.split('_')[0]} (proposed)",
        n_states=len(STATES), n_scan=len(FE2_SCAN), n_tasks=len(rows), module=MODULE, command=COMMAND,
        qz_max_fe=args.qz_max_fe))
    generator = Path(__file__).resolve()
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob("*")) if p.is_file()}
    package = {
        "package": out.name, "desk_number": desk, "owner": "ClusterMLIP",
        "generator": "experiments/gaussian_vasp_calibration/make_basis_check_package.py",
        "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "plan": "docs/gaussian-basis-check.md", "source_package": "62_clustermlip_gaussian_vasp_calibration",
        "cluster": "LONI QB4",
        "gaussian": {"module": MODULE, "command": COMMAND, "partition": "single", "cpus_per_task": 12,
                     "memory_per_task": "24GB", "time_hours": 12, "tasks": len(rows)},
        "bases": [B0[0]] + [b for b, _ in LARGER], "qz_max_fe": args.qz_max_fe,
        "files_sha256": files, "submitted": False,
    }
    write_lf(out / "package.json", json.dumps(package, indent=1) + "\n")
    print(json.dumps({k: package[k] for k in ("package", "gaussian", "bases")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
