"""Write the dispatch package for the full (Fe free) UBPW91 optimization of the lowest minimum.

    python make_full_opt_package.py FROZEN_FE_FINAL.extxyz TEMPLATE_PACKAGE OUT_PARENT --date YYYY-MM-DD [--number N]

Starts from the stationary points of the frozen-Fe optimizations (desk package 118, frames
0004 and 0005: minimum A at M = 49 and at M = 47). Each task rebuilds its electronic state
with the same flip-down ladder used throughout (Stable=Opt then Force per M, at the new
starting geometry) and then runs one unconstrained ``Opt`` link at the target M, as package
99 did for bare Fe16.

Why the ladder is rerun instead of reading package 118's checkpoints:
* those checkpoints carry the frozen-atom definitions of the ModRedundant optimization;
  reading the geometry section back could carry the constraints into this run;
* a package should not depend on files in another package's work directory;
* the Stable=Opt links test the wavefunction at the relaxed geometry, which has not been done.
The cost is about 2 h (M = 49) or 4 h (M = 47) per task.

Rules kept: IOP(5/13=1,5/36=1,8/11=1) in every link; one job type per link; G09 D.01.
Package 99's Slurm script and watchdog, with a longer wall time and link limit: its
bare-Fe16 optimizations needed 14 to 25 steps and hit the 12 h link limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_dft_inputs import gaussian_input  # noqa: E402

REPO = HERE.parents[1]
NAME = "clustermlip_fe16_coh_full_opt"
JOB = "fe16-coh-fullopt"
SOURCES = [("frame_0004", 49), ("frame_0005", 47)]  # package 118 frames: minimum A, stationary, Fe frozen
LADDER = [49, 47]
CORES, MEM = 12, "24GB"
HOURS, LINK_LIMIT = 72, 64 * 3600
TEMPLATE_HOURS, TEMPLATE_LINK_LIMIT = 48, 12 * 3600
JOB_TYPES = ("SP", "Stable=Opt", "Force", "Opt", "Opt=ModRedundant", "Freq")

README = """# {number}_{name}

Full UBPW91 optimization (all atoms free) of the lowest Fe16 + CO + 2H minimum at M = 49 and M = 47
(ClusterMLIP, {date}). **2 G09 D.01 tasks.** Owner: ClusterMLIP. Generator:
`experiments/acat_fe16_co_h/make_full_opt_package.py` at commit `{commit}` (branch `{branch}`). Regenerate
there, not here. `package.json` has every file's SHA-256.

## Why

Desk package 118 relaxed the adsorbates with the Fe atoms frozen. At its stationary points the frozen Fe
atoms still carry UBPW91 forces of up to 1.1 eV/Å at M = 49 and 0.44 eV/Å at M = 47, so those are not minima
of the full system, and the 49 → 47 gap (0.150 eV there) may change either way once Fe relaxes. This package
lets everything move.

## Frames

| frame | start geometry | target M | links |
|---|---|---|---|
{rows}

- Each input starts from the stationary point package 118 found at the same M (geometry written explicitly
  in link 1; no checkpoint from 118 is read).
- The electronic state is rebuilt with the usual ladder at that geometry: `Stable=Opt` then `Force` per M,
  one checkpoint, `Guess=Read` after the first link. Then one `Opt ... Geom=Checkpoint Guess=Read` link at
  the target M, with no constraint, as in package 99.
- `IOP(5/13=1,5/36=1,8/11=1)` is in every link; one job type per link (checked by the generator).

## Run

`sbatch run_gaussian.slurm` from this directory. QB4 `single`, {cores} cores, `%mem={mem}`.

**Two settings differ from packages 99 / 112 / 116 / 118, on purpose:** wall time **{hours} h** (was 48 h)
and per-link limit **{link_h} h** (was 12 h). Three of 118's frozen-Fe optimizations hit the 12 h limit after
13 to 23 steps, and a full relaxation has five times the degrees of freedom. If QB4 `single` does not allow
{hours} h, the desk sets the largest allowed wall time and a link limit 8 h below it, and says so.
`run_gaussian.slurm` is otherwise package 99's; `gaussian_watchdog.sh` is byte-identical to 99's. At most {su} SU.

Expected: ladder about 2 h (frame 0000) and 4 h (frame 0001); then about 30 minutes per optimization step.
The number of steps is not known; 40 to 100 is plausible.

## Done

`outputs/frame_*.log`, `.seconds`, `.watchdog`, plus `logs/`. Report per frame: Normal terminations
(3 and 5 for a full run), "Convergence criterion not met" per link, whether the optimization says
"Stationary point found", the number of optimization steps, and the watchdog reason if any.

An unconverged SCF, an unfinished optimization or a watchdog stop is a result, not a package failure.
Never resubmit or restart without asking. **Keep `work/frame_*`:** an unfinished optimization would be
continued from its checkpoint.
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def job_type(route: str) -> str:
    kinds = [k for k in JOB_TYPES if f" {k} " in f" {route} "]
    if len(kinds) != 1:
        raise SystemExit(f"a link must have exactly one job type, got {kinds}: {route}")
    if "IOP(5/13=1" not in route or "UBPW91/6-311++G*" not in route:
        raise SystemExit(f"route lost IOP(5/13=1) or the level: {route}")
    return kinds[0]


def build(name: str, atoms, target: int) -> tuple[str, list[str]]:
    mults = LADDER[: LADDER.index(target) + 1]
    links = gaussian_input(name, atoms, 0, mults, MEM, CORES).split("--Link1--\n")
    force = links[-1]
    route = next(line for line in force.splitlines() if line.startswith("#"))
    if job_type(route) != "Force" or "Geom=Checkpoint Guess=Read" not in route:
        raise SystemExit("the optimization link is derived from a Force link with Geom=Checkpoint Guess=Read")
    new_route = route.replace(" Force ", " Opt ")
    head = [line for line in force.splitlines() if line.startswith("%")]
    links.append("\n".join(head + [new_route, "", f"{name} M={target} Opt, all atoms free", "", f"0 {target}"]) + "\n\n")
    kinds = [job_type(next(line for line in link.splitlines() if line.startswith("#"))) for link in links]
    return "--Link1--\n".join(links), kinds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("final", type=Path, help="final geometries of the frozen-Fe package (review_opt_package.py --xyz)")
    ap.add_argument("template", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--number", type=int, default=125, help="proposed package number (the desk decides)")
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    pkg = args.out / f"{args.number}_{NAME}"
    if pkg.exists():
        raise SystemExit(f"{pkg} exists; a staged package is never overwritten (send the desk a revision note)")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    finals = {a.info["frame"]: a for a in read(args.final, index=":")}

    files: dict[str, bytes] = {}
    table, rows = [], []
    for i, (src, mult) in enumerate(SOURCES):
        atoms = finals[src]
        if int(atoms.info["multiplicity"]) != mult or str(atoms.info["stationary_point"]) != "True":
            raise SystemExit(f"{src}: not a stationary point at M = {mult}")
        symbols = atoms.get_chemical_symbols()
        if symbols[:16] != ["Fe"] * 16 or sorted(symbols[16:]) != ["C", "H", "H", "O"]:
            raise SystemExit(f"{src}: unexpected atom order {symbols}")
        name = f"fe16coh2_minA_m{mult}"
        text, kinds = build(name, atoms, mult)
        frame = f"frame_{i:04d}"
        files[f"inputs/{frame}.gjf"] = text.encode("ascii")
        table.append({"frame": frame, "name": name, "start": f"package 118 {src} ({atoms.info['draft_job']})",
                      "start_energy_ha": atoms.info["energy_ha"], "target_multiplicity": mult, "links": len(kinds),
                      "job_types": ";".join(kinds), "frozen_atoms": ""})
        rows.append(f"| {frame} | 118 {src}, stationary at M = {mult}, {atoms.info['energy_ha']} Ha | {mult} | "
                    f"{' + '.join(kinds)} |")
    lines = [",".join(table[0])] + [",".join(str(r[k]) for k in table[0]) for r in table]
    files["inputs/frames.csv"] = ("\n".join(lines) + "\n").encode("utf-8")

    watchdog = (args.template / "gaussian_watchdog.sh").read_bytes()
    files["gaussian_watchdog.sh"] = watchdog
    slurm = (args.template / "run_gaussian.slurm").read_text(encoding="utf-8")
    for pattern, new in ((r"#SBATCH --job-name=\S+", f"#SBATCH --job-name={JOB}"),
                         (r"#SBATCH --array=\S+", f"#SBATCH --array=0-{len(SOURCES) - 1}"),
                         (rf"#SBATCH --time={TEMPLATE_HOURS}:00:00", f"#SBATCH --time={HOURS}:00:00"),
                         (rf"LINK_LIMIT={TEMPLATE_LINK_LIMIT}\b", f"LINK_LIMIT={LINK_LIMIT}")):
        slurm, n = re.subn(pattern, new, slurm)
        if n != 1:
            raise SystemExit(f"template Slurm script: {pattern} matched {n} times")
    slurm, n = re.subn(r"# Written by ClusterMLIP .*?\n(?=# Memory comes)",
                       "# Written by ClusterMLIP experiments/acat_fe16_co_h/make_full_opt_package.py from package 99's script.\n"
                       "# One array task per frame: a G09 flip-down ladder to the target M at the start geometry, then an\n"
                       "# unconstrained optimization, in work/frame_NNNN, where its checkpoint stays; the log goes to\n"
                       f"# outputs/frame_NNNN.log. Wall time {HOURS} h and link limit {LINK_LIMIT // 3600} h (99 had 48 h and 12 h).\n",
                       slurm, flags=re.S)
    if n != 1 or f"--cpus-per-task={CORES}" not in slurm or "module load gaussian/g09-d01" not in slurm:
        raise SystemExit("template Slurm script is not the expected package-99 script")
    files["run_gaussian.slurm"] = slurm.encode("utf-8")
    for small in (".gitignore", "logs/README.txt"):
        files[small] = (args.template / small).read_bytes()
    files["export.list"] = b"outputs/*\nlogs/*\n"
    su = len(SOURCES) * CORES * HOURS
    files["README.md"] = README.format(number=args.number, name=NAME, date=args.date, commit=commit, branch=branch,
                                       rows="\n".join(rows), cores=CORES, mem=MEM, hours=HOURS,
                                       link_h=LINK_LIMIT // 3600, su=su).encode("utf-8")
    meta = {"package": pkg.name, "owner": "ClusterMLIP", "generator": "experiments/acat_fe16_co_h/make_full_opt_package.py",
            "commit": commit, "branch": branch, "template_package": args.template.name, "watchdog_sha256": sha(watchdog),
            "start_geometries_sha256": sha(args.final.read_bytes()),
            "cluster": "LONI QB4", "partition": "single", "gaussian": "gaussian/g09-d01", "level": "UBPW91/6-311++G*",
            "cpus_per_task": CORES, "memory_per_task": MEM, "time_hours": HOURS, "link_limit_s": LINK_LIMIT,
            "differs_from_package_99": {"time_hours": [TEMPLATE_HOURS, HOURS], "link_limit_s": [TEMPLATE_LINK_LIMIT, LINK_LIMIT]},
            "max_su": su, "charge": 0, "frozen_atoms": None, "array": [r["frame"] for r in table], "frames": table,
            "outputs": ["outputs/frame_NNNN.log", "outputs/frame_NNNN.seconds", "outputs/frame_NNNN.watchdog"],
            "files_sha256": {k: sha(v) for k, v in sorted(files.items())}, "submitted": False}
    for rel, data in files.items():
        path = pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.replace(b"\r\n", b"\n"))
    (pkg / "package.json").write_bytes((json.dumps(meta, indent=1) + "\n").encode("utf-8"))
    print(pkg, len(files) + 1, "files;", su, "SU max; commit", commit)


if __name__ == "__main__":
    main()
