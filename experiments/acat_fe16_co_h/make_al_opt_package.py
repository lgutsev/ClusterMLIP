"""Write the dispatch package for the full UBPW91 optimization of one active-learning structure at several M.

    python make_al_opt_package.py AL_PACKAGE FRAME TEMPLATE_PACKAGE OUT_PARENT --date YYYY-MM-DD [--number N]

AL_PACKAGE is the returned fixed-geometry package (inputs/FRAME.gjf holds the start geometry). One task per
target M in 49, 47, 45: the flip-down ladder at the start geometry down to the target M (Stable=Opt then
Force per M), then one unconstrained ``Opt`` link at the target M, as desk package 124 did for minimum A.
The M = 45 task therefore also gives energies and forces at M = 49, 47 and 45 on one geometry.

Rules kept: IOP(5/13=1,5/36=1,8/11=1) in every link; one job type per link; G09 D.01; package 99's Slurm
script and watchdog with 124's wall time and link limit.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_full_opt_package as base  # noqa: E402
from review_reference_batch import read_input  # noqa: E402

NAME = "clustermlip_fe16_coh_al1_opt"
JOB = "fe16-coh-al1opt"
TARGETS = [49, 47, 45]
base.LADDER = TARGETS

README = """# {number}_{name}

Full UBPW91 optimization (all atoms free) of the lowest structure of the first active-learning round,
at M = 49, 47 and 45 (ClusterMLIP, {date}). **3 G09 D.01 tasks.** Owner: ClusterMLIP. Generator:
`experiments/acat_fe16_co_h/make_al_opt_package.py` at commit `{commit}` (branch `{branch}`). Regenerate
there, not here. `package.json` has every file's SHA-256.

## Why

Desk package 142 labelled 8 model-proposed structures at fixed geometry. {src} ({motif}) is the lowest:
0.18 eV below panel structure 000 before any UBPW91 relaxation, with forces of up to 0.8 eV/Å left. The
relaxed minimum A (package 124) is 0.38 eV below 000, so this structure may relax below it. The three
multiplicities give the adiabatic spin gaps, and the ladder gives the three states on one geometry.

## Frames

| frame | start geometry | target M | links |
|---|---|---|---|
{rows}

- The start geometry is written explicitly in link 1 and equals package 142's input for {src}. No
  checkpoint from 142 is read.
- Ladder at the start geometry: `Stable=Opt` then `Force` per M, from 49 down to the target, one
  checkpoint, `Guess=Read` after the first link. Then one `Opt ... Geom=Checkpoint Guess=Read` link at the
  target M, with no constraint, as in packages 99 and 124.
- `IOP(5/13=1,5/36=1,8/11=1)` is in every link; one job type per link (checked by the generator).

## Run

`sbatch run_gaussian.slurm` from this directory. QB4 `single`, {cores} cores, `%mem={mem}`. Wall time
**{hours} h** and per-link limit **{link_h} h**, as package 124 (package 99 had 48 h and 12 h).
`run_gaussian.slurm` is otherwise package 99's; `gaussian_watchdog.sh` is byte-identical to 99's.
At most {su} SU.

Expected: ladder about 2 h per multiplicity (142 measured 2.0 h for M = 49 on this geometry); then about
30 minutes per optimization step. Package 124 needed 24 and 26 steps from a pre-relaxed geometry; this
start is further from a minimum, so 30 to 60 steps is plausible. About 700 to 1,200 core-hours in total.

## Done

`outputs/frame_*.log`, `.seconds`, `.watchdog`, plus `logs/`. Report per frame: Normal terminations
(3, 5 and 7 for a full run), "Convergence criterion not met" per link, whether the optimization says
"Stationary point found", the number of optimization steps, and the watchdog reason if any.

An unconverged SCF, an unfinished optimization or a watchdog stop is a result, not a package failure.
Never resubmit or restart without asking. **Keep `work/frame_*`:** an unfinished optimization would be
continued from its checkpoint.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("al_package", type=Path)
    ap.add_argument("frame")
    ap.add_argument("template", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--number", type=int, default=145, help="proposed package number (the desk decides)")
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    pkg = args.out / f"{args.number}_{NAME}"
    if pkg.exists():
        raise SystemExit(f"{pkg} exists; a staged package is never overwritten (send the desk a revision note)")
    git = lambda *a: subprocess.run(["git", *a], cwd=base.REPO, capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    commit, branch = git("rev-parse", "--short", "HEAD"), git("branch", "--show-current")
    source = args.al_package / "inputs" / f"{args.frame}.gjf"
    atoms, _ = read_input(source)
    symbols = atoms.get_chemical_symbols()
    if symbols[:16] != ["Fe"] * 16 or sorted(symbols[16:]) != ["C", "H", "H", "O"]:
        raise SystemExit(f"unexpected atom order {symbols}")
    import csv
    with open(args.al_package / "inputs" / "frames.csv", newline="", encoding="utf-8") as fh:
        row = next(r for r in csv.DictReader(fh) if r["frame"] == args.frame)
    src = f"142 {args.frame} ({row['al_id']}, pool {row['pool_id']})"

    files: dict[str, bytes] = {}
    table, rows = [], []
    for i, mult in enumerate(TARGETS):
        name = f"fe16coh2_{row['al_id'].lower()}_m{mult}"
        text, kinds = base.build(name, atoms, mult)
        frame = f"frame_{i:04d}"
        files[f"inputs/{frame}.gjf"] = text.encode("ascii")
        table.append({"frame": frame, "name": name, "start": src.replace(",", ""), "motif": row["motif"],
                      "target_multiplicity": mult, "links": len(kinds), "job_types": ";".join(kinds), "frozen_atoms": ""})
        rows.append(f"| {frame} | {src} | {mult} | {' + '.join(kinds)} |")
    files["inputs/frames.csv"] = ("\n".join([",".join(table[0])] + [",".join(str(r[k]) for k in table[0]) for r in table])
                                  + "\n").encode("utf-8")

    watchdog = (args.template / "gaussian_watchdog.sh").read_bytes()
    files["gaussian_watchdog.sh"] = watchdog
    slurm = (args.template / "run_gaussian.slurm").read_text(encoding="utf-8")
    for pattern, new in ((r"#SBATCH --job-name=\S+", f"#SBATCH --job-name={JOB}"),
                         (r"#SBATCH --array=\S+", f"#SBATCH --array=0-{len(TARGETS) - 1}"),
                         (rf"#SBATCH --time={base.TEMPLATE_HOURS}:00:00", f"#SBATCH --time={base.HOURS}:00:00"),
                         (rf"LINK_LIMIT={base.TEMPLATE_LINK_LIMIT}\b", f"LINK_LIMIT={base.LINK_LIMIT}")):
        slurm, n = re.subn(pattern, new, slurm)
        if n != 1:
            raise SystemExit(f"template Slurm script: {pattern} matched {n} times")
    slurm, n = re.subn(r"# Written by ClusterMLIP .*?\n(?=# Memory comes)",
                       "# Written by ClusterMLIP experiments/acat_fe16_co_h/make_al_opt_package.py from package 99's script.\n"
                       "# One array task per frame: a G09 flip-down ladder to the target M at the start geometry, then an\n"
                       "# unconstrained optimization, in work/frame_NNNN, where its checkpoint stays; the log goes to\n"
                       f"# outputs/frame_NNNN.log. Wall time {base.HOURS} h and link limit {base.LINK_LIMIT // 3600} h (99 had 48 h and 12 h).\n",
                       slurm, flags=re.S)
    if n != 1 or f"--cpus-per-task={base.CORES}" not in slurm or "module load gaussian/g09-d01" not in slurm:
        raise SystemExit("template Slurm script is not the expected package-99 script")
    files["run_gaussian.slurm"] = slurm.encode("utf-8")
    for small in (".gitignore", "logs/README.txt"):
        files[small] = (args.template / small).read_bytes()
    files["export.list"] = b"outputs/*\nlogs/*\n"
    su = len(TARGETS) * base.CORES * base.HOURS
    files["README.md"] = README.format(number=args.number, name=NAME, date=args.date, commit=commit, branch=branch,
                                       src=src, motif=row["motif"], rows="\n".join(rows), cores=base.CORES, mem=base.MEM,
                                       hours=base.HOURS, link_h=base.LINK_LIMIT // 3600, su=su).encode("utf-8")
    meta = {"package": pkg.name, "owner": "ClusterMLIP", "generator": "experiments/acat_fe16_co_h/make_al_opt_package.py",
            "commit": commit, "branch": branch, "template_package": args.template.name, "watchdog_sha256": base.sha(watchdog),
            "start_geometry_input_sha256": base.sha(source.read_bytes()),
            "cluster": "LONI QB4", "partition": "single", "gaussian": "gaussian/g09-d01", "level": "UBPW91/6-311++G*",
            "cpus_per_task": base.CORES, "memory_per_task": base.MEM, "time_hours": base.HOURS, "link_limit_s": base.LINK_LIMIT,
            "differs_from_package_99": {"time_hours": [base.TEMPLATE_HOURS, base.HOURS],
                                        "link_limit_s": [base.TEMPLATE_LINK_LIMIT, base.LINK_LIMIT]},
            "max_su": su, "charge": 0, "frozen_atoms": None, "array": [r["frame"] for r in table], "frames": table,
            "outputs": ["outputs/frame_NNNN.log", "outputs/frame_NNNN.seconds", "outputs/frame_NNNN.watchdog"],
            "files_sha256": {k: base.sha(v) for k, v in sorted(files.items())}, "submitted": False}
    for rel, data in files.items():
        path = pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.replace(b"\r\n", b"\n"))
    (pkg / "package.json").write_bytes((json.dumps(meta, indent=1) + "\n").encode("utf-8"))
    print(pkg, len(files) + 1, "files;", su, "SU max; commit", commit)


if __name__ == "__main__":
    main()
