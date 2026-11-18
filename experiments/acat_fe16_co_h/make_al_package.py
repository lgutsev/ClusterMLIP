"""Write a dispatch package for one active-learning round of Fe16 + CO + 2H reference labels.

    python make_al_package.py PROPOSAL_DIR TEMPLATE_PACKAGE OUT_PARENT --date YYYY-MM-DD --number N

PROPOSAL_DIR is the output of `al_propose.py propose` (al_round1_proposal.extxyz and .csv).
TEMPLATE_PACKAGE is the staged package 99: its watchdog is copied unchanged and its Slurm script is
reused with only the job name, array range, time limit and header comment changed.

Each input is a fixed-geometry M = 49 single state: one Stable=Opt link, then one Force link from the
same checkpoint. The routes come from make_dft_inputs.py (the panel's), so the labels are comparable
with packages 112 and 116. Atoms 1 to 16 must be the unchanged Fe16 optimum of the panel inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from ase.io import read

from make_dft_inputs import gaussian_input

REPO = Path(__file__).resolve().parents[2]
NAME = "clustermlip_fe16_coh_al_round1"
MULT, CHARGE = 49, 0
CORES, MEM, HOURS, LINK_LIMIT = 12, "24GB", 24, 12 * 3600

README = """# {number}_{name}

First active-learning round for Fe16 + CO + 2H (ClusterMLIP, {date}). **{n} G09 D.01 tasks.**
Owner: ClusterMLIP. Generator: `experiments/acat_fe16_co_h/make_al_package.py` at commit `{commit}`
(branch `{branch}`). Regenerate there, not here. `package.json` has every file's SHA-256.

## Why

The 20-structure reference panel (desk packages 112 and 116) showed that no corrected model passes the
gate and that the error is still falling with more structures. These {n} structures were chosen by the
model itself (r2SCAN foundation model fine-tuned on the 99 existing M = 49 labels, plus an energy
correction with an uncertainty): five with the lowest predicted energy and three with the largest
uncertainty. The two foundation models disagree about the CO site (top/bridge against hollow); this
list contains both, so the labels settle that and extend the training set.

## Frames

| frame | proposal id | picked for | motif |
|---|---|---|---|
{rows}

Every input: fixed geometry, charge 0, **M = 49 only**; one `Stable=Opt` link and one `Force` link
(2 links), one checkpoint, `Guess=Read` in the second link. Route as in packages 112 and 116, with
`IOP(5/13=1,5/36=1,8/11=1)` in both links and one job type per link (checked by the generator). Atoms 1
to 16 are the Fe16 `fe16-200-49-gf` M = 49 optimum, unchanged (checked against package 116's input);
atoms 17 to 20 are C, O, H, H.

## Run

`sbatch run_gaussian.slurm` from this directory. QB4 `single`, {cores} cores, `%mem={mem}`, {hours} h cap per
task, per-link cap {link_h} h, at most {su} SU. `run_gaussian.slurm` is package 99's with the job name,
array range and time limit ({hours} h instead of 48 h, for two links) changed; `gaussian_watchdog.sh` is
byte-identical to 99's.

Expected from 112 and 116: about 1.7 h per task for the two M = 49 links, so about {expect} core-hours.

## Done

`outputs/frame_*.log`, `.seconds` and `.watchdog`, plus `logs/`. Report the watchdog reasons, the number
of Normal terminations per frame (2 = full run) and the count of "Convergence criterion not met" per link.

An unconverged SCF or a watchdog stop is a result, not a package failure. Never resubmit with a longer
limit without asking. ClusterMLIP takes energies and forces only from links whose own SCF converged.
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_input(text: str, name: str) -> None:
    links = text.split("--Link1--\n")
    if len(links) != 2:
        raise SystemExit(f"{name}: {len(links)} links, expected 2")
    for i, (link, want) in enumerate(zip(links, ("Stable=Opt", "Force"))):
        route = next(line for line in link.splitlines() if line.startswith("#"))
        kinds = [k for k in ("SP", "Stable=Opt", "Force", "Opt", "Freq") if f" {k} " in f" {route} "]
        if kinds != [want]:
            raise SystemExit(f"{name} link {i + 1}: job types {kinds}, expected [{want}]")
        if "IOP(5/13=1" not in route or "UBPW91/6-311++G*" not in route:
            raise SystemExit(f"{name} link {i + 1}: route lost IOP(5/13=1) or the level")
        if f"\n{CHARGE} {MULT}\n" not in link:
            raise SystemExit(f"{name} link {i + 1}: charge/multiplicity line")
        if (i > 0) != ("Geom=Checkpoint Guess=Read" in route):
            raise SystemExit(f"{name} link {i + 1}: Guess=Read placement")
    if f"%mem={MEM}" not in text or f"%nprocshared={CORES}" not in text:
        raise SystemExit(f"{name}: %mem / %nprocshared differ from the package resources")


def panel_fe(path: Path) -> np.ndarray:
    rows = [line.split() for line in path.read_text(encoding="utf-8").split("--Link1--")[0].splitlines()
            if line.startswith("Fe ")]
    return np.array([[float(x) for x in r[1:4]] for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("proposal", type=Path)
    ap.add_argument("template", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--panel-input", type=Path, required=True, help="a staged panel input, for the Fe16 check")
    ap.add_argument("--number", type=int, required=True, help="proposed package number (the desk decides)")
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    pkg = args.out / f"{args.number}_{NAME}"
    if pkg.exists():
        raise SystemExit(f"{pkg} exists; a staged package is never overwritten (send the desk a revision note)")
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    commit, branch = git("rev-parse", "--short", "HEAD"), git("branch", "--show-current")
    frames = read(args.proposal / "al_round1_proposal.extxyz", index=":")
    with open(args.proposal / "al_round1_proposal.csv", newline="", encoding="utf-8") as fh:
        meta_rows = list(csv.DictReader(fh))
    if [a.info["al_id"] for a in frames] != [r["al_id"] for r in meta_rows]:
        raise SystemExit("proposal extxyz and csv disagree")
    fe = panel_fe(args.panel_input)

    files: dict[str, bytes] = {}
    table, rows = [], []
    for i, (a, m) in enumerate(zip(frames, meta_rows)):
        if a.get_chemical_symbols() != ["Fe"] * 16 + ["C", "O", "H", "H"]:
            raise SystemExit(f"{m['al_id']}: unexpected atom order")
        if np.abs(a.positions[:16] - fe).max() > 1e-8:  # the panel inputs carry 10 decimals
            raise SystemExit(f"{m['al_id']}: Fe16 differs from the panel geometry")
        d = a.get_all_distances()
        if min(d[p, q] for p in range(16, 20) for q in range(20) if p != q) < 0.9:
            raise SystemExit(f"{m['al_id']}: atoms closer than 0.9 A")
        job = f"fe16coh2_al1_{i:02d}"
        text = gaussian_input(job, a, CHARGE, [MULT], MEM, CORES)
        check_input(text, job)
        data = text.encode("utf-8")
        frame = f"frame_{i:04d}"
        files[f"inputs/{frame}.gjf"] = data
        why = "low predicted energy" if m["picked_for"].startswith("lowest") else "large uncertainty"
        table.append({"frame": frame, "job": job, "al_id": m["al_id"], "pool_id": m["pool_id"], "why": why, "motif": m["motif"],
                      "charge": CHARGE, "multiplicities": MULT, "links": 2, "catalyst_atoms": "1-16",
                      "adsorbate_atoms": "17:C:mol0 18:O:mol0 19:H:mol1 20:H:mol2", "input_sha256": sha(data)})
        rows.append(f"| {frame} | {m['al_id']} ({m['pool_id']}) | {why} | `{m['motif']}` |")
    lines = [",".join(table[0])] + [",".join(str(r[k]) for k in table[0]) for r in table]
    files["inputs/frames.csv"] = ("\n".join(lines) + "\n").encode("utf-8")

    watchdog = (args.template / "gaussian_watchdog.sh").read_bytes()
    files["gaussian_watchdog.sh"] = watchdog
    slurm = (args.template / "run_gaussian.slurm").read_text(encoding="utf-8")
    for pattern, new in ((r"#SBATCH --job-name=\S+", "#SBATCH --job-name=fe16-coh-al1"),
                         (r"#SBATCH --array=\S+", f"#SBATCH --array=0-{len(table) - 1}"),
                         (r"#SBATCH --time=48:00:00", f"#SBATCH --time={HOURS}:00:00")):
        slurm, n = re.subn(pattern, new, slurm)
        if n != 1:
            raise SystemExit(f"template Slurm script: {pattern} matched {n} times")
    head, n = re.subn(r"# Written by ClusterMLIP .*?\n(?=# Memory comes)",
                      "# Written by ClusterMLIP experiments/acat_fe16_co_h/make_al_package.py from package 99's script.\n"
                      "# One array task per frame: G09 Stable=Opt then Force at M = 49, fixed geometry, in\n"
                      "# work/frame_NNNN, where its checkpoint stays; the log goes to outputs/frame_NNNN.log.\n"
                      "# inputs/frames.csv names each frame.\n", slurm, flags=re.S)
    if n != 1 or f"LINK_LIMIT={LINK_LIMIT}" not in head or f"--cpus-per-task={CORES}" not in head \
            or "module load gaussian/g09-d01" not in head:
        raise SystemExit("template Slurm script is not the expected package-99 script")
    files["run_gaussian.slurm"] = head.encode("utf-8")
    for small in (".gitignore", "logs/README.txt"):
        files[small] = (args.template / small).read_bytes()
    export = (args.template / "export.list").read_text(encoding="utf-8").splitlines()
    if "logs/*" not in export:
        export.append("logs/*")
    files["export.list"] = "".join(line + "\n" for line in export).encode("utf-8")
    su = len(table) * CORES * HOURS
    files["README.md"] = README.format(number=args.number, name=NAME, date=args.date, commit=commit, branch=branch,
                                       n=len(table), rows="\n".join(rows), cores=CORES, mem=MEM, hours=HOURS,
                                       link_h=LINK_LIMIT // 3600, su=su,
                                       expect=round(len(table) * CORES * 1.7)).encode("utf-8")
    meta = {"package": pkg.name, "owner": "ClusterMLIP", "generator": "experiments/acat_fe16_co_h/make_al_package.py",
            "commit": commit, "branch": branch, "proposal_dir": args.proposal.name, "template_package": args.template.name,
            "watchdog_sha256": sha(watchdog), "cluster": "LONI QB4", "partition": "single", "gaussian": "gaussian/g09-d01",
            "level": "UBPW91/6-311++G*", "cpus_per_task": CORES, "memory_per_task": MEM, "time_hours": HOURS,
            "link_limit_s": LINK_LIMIT, "max_su": su, "charge": CHARGE, "multiplicities": [MULT], "links_per_frame": 2,
            "array": [r["frame"] for r in table], "frames": table,
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
