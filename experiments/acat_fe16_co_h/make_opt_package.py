"""Write the dispatch package that optimizes the lowest panel structures with UBPW91, Fe frozen.

    python make_opt_package.py DRAFT_DIR TEMPLATE_PACKAGE OUT_PARENT --date YYYY-MM-DD [--number N]

One task per (structure, target multiplicity). Each input is the draft's own flip-down ladder,
copied verbatim down to the target M (Stable=Opt then Force per M, as run in desk packages 112
and 116), followed by one optimization link at that M:

    Opt=ModRedundant ... Geom=Checkpoint Guess=Read      with  "X n F"  for the 16 Fe atoms

so only the four adsorbate atoms move (Stage A of the search, now at the reference level). The
constraint form is the one already run on this cluster under G09 (package 73: Opt=ModRedundant
with Geom=Check and the constraint lines after the charge/multiplicity line).

A last task retries the one panel state whose SCF did not converge (fe16coh2_001, M = 49):
Stable=Opt, Force, and a second Force link that continues the SCF from the first one's density.

Rules kept: IOP(5/13=1,5/36=1,8/11=1) in every link; one job type per link; G09 D.01; the Slurm
script and watchdog are package 99's.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NAME = "clustermlip_fe16_coh_opt_lowest"
JOB = "fe16-coh-opt"
STRUCTURES = [("fe16coh2_000", "lowest structure of the panel"),
              ("fe16coh2_007", "+0.02 eV at M = 49"),
              ("fe16coh2_008", "+0.07 eV at M = 49")]
TARGETS = [49, 47]
RETRY = ("fe16coh2_001", 49)
LADDER = [49, 47, 45]
N_FE = 16
CORES, MEM, HOURS, LINK_LIMIT = 12, "24GB", 48, 12 * 3600
JOB_TYPES = ("SP", "Stable=Opt", "Force", "Opt", "Opt=ModRedundant", "Freq")

README = """# {number}_{name}

UBPW91 optimizations of the three lowest Fe16 + CO + 2H structures, Fe frozen (ClusterMLIP, {date}).
**7 G09 D.01 tasks.** Owner: ClusterMLIP. Generator: `experiments/acat_fe16_co_h/make_opt_package.py`
at commit `{commit}` (branch `{branch}`). Regenerate there, not here. `package.json` has every file's SHA-256.

## Why

Desk packages 112 and 116 gave UBPW91 energies and forces for 20 structures at geometries taken from the
search model. Those are not UBPW91 minima: forces reach 1.6 eV/Å on adsorbate atoms. Two questions follow:

- where the adsorbates actually sit at the reference level, and how the three lowest structures order
  once relaxed;
- whether M = 47 becomes competitive with M = 49 after relaxation. Binding CO narrows the vertical
  49 → 47 gap from 0.52 eV (bare Fe16) to 0.13 to 0.24 eV for these three.

## Frames

| frame | draft id | target M | links | content |
|---|---|---|---|---|
{rows}

- Frames 0000 to 0005: the ladder links are the ones that ran in 112 / 116 for the same structure, copied
  verbatim down to the target M. Then one link `Opt=ModRedundant ... Geom=Checkpoint Guess=Read` with
  `X n F` for atoms 1 to 16, so the Fe16 `fe16-200-49-gf` M = 49 optimum stays fixed and only C, O, H, H move.
- Frame 0006 is a retry, not an optimization. In 116 this state's `Stable=Opt` and `Force` links both
  stopped at 129 SCF cycles unconverged. The second `Force` link continues from the first one's density.
- `IOP(5/13=1,5/36=1,8/11=1)` is in every link; one job type per link (checked by the generator).

## Run

`sbatch run_gaussian.slurm` from this directory. QB4 `single`, {cores} cores, `%mem={mem}`, {hours} h cap per
task, per-link cap {link_h} h, at most {su} SU. `run_gaussian.slurm` is package 99's with the job name and
array range changed; `gaussian_watchdog.sh` is byte-identical to 99's.

Expected: ladder links as in 112 (about 1.7 h per `Stable=Opt`, 10 minutes per `Force`). An optimization
step is one SCF from the previous density plus a gradient, about 10 to 20 minutes, so the {link_h} h link
cap allows roughly 40 steps for 12 degrees of freedom. An optimization cut by the cap is still useful.

## Done

`outputs/frame_*.log`, `.seconds` and `.watchdog`, plus `logs/`. Report per frame: the number of Normal
terminations (frames 0000/0002/0004: 3; frames 0001/0003/0005: 5; frame 0006: 3), the watchdog reason if
any, and whether the optimization reports "Stationary point found".

An unconverged SCF, an optimization that does not finish, or a watchdog stop is a result, not a package
failure. Never resubmit with a longer limit without asking. Please keep the `work/frame_*` checkpoints:
an unfinished optimization can be restarted from them.

ClusterMLIP checks afterwards that no Fe atom moved, screens every optimization step for unconverged SCFs,
and compares the relaxed energies at M = 49 and 47.
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def route_of(link: str) -> str:
    return next(line for line in link.splitlines() if line.startswith("#"))


def job_type(route: str) -> str:
    kinds = [k for k in JOB_TYPES if f" {k} " in f" {route} "]
    if len(kinds) != 1:
        raise SystemExit(f"a link must have exactly one job type, got {kinds}: {route}")
    if "IOP(5/13=1" not in route or "UBPW91/6-311++G*" not in route:
        raise SystemExit(f"route lost IOP(5/13=1) or the level: {route}")
    return kinds[0]


def header(link: str) -> list[str]:
    """The Link 0 lines (%chk, %mem, %nprocshared) of a draft link."""
    return [line for line in link.splitlines() if line.startswith("%")]


def opt_link(force_link: str, name: str, mult: int) -> str:
    route = route_of(force_link)
    if job_type(route) != "Force" or "Geom=Checkpoint Guess=Read" not in route:
        raise SystemExit("the optimization link is derived from a Force link with Geom=Checkpoint Guess=Read")
    new_route = route.replace(" Force ", " Opt=ModRedundant ")
    job_type(new_route)
    lines = header(force_link) + [new_route, "", f"{name} M={mult} Opt, Fe frozen", "", f"0 {mult}", ""]
    lines += [f"X {i} F" for i in range(1, N_FE + 1)]
    return "\n".join(lines) + "\n\n"


def build(draft_text: str, name: str, target: int, retry: bool) -> tuple[str, list[str]]:
    links = draft_text.split("--Link1--\n")
    if len(links) != 2 * len(LADDER):
        raise SystemExit(f"{name}: expected {2 * len(LADDER)} draft links")
    for i, link in enumerate(links):
        want = "Stable=Opt" if i % 2 == 0 else "Force"
        if job_type(route_of(link)) != want or f"\n0 {LADDER[i // 2]}\n" not in link:
            raise SystemExit(f"{name}: draft link {i + 1} is not {want} at M = {LADDER[i // 2]}")
    keep = links[: 2 * (LADDER.index(target) + 1)]
    if retry:
        out = keep + [keep[-1]]
    else:
        out = keep + [opt_link(keep[-1], name, target)]
    kinds = [job_type(route_of(link)) for link in out]
    return "--Link1--\n".join(out), kinds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("draft", type=Path)
    ap.add_argument("template", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--number", type=int, default=120, help="proposed package number (the desk decides)")
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    pkg = args.out / f"{args.number}_{NAME}"
    if pkg.exists():
        raise SystemExit(f"{pkg} exists; a staged package is never overwritten (send the desk a revision note)")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    with open(args.draft / "frames.csv", newline="", encoding="utf-8") as fh:
        drafts = {row["job"]: row for row in csv.DictReader(fh)}

    plan = [(job, why, m, False) for job, why in STRUCTURES for m in TARGETS]
    plan.append((RETRY[0], "retry: SCF did not converge in package 116", RETRY[1], True))
    files: dict[str, bytes] = {}
    table, rows = [], []
    for i, (job, why, mult, retry) in enumerate(plan):
        data = (args.draft / "inputs" / f"{job}.gjf").read_bytes()
        if b"\r" in data or sha(data) != drafts[job]["input_sha256"]:
            raise SystemExit(f"{job}: draft input changed or has CRLF line endings")
        text, kinds = build(data.decode("utf-8"), job, mult, retry)
        frame = f"frame_{i:04d}"
        files[f"inputs/{frame}.gjf"] = text.encode("utf-8")
        d = drafts[job]
        content = " + ".join(kinds)
        table.append({"frame": frame, "draft_job": job, "shortlist_id": d["shortlist_id"], "why": why,
                      "target_multiplicity": mult, "links": len(kinds), "job_types": content.replace(" + ", ";"),
                      "frozen_atoms": "" if retry else f"1-{N_FE}", "motif": d["motif"],
                      "adsorbate_atoms": d["adsorbate_atoms"], "draft_input_sha256": d["input_sha256"]})
        rows.append(f"| {frame} | {job} ({why}) | {mult} | {len(kinds)} | {content} |")
    lines = [",".join(table[0])] + [",".join(str(r[k]) for k in table[0]) for r in table]
    files["inputs/frames.csv"] = ("\n".join(lines) + "\n").encode("utf-8")

    watchdog = (args.template / "gaussian_watchdog.sh").read_bytes()
    files["gaussian_watchdog.sh"] = watchdog
    slurm = (args.template / "run_gaussian.slurm").read_text(encoding="utf-8")
    for pattern, new in ((r"#SBATCH --job-name=\S+", f"#SBATCH --job-name={JOB}"),
                         (r"#SBATCH --array=\S+", f"#SBATCH --array=0-{len(plan) - 1}")):
        slurm, n = re.subn(pattern, new, slurm)
        if n != 1:
            raise SystemExit(f"template Slurm script: {pattern} matched {n} times")
    slurm, n = re.subn(r"# Written by ClusterMLIP .*?\n(?=# Memory comes)",
                       "# Written by ClusterMLIP experiments/acat_fe16_co_h/make_opt_package.py from package 99's script.\n"
                       "# One array task per frame: a G09 flip-down ladder to the target M at fixed geometry, then an\n"
                       "# optimization with the Fe atoms frozen, in work/frame_NNNN, where its checkpoint stays; the log\n"
                       "# goes to outputs/frame_NNNN.log. inputs/frames.csv names each frame.\n", slurm, flags=re.S)
    if n != 1 or f"LINK_LIMIT={LINK_LIMIT}" not in slurm or f"--cpus-per-task={CORES}" not in slurm \
            or f"--time={HOURS}:00:00" not in slurm or "module load gaussian/g09-d01" not in slurm:
        raise SystemExit("template Slurm script is not the expected package-99 script")
    files["run_gaussian.slurm"] = slurm.encode("utf-8")
    for small in (".gitignore", "logs/README.txt"):
        files[small] = (args.template / small).read_bytes()
    files["export.list"] = b"outputs/*\nlogs/*\n"
    su = len(plan) * CORES * HOURS
    files["README.md"] = README.format(number=args.number, name=NAME, date=args.date, commit=commit, branch=branch,
                                       rows="\n".join(rows), cores=CORES, mem=MEM, hours=HOURS,
                                       link_h=LINK_LIMIT // 3600, su=su).encode("utf-8")
    meta = {"package": pkg.name, "owner": "ClusterMLIP", "generator": "experiments/acat_fe16_co_h/make_opt_package.py",
            "commit": commit, "branch": branch, "template_package": args.template.name, "watchdog_sha256": sha(watchdog),
            "cluster": "LONI QB4", "partition": "single", "gaussian": "gaussian/g09-d01", "level": "UBPW91/6-311++G*",
            "cpus_per_task": CORES, "memory_per_task": MEM, "time_hours": HOURS, "link_limit_s": LINK_LIMIT, "max_su": su,
            "charge": 0, "frozen_atoms": f"1-{N_FE} (Fe), by ModRedundant 'X n F'", "array": [r["frame"] for r in table],
            "frames": table, "outputs": ["outputs/frame_NNNN.log", "outputs/frame_NNNN.seconds", "outputs/frame_NNNN.watchdog"],
            "files_sha256": {k: sha(v) for k, v in sorted(files.items())}, "submitted": False}
    for rel, data in files.items():
        path = pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.replace(b"\r\n", b"\n"))
    (pkg / "package.json").write_bytes((json.dumps(meta, indent=1) + "\n").encode("utf-8"))
    print(pkg, len(files) + 1, "files;", su, "SU max; commit", commit)


if __name__ == "__main__":
    main()
