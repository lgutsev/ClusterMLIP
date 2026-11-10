"""Write the first dispatch package of the Fe16 + CO + 2H reference panel (4 of the 20 drafts).

    python make_first_batch_package.py DRAFT_DIR TEMPLATE_PACKAGE OUT_PARENT --date YYYY-MM-DD [--number N]

DRAFT_DIR is the output of make_dft_inputs.py (inputs/*.gjf and frames.csv). The inputs are
copied byte for byte; this script does not write Gaussian routes. TEMPLATE_PACKAGE is the
staged package 99 (Fe16 ladder, then Opt): its watchdog is copied unchanged and its Slurm
script is reused with only the job name, the array range and the header comment changed, so
the run environment is the one that worked for bare Fe16.

The four inputs measure timing and SCF behaviour before the other 16 are sent:
the model minimum, one desorbed-CO structure, one near-degenerate competitor, the O-bound control.
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
NAME = "clustermlip_fe16_coh_reference_batch1"
JOB = "fe16-coh-ref1"
PICKS = [("fe16coh2_000", "model minimum (MACE-MPA-0)"),
         ("fe16coh2_002", "desorbed CO: lowest such structure on the model"),
         ("fe16coh2_006", "near-degenerate competitor of the minimum"),
         ("fe16coh2_017", "constructed control: CO bound through O")]
MULTIPLICITIES = [49, 47, 45]
CORES, MEM, HOURS, LINK_LIMIT = 12, "24GB", 48, 12 * 3600

README = """# {number}_{name}

First batch of the Fe16 + CO + 2H reference panel (ClusterMLIP, {date}). **4 G09 D.01 tasks.**
Owner: ClusterMLIP. Generator: `experiments/acat_fe16_co_h/make_first_batch_package.py` at commit
`{commit}` (branch `{branch}`). Regenerate there, not here. `package.json` has every file's SHA-256.

## The question

The adsorbate search on Fe16 + CO + 2H ran on a foundation model (MACE-MPA-0) that fails our UBPW91
reference gate, so its ranking is only a list of candidates. These four structures are the first UBPW91
labels for this composition. They answer:

- does UBPW91 bind CO at all, and by how much (the model puts a desorbed CO within 0.02 eV of the best
  bound structure; the second model puts it about 2 eV higher);
- does UBPW91 order the model minimum and its near-degenerate competitor the same way;
- how long a link takes and how the SCF behaves with adsorbates, before the other 16 inputs are sent.

## Frames

| frame | draft id | why | motif |
|---|---|---|---|
{rows}

Every input: fixed geometry, charge 0, flip-down ladder M = 49, 47, 45; per M one `Stable=Opt` link and
one `Force` link (6 links), one checkpoint, `Guess=Read` after the first link. Same route as packages
91 and 99, with `IOP(5/13=1,5/36=1,8/11=1)` in every link and one job type per link (checked by the
generator). Atoms 1 to 16 are the Fe16 `fe16-200-49-gf` M = 49 optimum, unchanged; atoms 17 to 20 are
the adsorbates (`inputs/frames.csv`).

## Run

`sbatch run_gaussian.slurm` from this directory. QB4 `single`, {cores} cores, `%mem={mem}`, {hours} h cap per
task, per-link cap {link_h} h, at most {su} SU. `run_gaussian.slurm` is package 99's with the job name and
array range changed; `gaussian_watchdog.sh` is byte-identical to 99's.

Expected: 0.6 to 1 h per link if the SCFs converge as for bare Fe16 (4 to 6 h per task). The lower
rungs may run to the link cap.

## Done

`outputs/frame_*.log`, `.seconds` and `.watchdog`. Report the watchdog reasons and the number of Normal
terminations per frame (6 = full run).

An unconverged SCF or a watchdog stop is a result, not a package failure. Never resubmit with a longer
limit without asking. ClusterMLIP screens every link for unconverged SCFs ("Convergence criterion not
met") before using an energy or a force.
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def check_input(text: str, name: str) -> None:
    links = text.split("--Link1--\n")
    if len(links) != 2 * len(MULTIPLICITIES):
        raise SystemExit(f"{name}: {len(links)} links, expected {2 * len(MULTIPLICITIES)}")
    for i, link in enumerate(links):
        route = next(line for line in link.splitlines() if line.startswith("#"))
        want = "Stable=Opt" if i % 2 == 0 else "Force"
        kinds = [k for k in ("SP", "Stable=Opt", "Force", "Opt", "Freq") if f" {k} " in f" {route} "]
        if kinds != [want]:
            raise SystemExit(f"{name} link {i + 1}: job types {kinds}, expected [{want}]")
        if "IOP(5/13=1" not in route or "UBPW91/6-311++G*" not in route:
            raise SystemExit(f"{name} link {i + 1}: route lost IOP(5/13=1) or the level")
        if f"\n0 {MULTIPLICITIES[i // 2]}\n" not in link:
            raise SystemExit(f"{name} link {i + 1}: charge/multiplicity line is not '0 {MULTIPLICITIES[i // 2]}'")
        if (i > 0) != ("Geom=Checkpoint Guess=Read" in route):
            raise SystemExit(f"{name} link {i + 1}: Guess=Read placement")
    if f"%mem={MEM}" not in text or f"%nprocshared={CORES}" not in text:
        raise SystemExit(f"{name}: %mem / %nprocshared differ from the package resources")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("draft", type=Path)
    ap.add_argument("template", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--number", type=int, default=112)
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

    files: dict[str, bytes] = {}
    table, rows = [], []
    for i, (job, why) in enumerate(PICKS):
        data = (args.draft / "inputs" / f"{job}.gjf").read_bytes()
        if b"\r" in data:
            raise SystemExit(f"{job}: CRLF line endings")
        if sha(data) != drafts[job]["input_sha256"]:
            raise SystemExit(f"{job}: input differs from the draft manifest")
        check_input(data.decode("utf-8"), job)
        frame = f"frame_{i:04d}"
        files[f"inputs/{frame}.gjf"] = data
        d = drafts[job]
        table.append({"frame": frame, "draft_job": job, "shortlist_id": d["shortlist_id"], "why": why, "motif": d["motif"],
                      "charge": d["charge"], "multiplicities": d["multiplicities"], "links": d["links"],
                      "catalyst_atoms": d["catalyst_atoms"], "adsorbate_atoms": d["adsorbate_atoms"],
                      "input_sha256": d["input_sha256"]})
        rows.append(f"| {frame} | {job} ({d['shortlist_id']}) | {why} | `{d['motif']}` |")
    lines = [",".join(table[0])] + [",".join(str(r[k]) for k in table[0]) for r in table]
    files["inputs/frames.csv"] = ("\n".join(lines) + "\n").encode("utf-8")

    watchdog = (args.template / "gaussian_watchdog.sh").read_bytes()
    files["gaussian_watchdog.sh"] = watchdog
    slurm = (args.template / "run_gaussian.slurm").read_text(encoding="utf-8")
    for pattern, new in ((r"#SBATCH --job-name=\S+", f"#SBATCH --job-name={JOB}"),
                         (r"#SBATCH --array=\S+", f"#SBATCH --array=0-{len(PICKS) - 1}")):
        slurm, n = re.subn(pattern, new, slurm)
        if n != 1:
            raise SystemExit(f"template Slurm script: {pattern} matched {n} times")
    head, n = re.subn(r"# Written by ClusterMLIP .*?\n(?=# Memory comes)",
                      "# Written by ClusterMLIP experiments/acat_fe16_co_h/make_first_batch_package.py from package 99's script.\n"
                      "# One array task per frame: a G09 flip-down ladder (Stable=Opt, Force at M = 49, 47, 45) at fixed\n"
                      "# geometry in work/frame_NNNN, where its checkpoint stays; the log goes to outputs/frame_NNNN.log.\n"
                      "# inputs/frames.csv names each frame.\n", slurm, flags=re.S)
    if n != 1 or f"LINK_LIMIT={LINK_LIMIT}" not in head or f"--cpus-per-task={CORES}" not in head \
            or f"--time={HOURS}:00:00" not in head or "module load gaussian/g09-d01" not in head:
        raise SystemExit("template Slurm script is not the expected package-99 script")
    files["run_gaussian.slurm"] = head.encode("utf-8")
    for small in (".gitignore", "export.list", "logs/README.txt"):
        files[small] = (args.template / small).read_bytes()
    # desk change at intake as package 112: the watchdog's stderr (Slurm logs) comes back with the outputs
    export = files["export.list"].decode("utf-8").splitlines()
    if "logs/*" not in export:
        export.append("logs/*")
    files["export.list"] = "".join(line + chr(10) for line in export).encode("utf-8")
    su = len(PICKS) * CORES * HOURS
    files["README.md"] = README.format(number=args.number, name=NAME, date=args.date, commit=commit, branch=branch,
                                       rows="\n".join(rows), cores=CORES, mem=MEM, hours=HOURS,
                                       link_h=LINK_LIMIT // 3600, su=su).encode("utf-8")
    meta = {"package": pkg.name, "owner": "ClusterMLIP", "generator": "experiments/acat_fe16_co_h/make_first_batch_package.py",
            "commit": commit, "branch": branch, "draft_dir": str(args.draft), "template_package": args.template.name,
            "watchdog_sha256": sha(watchdog), "cluster": "LONI QB4", "partition": "single", "gaussian": "gaussian/g09-d01",
            "level": "UBPW91/6-311++G*", "cpus_per_task": CORES, "memory_per_task": MEM, "time_hours": HOURS,
            "link_limit_s": LINK_LIMIT, "max_su": su, "charge": 0, "multiplicities": MULTIPLICITIES, "links_per_frame": 6,
            "array": [r["frame"] for r in table], "frames": table,
            "outputs": ["outputs/frame_NNNN.log", "outputs/frame_NNNN.seconds", "outputs/frame_NNNN.watchdog"],
            "files_sha256": {k: sha(v) for k, v in sorted(files.items())}, "submitted": False}
    for rel, data in files.items():
        path = pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.replace(b"\r\n", b"\n"))
    write_lf(pkg / "package.json", json.dumps(meta, indent=1) + "\n")
    print(pkg, len(files) + 1, "files;", su, "SU max; commit", commit)


if __name__ == "__main__":
    main()
