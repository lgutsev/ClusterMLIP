"""Write the Fe16 tandem smoke or pilot jobs as a LONI dispatch-desk package.

    python make_loni_fe16_package.py KIT OUT_PARENT --stage smoke|pilot [--number 35] [--gaussian g09|g16]

KIT is the generated `fe16_bs_tandem_kit` (its `smoke/`, `afm_pilot/` and `afm_control/` folders hold
inputs that passed `inspect_tandem_input` when written). The package uses the desk's Gaussian layout:
inputs/frame_NNNN.gjf -> outputs/frame_NNNN.log, one `single` array task per frame.

`--gaussian g09` (default) runs gaussian/g09-d01 on QB3, the code of the archived tandem: the user
(2026-10-16) keeps these Fe systems on G09. `--gaussian g16` runs gaussian/g16-c01 on QB4 (packages 34
and 36); package 34 showed G16 C.01 reproduces the archived G09 D.01 fragment SCFs to 1e-6 Ha, but G16
is secondary for the Fe16 tandem. QB4 has no G09 module.

`spin_jobs.csv` and `sites.csv` travel with the package, with input/output renamed to the frames, so
the analysis runs directly on the export:
    PYTHONPATH=src python experiments/fe16_bs_pilot/analyze.py <package dir> <results dir>

Gate (experiments/fe16_bs_pilot/README.md): the pilot stage only after the smoke is `ok` with a
measured wall time and a Hirshfeld table.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

from cluster_mlip.fragment_tandem import plan_from_input
from cluster_mlip.io import write_text_lf

# The pilot is the AFM pilot (kit afm_pilot/ + afm_control/): package 36's M = 49 smoke collapsed to the
# ferromagnetic state, and the PI's direction (2026-10-09) is to start at full AFM and flip outward. The
# old M = 49 pilot/ and charge_init/ folders stay in the kit, unused.
STAGES = {"smoke": ["smoke"], "pilot": ["afm_pilot", "afm_control"]}
NAMES = {"smoke": "clustermlip_fe16_bs_smoke", "pilot": "clustermlip_fe16_bs_afm_pilot"}
# smoke: the kit's estimate. pilot: package 36 measured 1.8 h (6473 s) for one M = 49 tandem on 12
# cores; low-Ms SCFs and Stable=Opt re-optimisations take longer, so ~6x that
HOURS = {"smoke": 48, "pilot": 12}
# Gaussian build -> where it runs. The mvapich2 load before G09 on QB3 comes from the kit's batch scripts.
PROFILES = {
    "g09": dict(cluster="QB3", host="qbc", module="gaussian/g09-d01", command="g09", node_mem="192 GB",
                modules="module load mvapich2\nmodule load gaussian/g09-d01"),
    "g16": dict(cluster="QB4", host="qbd", module="gaussian/g16-c01", command="g16", node_mem="256 GB",
                modules="module load gaussian/g16-c01"),
}

SLURM = """#!/bin/bash
#SBATCH --job-name=fe16-bs-{stage}
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --array=0-{last}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={hours}:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Written by ClusterMLIP experiments/fe16_bs_pilot/make_loni_fe16_package.py. One array task per frame:
# a Gaussian tandem (fragment guess > Stable=Opt > Force; the C0 control has no fragment link) run in
# work/frame_NNNN, where its checkpoint stays; the log goes to outputs/frame_NNNN.log.
# inputs/frames.csv names each frame. Memory comes from each input's %mem ({mem}); no --mem, as on QB4.
# LONI's module function reads unset variables: -u only after it (as package 24).
set -eo pipefail
cd "${{SLURM_SUBMIT_DIR:?submit from the package directory}}"
PKG=$PWD
frame=$(printf 'frame_%04d' "$SLURM_ARRAY_TASK_ID")
mkdir -p outputs "work/$frame"
{modules}
set -u
scratch="${{TMPDIR:-/tmp}}/{command}_${{SLURM_JOB_ID}}_${{frame}}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
cd "work/$frame"
start=$(date +%s)
# desk 2026-10-08 (package 36): write .seconds even when {command} ends in an Error termination (a
# Stable=Opt l508 abort is a result here, and the wall time is measured), then pass {command}'s status on
status=0
GAUSS_SCRDIR="$scratch" {command} <"$PKG/inputs/$frame.gjf" > "$PKG/outputs/$frame.log" || status=$?
printf '%s\\n' "$(( $(date +%s) - start ))" > "$PKG/outputs/$frame.seconds"
exit "$status"
"""

README = """# {number}_{name}: Fe16 broken-symmetry {stage} (ClusterMLIP)

Owner: ClusterMLIP. Generator: `experiments/fe16_bs_pilot/make_loni_fe16_package.py` on branch
`{branch}` (commit `{commit}`), from the kit `fe16_bs_tandem_kit` ({kit_folders}). Regenerate there,
not here. `package.json` records every file's SHA-256.

## The question

{question} Each frame is one fixed geometry, one multiplicity and one starting spin pattern, run as
the archived Gaussian fragment tandem on one checkpoint:

```
link 0  SP Guess=(Fragment=16)                                fragment guess, as archived
link 1  Stable=Opt Pop=Hirshfeld Geom=Checkpoint Guess=Read   fixed geometry
link 2  Force Pop=Hirshfeld Geom=Checkpoint Guess=Read
every link: IOP(5/13=1,5/36=1,8/11=1)
```

{stage_note}

| frame | geometry | M | pattern | start |
|---|---|---|---|---|
{rows}

This runs on `{module}`. The archived tandem is G09 D.01; package 34 (QB4) showed G16 C.01 reproduces
two archived 2019 G09 jobs (fragment SCFs, link-1 guess ⟨S²⟩ and first link-1 SCF to 1e-6 Ha), but G09
stays the reference code for these Fe systems.

## Run ({cluster}, submit from {host})

```bash
bash submit_smokes.sh --dry-run {number}
bash submit_smokes.sh {number}
```

Uses `run_gaussian.slurm` and the existing Gaussian handler (`inputs/frame_*.gjf` →
`outputs/frame_*.log`). `{module}`, `single`, {cpus} cores and {mem} per task ({cluster} CPU nodes
have {node_mem}), {hours} h cap. {n} task(s), at most {sus} SUs. {time_note} Checkpoints stay in `work/` on LONI and are not exported.
`outputs/frame_NNNN.seconds` records the wall time of each Gaussian run.

## Pass criteria (desk)

- Every `outputs/frame_*.log` ends in `Normal termination`, three times ({control_note}).
- Physics is checked by ClusterMLIP, not the desk:
  `PYTHONPATH=src python experiments/fe16_bs_pilot/analyze.py <this package> <results>`.
  A clean termination alone does not pass; an Error termination in Stable=Opt (l508) is a result
  to report, not a reason to resubmit.

## Hand-off

Route: agent `ClusterMLIP`, source `experiments/fe16_bs_pilot/make_loni_fe16_package.py`. Check:
`outputs/frame_*.log` and `outputs/frame_*.seconds`. Next step: {next_step}
"""

NOTES = {
    "smoke": dict(
        question="At a fixed Fe16 geometry with q = 0 and M = 49, does UBPW91/6-311++G* have more than one "
                 "stable SCF solution, with different site-resolved spins and different forces?",
        stage_note="This is the **smoke** (kit step 2): one Fe16 job, to measure the wall time and check that "
                   "a 16-fragment tandem completes and gives a Hirshfeld table before the 15-job pilot.",
        time_note="The cap is the kit's estimate for one Fe16 tandem, not a measurement; if it hits the cap, "
                  "report the time and the last SCF cycle reached rather than resubmitting.",
        control_note="one log",
        next_step="ClusterMLIP runs `analyze.py`, reads the wall time and Hirshfeld spins, and decides "
                  "whether the pilot package (same generator, `--stage pilot`) goes."),
    "pilot": dict(
        question="Where do the broken-symmetry UBPW91/6-311++G* solutions of neutral Fe16 lie, at fixed "
                 "geometry, starting from the fully antiferromagnetic arrangement (low Ms) and flipping "
                 "outward, and how do their energies, local spins and forces compare with the M = 49 "
                 "ferromagnetic state?",
        stage_note="This is the **AFM pilot** (kit step 3), rebuilt after the smoke (package 36): at M = 49 "
                   "the flipped central Fe of P1 collapsed and the SCF converged to the ferromagnetic state "
                   "(<S²> 600.01, all 16 Hirshfeld spins positive). At Ms = 24 there is no room for a "
                   "flipped site, so, on the PI's direction (2026-10-09), every start here is at or near "
                   "full AFM, every Fe a neutral quintet fragment (±4 unpaired):\n\n"
                   "- **A0** (M = 1): the Fe–Fe bonds (< 3.0 Å) split into two equal 8/8 sublattices with "
                   "the most antiparallel bonds (central Fe α by convention; ties broken towards short bonds);\n"
                   "- **A1** (M = 9): A0 with its cheapest β → α flip (fewest antiparallel bonds lost);\n"
                   "- **A2** (M = 17): A1 with the next cheapest flip;\n"
                   "- **A1C** (M = 9): A0 with the central Fe reversed against its former sublattice;\n"
                   "- **C0** (M = 1, geometry 1 only): default-guess control, no fragment link.\n\n"
                   "`sites.csv` lists, per atom, the patterns in which it starts β.",
        time_note="The cap is ~6x the smoke's measured 1.8 h (one M = 49 tandem, 12 cores); low-Ms SCFs "
                  "and Stable=Opt re-optimisations are expected to take longer. If a task hits it, report the "
                  "time and the last SCF cycle reached rather than resubmitting.",
        control_note="two for the C0 control, which has no fragment link",
        next_step="ClusterMLIP runs `analyze.py` and reports, per geometry, the energy of each stable "
                  "solution across M against the M = 49 reference, with ⟨S²⟩ and Hirshfeld sublattice "
                  "spins; the converged low-Ms solutions are the starting points for the next flips."),
}

_GEOM_RE = re.compile(r"^a-(fe16-[^_]+)__")
_PAT_RE = re.compile(r"__bs-m(\d+)-(\w+)\.gjf$")
_MEM_RE = re.compile(r"^%mem=(\S+)", re.M | re.I)
_NPROC_RE = re.compile(r"^%nprocshared=(\d+)", re.M | re.I)


def main() -> None:
    args = sys.argv[1:]
    kit, parent = Path(args[0]), Path(args[1])
    stage = args[args.index("--stage") + 1]
    number = args[args.index("--number") + 1] if "--number" in args else "35"
    profile = PROFILES[args[args.index("--gaussian") + 1] if "--gaussian" in args else "g09"]
    name = NAMES[stage]
    out = parent / f"{number}_{name}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite {out}")
    repo = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,
                                    check=True).stdout.strip()
    commit, branch = git("rev-parse", "--short", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    if git("status", "--porcelain", "--", "src", "experiments"):
        raise SystemExit("commit src/ and experiments/ first: package.json must name a real commit")

    (out / "inputs").mkdir(parents=True)
    (out / "logs").mkdir()
    frames, table, job_rows, sites, mems, cpus = [["frame", "kit_folder", "kit_input"]], [], [], [], set(), set()
    index = 0
    for folder in STAGES[stage]:
        sub = kit / folder
        rows = list(csv.DictReader((sub / "spin_jobs.csv").open(encoding="utf-8")))
        for kit_input in dict.fromkeys(r["input"] for r in rows):
            text = (sub / kit_input).read_text(encoding="utf-8").replace("\r\n", "\n")
            plan, n_stages = plan_from_input(text)
            if n_stages != (2 if plan is None else 3):
                raise SystemExit(f"{kit_input}: {n_stages} links, expected a {'2' if plan is None else '3'}-link tandem")
            mems.update(_MEM_RE.findall(text))
            cpus.update(int(v) for v in _NPROC_RE.findall(text))
            frame = f"frame_{index:04d}"
            write_text_lf(out / "inputs" / f"{frame}.gjf", text)
            frames.append([frame, folder, Path(kit_input).name])
            m, pattern = _PAT_RE.search(kit_input).groups()
            pattern = pattern.upper()
            start = "default guess (control)" if plan is None else (
                "charge-separated fragments" if pattern.endswith("Q") else "neutral fragments")
            table.append(f"| {index:04d} | {_GEOM_RE.search(Path(kit_input).name).group(1)} | {m} | {pattern} | {start} |")
            for r in rows:
                if r["input"] == kit_input:
                    job_rows.append({**r, "input": f"inputs/{frame}.gjf", "output": f"{frame}.log",
                                     "kit_folder": folder, "kit_input": kit_input})
            index += 1
        if (sub / "sites.csv").exists():
            sites.extend(csv.DictReader((sub / "sites.csv").open(encoding="utf-8")))
    if len(cpus) != 1 or len(mems) != 1:
        raise SystemExit(f"inputs disagree on resources: nprocshared {cpus}, mem {mems}")
    n, ncpu, mem = index, cpus.pop(), mems.pop()

    def write_csv(path, rows, fields=None):
        buffer = io.StringIO()
        if fields is None:
            csv.writer(buffer, lineterminator="\n").writerows(rows)
        else:
            w = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        write_text_lf(path, buffer.getvalue())

    write_csv(out / "inputs" / "frames.csv", frames)
    write_csv(out / "spin_jobs.csv", job_rows, list(job_rows[0]))
    if sites:
        unique = list({tuple(s.items()): s for s in sites}.values())
        write_csv(out / "sites.csv", unique, list(unique[0]))
    hours = HOURS[stage]
    write_text_lf(out / "run_gaussian.slurm", SLURM.format(stage=stage, last=n - 1, cpus=ncpu, hours=hours, mem=mem,
                                                             **profile))
    write_text_lf(out / "README.md", README.format(
        number=number, name=name, stage=stage, branch=branch, commit=commit, kit_folders=", ".join(STAGES[stage]),
        rows="\n".join(table), cpus=ncpu, mem=mem, hours=hours, n=n, sus=n * ncpu * hours, **profile, **NOTES[stage]))
    write_text_lf(out / "export.list", "outputs/*\n")
    write_text_lf(out / ".gitignore", "outputs/\nwork/\nlogs/*\n!logs/README.txt\n")
    write_text_lf(out / "logs" / "README.txt", "Slurm logs land here.\n")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        "package": f"{number}_{name}", "owner": "ClusterMLIP",
        "generator": "experiments/fe16_bs_pilot/make_loni_fe16_package.py", "commit": commit, "branch": branch,
        "kit": str(kit), "kit_folders": STAGES[stage], "cluster": f"LONI {profile['cluster']}", "partition": "single",
        "gaussian": profile["module"], "cpus_per_task": ncpu, "memory_per_task": mem, "time_hours": hours,
        "array": [f"frame_{i:04d}" for i in range(n)],
        "outputs": ["outputs/frame_NNNN.log", "outputs/frame_NNNN.seconds"],
        "files_sha256": {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "submitted": False,
    }
    write_text_lf(out / "package.json", json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(files) + 1} files, {n} frames, {ncpu} cores, {mem}, {profile['module']} on "
          f"{profile['cluster']}, commit {commit})")


if __name__ == "__main__":
    main()
