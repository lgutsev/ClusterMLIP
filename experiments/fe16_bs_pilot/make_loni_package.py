"""Write the tandem-fragment validation jobs as a LONI dispatch-desk package.

    python make_loni_package.py OUT_PARENT [--number 34]

Writes OUT_PARENT/<NN>_clustermlip_bs_tandem_validation/ for the loni_smoke_tests desk:
inputs/frame_NNNN.gjf (one per validation job, the layout submit_smokes.sh's
run_gaussian.slurm handler knows), run_gaussian.slurm (one array task per frame,
gaussian/g16-c01 on QB4 `single`), expected.json (archive targets for the check),
README.md, export.list, .gitignore, logs/README.txt and package.json (commit and
SHA-256 of every file). Every input passes fragment_tandem.inspect_tandem_input.

Check on the laptop after export and intake:
    PYTHONPATH=src python experiments/fe16_bs_pilot/make_validation_kit.py --check <package dir>
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from cluster_mlip.fragment_tandem import inspect_tandem_input, plan_from_input, render_tandem
from cluster_mlip.io import write_text_lf

sys.path.insert(0, str(Path(__file__).parent))
from make_validation_kit import jobs  # noqa: E402

NAME = "clustermlip_bs_tandem_validation"
CPUS = 4  # matches %nprocshared in make_validation_kit (NPROC)

SLURM = """#!/bin/bash
#SBATCH --job-name=bs-tandem-val
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --array=0-{last}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Written by ClusterMLIP experiments/fe16_bs_pilot/make_loni_package.py. One array task per frame:
# a three-link Gaussian tandem (fragment guess > Stable=Opt > Force) run in work/frame_NNNN, where its
# checkpoint stays; the log goes to outputs/frame_NNNN.log. inputs/frames.csv names each frame.
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:?submit from the package directory}}"
PKG=$PWD
frame=$(printf 'frame_%04d' "$SLURM_ARRAY_TASK_ID")
mkdir -p outputs "work/$frame"
# Which Gaussian builds this cluster offers (the archive used G09 D.01): recorded once per task.
{{ module -t avail gaussian 2>&1 || true; }} > "outputs/${{frame}}.modules.txt"
module load gaussian/g16-c01
scratch="${{TMPDIR:-/tmp}}/g16_${{SLURM_JOB_ID}}_${{frame}}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
cd "work/$frame"
GAUSS_SCRDIR="$scratch" g16 < "$PKG/inputs/$frame.gjf" > "$PKG/outputs/$frame.log"
"""

README = """# {number}_{name}: Gaussian fragment-tandem validation (ClusterMLIP)

Owner: ClusterMLIP. Generator: `experiments/fe16_bs_pilot/make_loni_package.py` on branch
`fix/bs-fragment-tandem` (commit `{commit}`). Regenerate there, not here. `package.json` records
every file's SHA-256.

## What this checks

ClusterMLIP rebuilt its broken-symmetry generator on the Gaussian fragment-guess tandem recovered
from the 2019 LG_Calcs archive (`docs/fragment-tandem-method.md` in ClusterMLIP). Each input has
three links on one checkpoint:

```
link 0  SP Guess=(Fragment=N)                         fragment guess, as archived
link 1  Stable=Opt Pop=Hirshfeld Geom=Checkpoint Guess=Read   fixed geometry
link 2  Force Pop=Hirshfeld Geom=Checkpoint Guess=Read
```

Seven small jobs test whether that works before any Fe16 pilot is run:

| frame | job | question |
|---|---|---|
{rows}

## Run (QB4)

```bash
bash submit_smokes.sh --dry-run {number}
bash submit_smokes.sh {number}
```

Uses `run_gaussian.slurm` and the existing Gaussian handler (`inputs/frame_*.gjf` →
`outputs/frame_*.log`). `gaussian/g16-c01`, `single`, {cpus} cores per task, 6 h cap (7 tasks, at
most {sus} SUs; expected minutes to about an hour each). Checkpoints stay in `work/` on LONI.

Each task also writes `outputs/frame_NNNN.modules.txt` (`module -t avail gaussian`). The archive
ran on G09 D.01. If QB4 offers it, ClusterMLIP may ask for a G09 rerun of the same inputs.

## Pass criteria (desk)

- Every `outputs/frame_*.log` ends in `Normal termination` (three of them per log).
- Physics is checked by ClusterMLIP, not the desk:
  `PYTHONPATH=src python experiments/fe16_bs_pilot/make_validation_kit.py --check <this package>`.
  That compares H1/H2 with the archived fragment-SCF energies, the link-1 guess ⟨S²⟩ and the first
  link-1 SCF energy, and V1 with V5. A clean Gaussian termination alone does not pass.

## Hand-off

Route: agent `ClusterMLIP`, source `experiments/fe16_bs_pilot/make_loni_package.py`. Check:
`outputs/frame_*.log` complete (3 Normal terminations). Next step: the ClusterMLIP agent runs
`make_validation_kit.py --check` and decides whether the Fe16 pilot (`fe16_bs_tandem_kit`) may go.
"""

PURPOSE = {
    "V1": "Fe₂ Fe(0,5)/Fe(0,−5), archived link-0 syntax",
    "V2": "same with `Guess=(Fragment=2,Only)` (no fragment SCFs)",
    "V3": "Fe₂ charge-separated Fe(+1,6)/Fe(−1,−6)",
    "V4": "Fe₂ default guess, `Stable=Opt` alone (control)",
    "V5": "exact repeat of V1 (reproducibility)",
    "H1": "replay of archived Fe2O2N2_DimAd_AFM_1 (neutral fragments)",
    "H2": "replay of archived Fe2O4H2_FeFe_TS_AFM_1 (Fe(+1) and O₂(2−) fragments)",
}


def main() -> None:
    parent = Path(sys.argv[1])
    number = sys.argv[sys.argv.index("--number") + 1] if "--number" in sys.argv else "34"
    out = parent / f"{number}_{NAME}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite {out}")
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src", "experiments"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit("commit src/ and experiments/ first: package.json must name a real commit")
    (out / "inputs").mkdir(parents=True)
    (out / "logs").mkdir()
    expected, frame_rows, table = {}, ["frame,label,job,purpose"], []
    for index, (label, job, ref) in enumerate(jobs()):
        frame = f"frame_{index:04d}"
        text = render_tandem(job)
        problems = inspect_tandem_input(text, job).problems
        if problems:
            raise SystemExit(f"{label}: {problems}")
        plan, stages = plan_from_input(text)
        write_text_lf(out / "inputs" / f"{frame}.gjf", text)
        entry = {"label": label, "job": job.name, "input": f"{frame}.gjf", "stages": stages,
                 "ideal_guess_s2": None if plan is None else plan.ideal_guess_s2}
        if ref:
            entry.update({
                "archived_fragment_scf_energies": [f["energy"] for f in ref["stage1_fragment_scf"]],
                "archived_unconverged_fragment_scf": ref["stage1_unconverged_fragment_scf"],
                "archived_link1_guess_s2": ref["stage2_guess_s2"],
                "archived_link1_first_scf": ref["stage2_first_scf"],
            })
        expected[label] = entry
        frame_rows.append(f"{frame},{label},{job.name},{PURPOSE[label]}")
        table.append(f"| {index:04d} | {label} `{job.name}` | {PURPOSE[label]} |")
    last = len(expected) - 1
    write_text_lf(out / "inputs" / "frames.csv", "\n".join(frame_rows) + "\n")
    write_text_lf(out / "expected.json", json.dumps(expected, indent=1, ensure_ascii=False) + "\n")
    write_text_lf(out / "run_gaussian.slurm", SLURM.format(last=last, cpus=CPUS))
    write_text_lf(out / "README.md", README.format(
        number=number, name=NAME, commit=commit, rows="\n".join(table), cpus=CPUS,
        sus=len(expected) * CPUS * 6))
    write_text_lf(out / "export.list", "outputs/*\n")
    write_text_lf(out / ".gitignore", "outputs/\nwork/\nlogs/*\n!logs/README.txt\n")
    write_text_lf(out / "logs" / "README.txt", "Slurm logs land here.\n")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        "package": f"{number}_{NAME}", "owner": "ClusterMLIP",
        "generator": "experiments/fe16_bs_pilot/make_loni_package.py", "commit": commit,
        "branch": "fix/bs-fragment-tandem", "cluster": "LONI QB4", "partition": "single",
        "gaussian": "gaussian/g16-c01", "array": [f"frame_{i:04d}" for i in range(last + 1)],
        "outputs": ["outputs/frame_NNNN.log", "outputs/frame_NNNN.modules.txt"],
        "files_sha256": {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "submitted": False,
    }
    write_text_lf(out / "package.json", json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(files) + 1} files, {last + 1} frames, commit {commit})")


if __name__ == "__main__":
    main()
