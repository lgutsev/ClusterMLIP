"""Write the Fe16 ladder-then-optimize package for the LONI dispatch desk.

    python make_ladder_opt_package.py KIT_A0_INPUT OUT_PARENT [--number NN]

Why: package 91 (REVIEW_91.md) gave the first converged low-spin Fe16 states, by flipping down from
the converged FM state at the fixed M = 49 geometry. These were vertical energies:
- M = 47: +0.52 eV
- M = 45: +1.24 eV
- M = 41: +1.57 eV, with the central Fe4 flipped to beta (Hirshfeld -2.57)
The run's checkpoints were overwritten by later links, so this package regenerates each rung and then
relaxes the geometry at that M, to get adiabatic gaps against M = 49.

One frame per target M (47, 45, 41). Each input is package 91 frame_0000's links, verbatim down to
the target M (from make_ladder_package.ladder_input; only the %chk name differs), followed by one Opt
link:

    M = 49 ... target   Stable=Opt, Force per M          (package 91's links, standard route)
    target              Opt Pop=Hirshfeld Geom=Checkpoint Guess=Read

Repeating 91's exact link sequence matters for two reasons:
- M = 45 and 41 converged only in their Force links, after the Stable=Opt link hit the 129-cycle cap;
- M = 41's central-beta state came from that path.
IOP(5/13=1,5/36=1,8/11=1) is in every link. The run uses G09 D.01 on QB4 under the log-growth watchdog,
with a 12 h per-link cap. In 91 every converged rung took well under 2 h; only QC crawls ran to 20 h.
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

from cluster_mlip.fragment_tandem import split_input
from cluster_mlip.io import write_text_lf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_ladder_package import BASE, FM_REFERENCE, HIGH, coordinates, job_types, ladder_input  # noqa: E402
from make_scf_gate_package import GRACE, SLURM, STALL_LIMIT, WATCHDOG  # noqa: E402

NAME = "clustermlip_fe16_ladder_opt"
JOB = "fe16-ladder-opt"
TARGETS = [47, 45, 41]
HOURS = 48
LINK_LIMIT = 12 * 3600
VERTICAL_EV = {47: 0.518, 45: 1.240, 41: 1.572}  # package 91, fixed M = 49 geometry

README = """# {number}_{name}: Fe16 ladder, then geometry optimization (ClusterMLIP)

Owner: ClusterMLIP. Generator: `experiments/fe16_bs_pilot/make_ladder_opt_package.py` on branch `{branch}`
(commit `{commit}`); coordinates from `{kit_input}`. Regenerate there, not here. `package.json` records
every file's SHA-256.

## The question

Package 91 (desk numbering) produced converged low-spin Fe16 states by flipping down from FM at the
fixed M = 49 geometry. Those were vertical energies:
- M = 47: +0.52 eV
- M = 45: +1.24 eV
- M = 41: +1.57 eV, with the central Fe4 β (Hirshfeld −2.57)

What are the **adiabatic** gaps once each state relaxes its own geometry? Does the central-β M = 41
state stay central-β, and does it come down toward M = 49?

| frame | target M | links | ladder part |
|---|---|---|---|
{rows}

Each input repeats package 91 frame_0000's links verbatim down to the target M (Stable=Opt + Force per
M, standard route; only the `%chk` name differs), then adds one
`Opt Pop=Hirshfeld Geom=Checkpoint Guess=Read` link at the target M. IOP(5/13=1,5/36=1,8/11=1) is in
every link.

## Run (QB4, submit from qbd)

`run_gaussian.slurm`, array 0-{last}, `gaussian/g09-d01`, `single`, {cpus} cores and {mem} per task,
{hours} h cap, at most {sus} SUs. Watchdog (log growth = progress):
- `error_hang`: {grace} s after an Error termination;
- `stall`: no log growth for {stall_hours} h;
- `link_timeout`: {link_hours} h in one link.

## Pass criteria (desk)

- Every `outputs/frame_*.log`, `.seconds` and `.watchdog` file exists, and the watchdog reasons are
  reported.
- The full run gives the Normal-termination count per frame listed in the table. Fewer is a result
  (report the last link reached), not a package failure. Do not resubmit with a longer limit without
  asking.
- Physics is checked by ClusterMLIP:
  - whether each Opt converged;
  - its final energy against the M = 49 optimum ({fm} Ha);
  - ⟨S²⟩ and the Hirshfeld spins, including whether Fe4 stays β at M = 41;
  - whether the SCF converged at every Opt step (IOP(5/13=1) hides failures).

## Hand-off

Route: agent `ClusterMLIP`, source `experiments/fe16_bs_pilot/make_ladder_opt_package.py`.
"""


def opt_input(atoms: list[str], target: int, chk: str, mem: str, cpus: int) -> str:
    """Package 91 frame_0000's links down to `target`, then an Opt link at `target`."""
    full = ladder_input(atoms, "", chk, mem, cpus)  # standard route, M = 49 .. 33
    links = full.split("--Link1--\n")
    keep = 2 * ((HIGH - target) // 2 + 1)
    header = f"%chk={chk}\n%mem={mem}\n%nprocshared={cpus}\n"
    route = BASE.format(scf="", job="Opt") + " Geom=Checkpoint Guess=Read"
    title = (f"ClusterMLIP Fe16 ladder-opt; geometry=fe16-200-49-gf start; multiplicity={target}; "
             f"step=opt; scf=standard")
    text = "--Link1--\n".join(links[:keep] + [f"{header}{route}\n\n{title}\n\n0 {target}\n\n"])
    stages = split_input(text)
    expected = [m for m in range(HIGH, target - 1, -2) for _ in (0, 1)] + [target]
    assert [s.charge_multiplicity for s in stages] == [[0, m] for m in expected]
    assert [job_types(s.route) for s in stages] == [["Stable"], ["Force"]] * (keep // 2) + [["Opt"]]
    assert all("IOP(5/13=1,5/36=1,8/11=1)" in s.route for s in stages)
    assert all("Geom=Checkpoint Guess=Read" in s.route for s in stages[1:])
    return text


def main() -> None:
    args = sys.argv[1:]
    kit_input, parent = Path(args[0]), Path(args[1])
    number = args[args.index("--number") + 1] if "--number" in args else "NN"
    out = parent / f"{number}_{NAME}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite {out}")
    repo = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,
                                    check=True).stdout.strip()
    commit, branch = git("rev-parse", "--short", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    if git("status", "--porcelain", "--", "src", "experiments"):
        raise SystemExit("commit src/ and experiments/ first: package.json must name a real commit")

    text = kit_input.read_text(encoding="utf-8")
    atoms = coordinates(text)
    mem = re.search(r"(?m)^%mem=(\S+)", text).group(1)
    cpus = int(re.search(r"(?m)^%nprocshared=(\d+)", text).group(1))

    (out / "inputs").mkdir(parents=True)
    (out / "logs").mkdir()
    frames, rows = [["frame", "target_multiplicity", "links", "vertical_ev_pkg91"]], []
    for i, target in enumerate(TARGETS):
        frame = f"frame_{i:04d}"
        body = opt_input(atoms, target, f"fe16-200-49-gf-ladder-opt-m{target}.chk", mem, cpus)
        write_text_lf(out / "inputs" / f"{frame}.gjf", body)
        nlinks = body.count("--Link1--") + 1
        frames.append([frame, target, nlinks, VERTICAL_EV[target]])
        rows.append(f"| {i:04d} | {target} | {nlinks} | 49 → {target} (vertical +{VERTICAL_EV[target]:.2f} eV in 91), then Opt |")
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(frames)
    write_text_lf(out / "inputs" / "frames.csv", buffer.getvalue())
    n = len(TARGETS)
    slurm = SLURM.replace(
        "make_scf_gate_package.py. One array task per frame:\n"
        "# a 4-link G09 tandem (fragment guess > SP convergence > Stable=Opt > Force) in work/frame_NNNN, where",
        "make_ladder_opt_package.py. One array task per frame:\n"
        "# a G09 flip-down ladder to the target M, then Opt at that M, in work/frame_NNNN, where")
    assert "make_ladder_opt_package.py" in slurm
    fmt = dict(number=number, name=NAME, job=JOB, branch=branch, commit=commit, kit_input=kit_input.name,
               rows="\n".join(rows), last=n - 1, cpus=cpus, mem=mem, hours=HOURS, sus=n * cpus * HOURS,
               grace=GRACE, stall_limit=STALL_LIMIT, stall_hours=STALL_LIMIT // 3600, link_limit=LINK_LIMIT,
               link_hours=LINK_LIMIT // 3600, fm=f"{FM_REFERENCE:.4f}")
    write_text_lf(out / "run_gaussian.slurm", slurm.format(**fmt))
    write_text_lf(out / "gaussian_watchdog.sh", WATCHDOG)
    write_text_lf(out / "README.md", README.format(**fmt))
    write_text_lf(out / "export.list", "outputs/*\n")
    write_text_lf(out / ".gitignore", "outputs/\nwork/\nlogs/*\n!logs/README.txt\n")
    write_text_lf(out / "logs" / "README.txt", "Slurm logs land here.\n")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        "package": f"{number}_{NAME}", "owner": "ClusterMLIP",
        "generator": "experiments/fe16_bs_pilot/make_ladder_opt_package.py", "commit": commit, "branch": branch,
        "kit_input": str(kit_input), "start_geometry": "fe16-200-49-gf", "targets": TARGETS,
        "fm_reference_ha": FM_REFERENCE, "cluster": "LONI QB4", "partition": "single",
        "gaussian": "gaussian/g09-d01", "cpus_per_task": cpus, "memory_per_task": mem, "time_hours": HOURS,
        "watchdog": {"grace_s": GRACE, "stall_limit_s": STALL_LIMIT, "link_limit_s": LINK_LIMIT},
        "array": [f"frame_{i:04d}" for i in range(n)],
        "outputs": ["outputs/frame_NNNN.log", "outputs/frame_NNNN.seconds", "outputs/frame_NNNN.watchdog"],
        "files_sha256": {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "submitted": False,
    }
    write_text_lf(out / "package.json", json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(files) + 1} files, {n} frames, {cpus} cores, {mem}, {HOURS} h, commit {commit})")


if __name__ == "__main__":
    main()
