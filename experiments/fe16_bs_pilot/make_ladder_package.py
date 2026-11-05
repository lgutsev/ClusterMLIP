"""Write the Fe16 flip-down spin-ladder test as a LONI dispatch-desk package.

    python make_ladder_package.py KIT_A0_INPUT OUT_PARENT [--number NN]

Why: three runs (packages 61, 78 and 82, desk numbering) never produced a converged low-spin Fe16
state when they started from a fresh full-AFM fragment guess (A0, M = 1).
- In the first-order SCF, every run hit the 129-cycle cap.
- XQC's quadratic stage crawled to a plateau about 6.7 eV above FM and never converged (REVIEW_82.md).
The user's direction (2026-11-04) is to switch to flip-down ladders from the converged FM state.
Each step then starts from a converged wavefunction one spin flip away.

One input per frame, one checkpoint, at the fixed geometry fe16-200-49-gf (the A0 runs' geometry,
coordinates taken from the kit A0 input):

    M = 49   Stable=Opt Pop=Hirshfeld           default guess, coordinates (package 36 converged the
                                                FM state here in 98 cycles: -20221.7601 Ha)
    M = 49   Force Pop=Hirshfeld                Geom=Checkpoint Guess=Read
    M = 47   Stable=Opt Pop=Hirshfeld           Geom=Checkpoint Guess=Read  (one alpha -> beta)
    M = 47   Force Pop=Hirshfeld                Geom=Checkpoint Guess=Read
    ...      down to M = 33

Stable=Opt in every step: package 34 showed that identical inputs can land on different SCF solutions,
so each M is taken to a stable one before Force.
- frame_0000 uses the group's standard ladder route (DEFAULT_SPIN_ROUTE without Opt).
- frame_0001 adds XQC to SCF=(...), as a fallback if a step's first-order SCF hits the 129-cycle cap.

IOP(5/13=1,5/36=1,8/11=1) is in every link. Stable=Opt stands alone as the job type: the first
staging (desk 91, job 1163098) wrote "SP Stable=Opt", and G09 turned that into an l108 scan step that
failed at once with "No variables in Z-matrix" (package 78's working Stable=Opt link has no SP).
`job_types` refuses any route with more than one job-type keyword. The run uses G09 D.01 on QB4 under the log-growth
watchdog from make_scf_gate_package.py.
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
from make_scf_gate_package import GRACE, LINK_LIMIT, SLURM, STALL_LIMIT, WATCHDOG  # noqa: E402

NAME = "clustermlip_fe16_flipdown_ladder"
JOB = "fe16-ladder"
HIGH, LOW = 49, 33
HOURS = 48
FM_REFERENCE = -20221.7600899  # Ha, package 36 (G16), M = 49, same geometry
BASE = ("#p UBPW91/6-311++G* SCF=({scf}VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm {job} "
        "Pop=Hirshfeld IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine")
VARIANTS = [("standard", ""), ("xqc", "XQC,")]
_ATOM_RE = re.compile(r"^\s*Fe(?:\(Fragment=\d+\))?(\s+\S+\s+\S+\s+\S+)\s*$")

README = """# {number}_{name}: Fe16 flip-down spin ladder (ClusterMLIP)

Owner: ClusterMLIP. Generator: `experiments/fe16_bs_pilot/make_ladder_package.py` on branch `{branch}`
(commit `{commit}`); coordinates from `{kit_input}`. Regenerate there, not here. `package.json` records
every file's SHA-256.

## The question

Starting from the converged ferromagnetic Fe16 state (M = 49), does a fixed-geometry flip-down ladder
give a converged, stable UBPW91/6-311++G* state at every M down to {low}? If so, how do energy, ⟨S²⟩,
Hirshfeld spins and forces change along it?

This replaces the fresh full-AFM (A0) start, which failed in packages 61, 78 and 82 (desk numbering):
no converged state, and an XQC plateau about 6.7 eV above FM.

Geometry fe16-200-49-gf, neutral, fixed. One input per frame, one checkpoint, {nlinks} links:

```
M = 49        Stable=Opt Pop=Hirshfeld      default guess, coordinates
M = 49        Force Pop=Hirshfeld           Geom=Checkpoint Guess=Read
M = 47 ... {low}   per M: Stable=Opt, then Force, both Geom=Checkpoint Guess=Read
every link: IOP(5/13=1,5/36=1,8/11=1)
```

| frame | route | SCF options |
|---|---|---|
| 0000 | standard ladder route (no Opt) | `SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)` |
| 0001 | same + XQC fallback | `SCF=(XQC,VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)` |

## Run (QB4, submit from qbd)

`run_gaussian.slurm`, array 0-1, `gaussian/g09-d01`, `single`, {cpus} cores and {mem} per task, {hours} h
cap, at most {sus} SUs. Gaussian runs under `gaussian_watchdog.sh`, which treats log growth as progress
and kills the G09 process group in three cases:
- `error_hang`: {grace} s after an Error termination;
- `stall`: no log growth for {stall_hours} h;
- `link_timeout`: one link has run for {link_hours} h.

`outputs/frame_NNNN.watchdog` holds the reason (an empty file means G09 exited by itself), and
`outputs/frame_NNNN.seconds` the wall time. Package 36 needed about 1.6 h for the M = 49 Stable=Opt link
and 8 min for Force (G16, 12 cores).

## Pass criteria (desk)

- Every `outputs/frame_*.log`, `.seconds` and `.watchdog` file exists, and the watchdog reasons are
  reported.
- {nterm} Normal terminations per log means the whole ladder ran. Fewer is a result (report the last M
  reached), not a package failure. Do not resubmit with a longer limit without asking.
- Physics is checked by ClusterMLIP: for each M, whether the SCF converged, the stable energy against
  the FM reference ({fm} Ha), ⟨S²⟩, Hirshfeld spins and forces.

## Hand-off

Route: agent `ClusterMLIP`, source `experiments/fe16_bs_pilot/make_ladder_package.py`.
"""


_JOB_TYPES = ("SP", "Opt", "Force", "Freq", "Stable", "IRC", "Scan")


def job_types(route: str) -> list[str]:
    """Job-type keywords in a Gaussian route (Stable=Opt counts as Stable, not Opt)."""
    found = []
    for token in route.split()[1:]:
        key = token.split("=", 1)[0].split("(", 1)[0]
        for job in _JOB_TYPES:
            if key.lower() == job.lower():
                found.append(job)
    return found


def coordinates(kit_text: str) -> list[str]:
    stage0 = split_input(kit_text.replace("\r\n", "\n"))[0]
    atoms = []
    for line in stage0.atom_lines:
        m = _ATOM_RE.match(line)
        if not m:
            raise SystemExit(f"unexpected atom line in the kit input: {line!r}")
        atoms.append("Fe" + m.group(1))
    if len(atoms) != 16:
        raise SystemExit(f"expected 16 Fe atoms, found {len(atoms)}")
    return atoms


def ladder_input(atoms: list[str], scf: str, chk: str, mem: str, cpus: int) -> str:
    header = f"%chk={chk}\n%mem={mem}\n%nprocshared={cpus}\n"
    links = []
    for m in range(HIGH, LOW - 1, -2):
        for job in ("Stable=Opt", "Force"):
            first = not links
            route = BASE.format(scf=scf, job=job) + ("" if first else " Geom=Checkpoint Guess=Read")
            kind = "stable" if "Stable" in job else "force"
            title = (f"ClusterMLIP Fe16 flip-down ladder; geometry=fe16-200-49-gf; multiplicity={m}; "
                     f"step={kind}; scf={scf.rstrip(',') or 'standard'}")
            body = f"{header}{route}\n\n{title}\n\n0 {m}\n"
            if first:
                body += "\n".join(atoms) + "\n"
            links.append(body + "\n")
    text = "--Link1--\n".join(links)
    stages = split_input(text)
    expected = [m for m in range(HIGH, LOW - 1, -2) for _ in (0, 1)]
    assert [s.charge_multiplicity for s in stages] == [[0, m] for m in expected]
    assert all("IOP(5/13=1,5/36=1,8/11=1)" in s.route for s in stages)
    assert [job_types(s.route) for s in stages] == [["Stable"], ["Force"]] * (len(stages) // 2),         [s.route for s in stages]
    assert len(stages[0].atom_lines) == 16 and all(not s.atom_lines for s in stages[1:])
    assert all("Geom=Checkpoint Guess=Read" in s.route for s in stages[1:])
    assert "Guess=" not in stages[0].route and " Opt " not in f" {stages[0].route} ".replace("Stable=Opt", "")
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
    frames = [["frame", "variant", "multiplicities"]]
    for i, (variant, scf) in enumerate(VARIANTS):
        frame = f"frame_{i:04d}"
        chk = f"fe16-200-49-gf-ladder-m{HIGH}-m{LOW}-{variant}.chk"
        write_text_lf(out / "inputs" / f"{frame}.gjf", ladder_input(atoms, scf, chk, mem, cpus))
        frames.append([frame, variant, ";".join(str(m) for m in range(HIGH, LOW - 1, -2))])
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(frames)
    write_text_lf(out / "inputs" / "frames.csv", buffer.getvalue())
    n = len(VARIANTS)
    nlinks = 2 * len(range(HIGH, LOW - 1, -2))
    slurm = SLURM.replace(
        "make_scf_gate_package.py. One array task per frame:\n"
        "# a 4-link G09 tandem (fragment guess > SP convergence > Stable=Opt > Force) in work/frame_NNNN, where",
        "make_ladder_package.py. One array task per frame:\n"
        "# a G09 flip-down spin ladder (per M: Stable=Opt > Force) in work/frame_NNNN, where")
    assert "make_ladder_package.py" in slurm
    fmt = dict(number=number, name=NAME, job=JOB, branch=branch, commit=commit, kit_input=kit_input.name,
               last=n - 1, cpus=cpus, mem=mem, hours=HOURS, sus=n * cpus * HOURS, grace=GRACE,
               stall_limit=STALL_LIMIT, stall_hours=STALL_LIMIT // 3600, link_limit=LINK_LIMIT,
               link_hours=LINK_LIMIT // 3600, low=LOW, nlinks=nlinks, nterm=nlinks, fm=f"{FM_REFERENCE:.4f}")
    write_text_lf(out / "run_gaussian.slurm", slurm.format(**fmt))
    write_text_lf(out / "gaussian_watchdog.sh", WATCHDOG)
    write_text_lf(out / "README.md", README.format(**fmt))
    write_text_lf(out / "export.list", "outputs/*\n")
    write_text_lf(out / ".gitignore", "outputs/\nwork/\nlogs/*\n!logs/README.txt\n")
    write_text_lf(out / "logs" / "README.txt", "Slurm logs land here.\n")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        "package": f"{number}_{NAME}", "owner": "ClusterMLIP",
        "generator": "experiments/fe16_bs_pilot/make_ladder_package.py", "commit": commit, "branch": branch,
        "kit_input": str(kit_input), "geometry": "fe16-200-49-gf", "multiplicities": list(range(HIGH, LOW - 1, -2)),
        "fm_reference_ha": FM_REFERENCE, "cluster": "LONI QB4", "partition": "single",
        "gaussian": "gaussian/g09-d01", "cpus_per_task": cpus, "memory_per_task": mem, "time_hours": HOURS,
        "watchdog": {"grace_s": GRACE, "stall_limit_s": STALL_LIMIT, "link_limit_s": LINK_LIMIT},
        "array": [f"frame_{i:04d}" for i in range(n)], "variants": [v for v, _ in VARIANTS],
        "links_per_frame": nlinks,
        "outputs": ["outputs/frame_NNNN.log", "outputs/frame_NNNN.seconds", "outputs/frame_NNNN.watchdog"],
        "files_sha256": {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "submitted": False,
    }
    write_text_lf(out / "package.json", json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(files) + 1} files, {n} frames x {nlinks} links, {cpus} cores, {mem}, "
          f"{HOURS} h, commit {commit})")


if __name__ == "__main__":
    main()
