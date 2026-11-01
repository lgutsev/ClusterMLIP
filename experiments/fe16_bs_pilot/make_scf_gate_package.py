"""Write the Fe16 A0 SCF-convergence packages for the LONI dispatch desk.

    python make_scf_gate_package.py KIT_A0_INPUT OUT_PARENT --stage gate|xqc [--number NN]

Why: in package 61 (desk numbering; staged as 41), the first full-cluster Fe16 SCF never converged.
It ran in the Stable=Opt link from the fragment guess and hit the 129-cycle cap in 13/13 frames
(dE ~ Ha per cycle, negative HOMO-LUMO gap). IOP(5/13=1) let l508 run on that non-stationary density,
so every stability analysis was meaningless (REVIEW_61.md).

Both stages take the kit's A0 input (M = 1, full AFM, geometry fe16-200-49-gf) and change only the
SCF strategy and the link order:

    link 0  SP Guess=(Fragment=16)                          kit A0 link (checkpoint name suffixed)
    link 1  SP Geom=Checkpoint Guess=Read Pop=Hirshfeld     converge the SCF on its own
    link 2  Stable=Opt Geom=Checkpoint Guess=Read           on a converged wavefunction
    link 3  Force Geom=Checkpoint Guess=Read

    --stage gate  2 frames, XQC vs Fermi added to SCF=(...) in links 1-3, 12 h cap.
                  This is package 78 (desk numbering; staged as 76), generated at 6f2edba with the
                  older watchdog. Regenerating it now gives the log-growth watchdog.
    --stage xqc   1 frame, XQC only, 48 h cap. Package 78 showed that XQC's quadratic stage converges
                  slowly but monotonically. It ran 98 steepest-descent iterations to -20221.5136 Ha
                  before the old watchdog killed it, because that watchdog counted only "Normal
                  termination" lines as progress (REVIEW_78.md).

Every other route token stays, including IOP(5/13=1,5/36=1,8/11=1) in every link (never drop it from
an Fe route). Runs G09 D.01 on QB4 through gaussian_watchdog.sh, which treats log growth as progress.
It kills the whole G09 process group in three cases:
- GRACE s after an Error termination (package 61: G09 hung after an l508 error);
- after STALL_LIMIT s with no log growth (a real stall);
- after LINK_LIMIT s in one link (a hard cap, e.g. LinEq1 diverging while it still prints).
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

from cluster_mlip.fragment_tandem import plan_from_input, split_input
from cluster_mlip.io import write_text_lf

GRACE = 180            # s after an Error termination before the watchdog kills G09
STALL_LIMIT = 2 * 3600  # s without log growth; a silent l703 Force step on Fe16 can take ~1 h
LINK_LIMIT = 20 * 3600  # s in one link; the 78 QC stage ran ~100 s/iteration in steepest descent
STAGES = {
    "gate": dict(aids=["XQC", "Fermi"], hours=12, name="clustermlip_fe16_scf_gate", job="fe16-scf-gate",
                 title="Fe16 A0 SCF-convergence gate",
                 question="Does the first full-cluster SCF converge from the full-AFM fragment guess (A0) "
                          "when it runs in its own SP link with a convergence aid? Two frames, identical "
                          "except for the aid."),
    "xqc": dict(aids=["XQC"], hours=48, name="clustermlip_fe16_a0_xqc", job="fe16-a0-xqc",
                title="Fe16 A0 XQC rerun",
                question="Does XQC's quadratic stage take the full-AFM fragment guess (A0) to a "
                         "converged M = 1 SCF, and then through Stable=Opt and Force, when it gets the "
                         "time? In package 78 (desk numbering) the quadratic stage was still descending "
                         "monotonically (98 steepest-descent iterations, -20202.53 -> -20221.5136 Ha, "
                         "dE 9.5e-5 Ha) when the old watchdog stopped it, because that watchdog counted "
                         "only Normal terminations as progress. One frame, identical to 78's frame_0000; "
                         "only the watchdog and the time cap change."),
}
_SCF_RE = re.compile(r"SCF=\(([^)]*)\)", re.I)

SLURM = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --array=0-{last}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={hours}:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Written by ClusterMLIP experiments/fe16_bs_pilot/make_scf_gate_package.py. One array task per frame:
# a 4-link G09 tandem (fragment guess > SP convergence > Stable=Opt > Force) in work/frame_NNNN, where
# its checkpoint stays; the log goes to outputs/frame_NNNN.log. inputs/frames.csv names each frame.
# Memory comes from each input's %mem ({mem}); no --mem, as on QB4.
# LONI's module function reads unset variables: -u only after it (as package 24).
set -eo pipefail
cd "${{SLURM_SUBMIT_DIR:?submit from the package directory}}"
PKG=$PWD
frame=$(printf 'frame_%04d' "$SLURM_ARRAY_TASK_ID")
mkdir -p outputs "work/$frame"
module load gaussian/g09-d01
set -u
scratch="${{TMPDIR:-/tmp}}/g09_${{SLURM_JOB_ID}}_${{frame}}"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
cd "work/$frame"
start=$(date +%s)
# .seconds is written whatever the outcome (desk 2026-10-08); outputs/frame_NNNN.watchdog says
# whether the watchdog stopped G09 and why (empty file = G09 exited by itself)
status=0
GAUSS_SCRDIR="$scratch" GRACE={grace} STALL_LIMIT={stall_limit} LINK_LIMIT={link_limit} \\
  bash "$PKG/gaussian_watchdog.sh" g09 "$PKG/inputs/$frame.gjf" "$PKG/outputs/$frame.log" \\
  "$PKG/outputs/$frame.watchdog" || status=$?
printf '%s\\n' "$(( $(date +%s) - start ))" > "$PKG/outputs/$frame.seconds"
exit "$status"
"""

WATCHDOG = r"""#!/bin/bash
# gaussian_watchdog.sh COMMAND INPUT LOG REASON_FILE
# Runs `COMMAND < INPUT > LOG` in its own process group. Log growth is progress (an SCF cycle, a QC
# iteration, a LinEq1 step). The watchdog kills the whole group when
#  - error_hang:   the log shows an Error termination and Gaussian is still running GRACE s later
#                  (package 61: G09 crashed in its exit path after an l508 error and held the node);
#  - stall:        the log has not grown for STALL_LIMIT s;
#  - link_timeout: one link (between "Normal termination" lines) has run for LINK_LIMIT s.
# Package 78's watchdog counted only Normal terminations as progress and stopped an XQC quadratic
# stage that was still converging. Writes "<reason> link=N" to REASON_FILE (empty when Gaussian exits
# by itself). Exit status: Gaussian's own when it exits by itself; 124 when the watchdog stopped it.
set -u
cmd=$1 input=$2 log=$3 reason_file=$4
GRACE=${GRACE:-180}
STALL_LIMIT=${STALL_LIMIT:-7200}
LINK_LIMIT=${LINK_LIMIT:-72000}
POLL=${POLL:-30}
: > "$reason_file"
if command -v setsid >/dev/null 2>&1; then
  setsid "$cmd" < "$input" > "$log" 2>&1 &
  group=1
else
  "$cmd" < "$input" > "$log" 2>&1 &
  group=0
fi
pid=$!
now=$(date +%s)
seen=0 link_start=$now size=0 grew=$now err_at= reason=
while kill -0 "$pid" 2>/dev/null; do
  sleep "$POLL"
  kill -0 "$pid" 2>/dev/null || break
  now=$(date +%s)
  sz=$(wc -c < "$log" 2>/dev/null || echo 0)
  if (( sz != size )); then size=$sz; grew=$now; fi
  n=$(grep -c "Normal termination" "$log" 2>/dev/null || true)
  if (( ${n:-0} != seen )); then seen=${n:-0}; link_start=$now; fi
  if [[ -z $err_at ]] && grep -q "Error termination" "$log" 2>/dev/null; then err_at=$now; fi
  if [[ -n $err_at ]] && (( now - err_at >= GRACE )); then reason=error_hang; break; fi
  if (( now - grew >= STALL_LIMIT )); then reason=stall; break; fi
  if (( now - link_start >= LINK_LIMIT )); then reason=link_timeout; break; fi
done
if [[ -n $reason ]]; then
  echo "watchdog: $reason in link $(( seen + 1 )) (link ran $(( now - link_start )) s, log idle $(( now - grew )) s); stopping $cmd (pid $pid)" >&2
  printf '%s link=%d\n' "$reason" "$(( seen + 1 ))" > "$reason_file"
  # group kill (setsid made $pid the group leader), plus the pid and its children directly in case
  # the group signal misses: G09's l*.exe links are children of the g09 driver
  stop() { (( group )) && kill -"$1" -- "-$pid" 2>/dev/null; pkill -"$1" -P "$pid" 2>/dev/null; kill -"$1" "$pid" 2>/dev/null; }
  stop TERM
  sleep 20
  stop KILL
  wait "$pid" 2>/dev/null
  exit 124
fi
wait "$pid"
"""

README = """# {number}_{name}: {title} (ClusterMLIP)

Owner: ClusterMLIP. Generator: `experiments/fe16_bs_pilot/make_scf_gate_package.py --stage {stage}` on
branch `{branch}` (commit `{commit}`), from the kit input `{kit_input}`. Regenerate there, not here.
`package.json` records every file's SHA-256.

## The question

{question}

Background: package 61 (desk numbering) ran the first full-cluster UBPW91/6-311++G* SCF inside
Stable=Opt, and it hit the 129-cycle cap unconverged in all 13 frames. Geometry fe16-200-49-gf,
neutral, M = 1, A0 (8 α / 8 β fragment guess).

| frame | aid in SCF=(...), links 1-3 | links |
|---|---|---|
{rows}

```
link 0  SP Guess=(Fragment=16)                        the kit A0 link, checkpoint renamed
link 1  SP Pop=Hirshfeld Geom=Checkpoint Guess=Read   SCF convergence on its own
link 2  Stable=Opt Pop=Hirshfeld Geom=Checkpoint Guess=Read
link 3  Force Pop=Hirshfeld Geom=Checkpoint Guess=Read
every link: IOP(5/13=1,5/36=1,8/11=1); every other route token as in package 61
```

## Run (QB4, submit from qbd)

`run_gaussian.slurm`, array 0-{last}, `gaussian/g09-d01`, `single`, {cpus} cores and {mem} per task,
{hours} h cap, at most {sus} SUs. Gaussian runs under `gaussian_watchdog.sh`, which treats log growth
as progress and kills the whole G09 process group in three cases:
- `error_hang`: {grace} s after an Error termination;
- `stall`: no log growth for {stall_hours} h;
- `link_timeout`: one link has run for {link_hours} h.

`outputs/frame_NNNN.watchdog` holds the reason and link (an empty file means G09 exited by itself).
`outputs/frame_NNNN.seconds` is the wall time. Checkpoints stay in `work/` and are not exported.

## Pass criteria (desk)

- Every `outputs/frame_*.log`, `.seconds` and `.watchdog` file exists, and the watchdog reasons are
  reported.
- An Error termination, an SCF that does not converge, or a watchdog stop is a **result**, not a
  package failure. Do not resubmit with a longer limit without asking.
- Physics is checked by ClusterMLIP:
  - does link 1 end without "Convergence criterion not met"?
  - what are its energy, ⟨S²⟩ and Hirshfeld spins, against the M = 49 FM reference (−20221.7601 Ha,
    package 36)?

## Hand-off

Route: agent `ClusterMLIP`, source `experiments/fe16_bs_pilot/make_scf_gate_package.py`.
"""


def gate_input(text: str, aid: str) -> str:
    """Kit A0 3-link tandem -> 4-link tandem with `aid` in the SCF options of links 1-3."""
    chunks = re.split(r"(?im)^[ \t]*--link1--[ \t]*\n", text)
    if len(chunks) != 3:
        raise SystemExit(f"expected the kit's 3-link A0 tandem, found {len(chunks)} links")
    link0, stable, force = chunks
    if "Stable=Opt" not in stable or " Force " not in force:
        raise SystemExit("unexpected kit links: link 1 must be Stable=Opt and link 2 Force")

    def with_aid(chunk: str, tag: str) -> str:
        route = re.search(r"(?m)^#p .*$", chunk).group(0)
        if "IOP(5/13=1" not in route:
            raise SystemExit(f"route lost IOP(5/13=1): {route}")
        new = _SCF_RE.sub(lambda m: f"SCF=({aid},{m.group(1)})", route, count=1)
        chunk = chunk.replace(route, new)
        chunk = re.sub(r"(?m)^(%chk=.*?)\.chk$", rf"\1-{aid.lower()}.chk", chunk)
        return chunk.replace("stage=1 stable_opt", tag).replace("stage=2 force", tag)

    sp = stable.replace(" Stable=Opt ", " SP ")
    link0 = re.sub(r"(?m)^(%chk=.*?)\.chk$", rf"\1-{aid.lower()}.chk", link0)
    links = [link0,
             with_aid(sp, f"stage=1 sp_converge; aid={aid}"),
             with_aid(stable, f"stage=2 stable_opt; aid={aid}"),
             with_aid(force, f"stage=3 force; aid={aid}")]
    out = "--Link1--\n".join(links)
    stages = split_input(out)
    routes = [s.route for s in stages]
    assert len(stages) == 4 and all("IOP(5/13=1,5/36=1,8/11=1)" in r for r in routes), routes
    assert "Guess=(Fragment=16)" in routes[0] and all(f"SCF=({aid}," in r for r in routes[1:]), routes
    assert [(" SP " in r, "Stable=Opt" in r, " Force " in r) for r in routes[1:]] == [
        (True, False, False), (False, True, False), (False, False, True)], routes
    assert len({re.search(r"(?m)^%chk=(.*)$", c).group(1) for c in links}) == 1
    return out


def main() -> None:
    args = sys.argv[1:]
    kit_input, parent = Path(args[0]), Path(args[1])
    stage = args[args.index("--stage") + 1]
    spec = STAGES[stage]
    number = args[args.index("--number") + 1] if "--number" in args else "NN"
    out = parent / f"{number}_{spec['name']}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite {out}")
    repo = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,
                                    check=True).stdout.strip()
    commit, branch = git("rev-parse", "--short", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    if git("status", "--porcelain", "--", "src", "experiments"):
        raise SystemExit("commit src/ and experiments/ first: package.json must name a real commit")

    text = kit_input.read_text(encoding="utf-8").replace("\r\n", "\n")
    plan, n_stages = plan_from_input(text)
    if plan is None or n_stages != 3:
        raise SystemExit(f"{kit_input.name}: not a 3-link fragment tandem")
    mem = re.search(r"(?m)^%mem=(\S+)", text).group(1)
    cpus = int(re.search(r"(?m)^%nprocshared=(\d+)", text).group(1))

    aids, hours = spec["aids"], spec["hours"]
    (out / "inputs").mkdir(parents=True)
    (out / "logs").mkdir()
    frames = [["frame", "aid", "kit_input"]]
    for i, aid in enumerate(aids):
        frame = f"frame_{i:04d}"
        write_text_lf(out / "inputs" / f"{frame}.gjf", gate_input(text, aid))
        frames.append([frame, aid, kit_input.name])
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(frames)
    write_text_lf(out / "inputs" / "frames.csv", buffer.getvalue())
    n = len(aids)
    rows = "\n".join(f"| {i:04d} | {aid} | fragment guess > SP > Stable=Opt > Force |" for i, aid in enumerate(aids))
    fmt = dict(number=number, name=spec["name"], title=spec["title"], question=spec["question"],
               job=spec["job"], stage=stage, branch=branch, commit=commit, kit_input=kit_input.name,
               rows=rows, last=n - 1, cpus=cpus, mem=mem, hours=hours, sus=n * cpus * hours,
               grace=GRACE, stall_limit=STALL_LIMIT, stall_hours=STALL_LIMIT // 3600,
               link_limit=LINK_LIMIT, link_hours=LINK_LIMIT // 3600)
    write_text_lf(out / "run_gaussian.slurm", SLURM.format(**fmt))
    write_text_lf(out / "gaussian_watchdog.sh", WATCHDOG)
    write_text_lf(out / "README.md", README.format(**fmt))
    write_text_lf(out / "export.list", "outputs/*\n")
    write_text_lf(out / ".gitignore", "outputs/\nwork/\nlogs/*\n!logs/README.txt\n")
    write_text_lf(out / "logs" / "README.txt", "Slurm logs land here.\n")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        "package": f"{number}_{spec['name']}", "owner": "ClusterMLIP",
        "generator": "experiments/fe16_bs_pilot/make_scf_gate_package.py", "stage": stage,
        "commit": commit, "branch": branch,
        "kit_input": str(kit_input), "cluster": "LONI QB4", "partition": "single",
        "gaussian": "gaussian/g09-d01", "cpus_per_task": cpus, "memory_per_task": mem, "time_hours": hours,
        "watchdog": {"grace_s": GRACE, "stall_limit_s": STALL_LIMIT, "link_limit_s": LINK_LIMIT},
        "array": [f"frame_{i:04d}" for i in range(n)], "aids": aids,
        "outputs": ["outputs/frame_NNNN.log", "outputs/frame_NNNN.seconds", "outputs/frame_NNNN.watchdog"],
        "files_sha256": {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "submitted": False,
    }
    write_text_lf(out / "package.json", json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(files) + 1} files, {n} frame(s), {cpus} cores, {mem}, {hours} h, commit {commit})")


if __name__ == "__main__":
    main()
