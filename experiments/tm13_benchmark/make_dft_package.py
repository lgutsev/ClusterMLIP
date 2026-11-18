"""Stage the UBPW91 reference package for a TM13 held-out set (Gaussian 09, LONI desk layout). Stages only.

    python experiments/tm13_benchmark/make_dft_package.py --set experiments/tm13_benchmark/sets/fe13_legacy_ubpw91_m45 \
        --template D:/MLIP_Work_Folder/hpc_smoke_tests/112_clustermlip_fe16_coh_reference_batch1 \
        --out D:/MLIP_Work_Folder/Cluster_MLIP/loni_package/143_clustermlip_fe13_heldout_reference

Route, link layout, Slurm script and watchdog are package 112's (the last ladder package that ran cleanly):
per multiplicity one Stable=Opt link and one Force link, one checkpoint, Guess=Read after the first link,
IOP(5/13=1,5/36=1,8/11=1) in every link, one job type per link.

Tasks:
  ladder  every configuration: M = 47 -> 45 -> 43 (flip-down from the default guess at 47), 6 links
  direct  c0_source and c4_random: a fresh default guess at M = 45, and one at M = 43, 2 links each
The direct tasks are the alternative starting guesses: they show whether the ladder state at 45 / 43 is the
state a fresh SCF finds. 6-311++G* equals 6-311+G* for Fe; the ++ is kept so the route matches the Fe16 labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROUTE = ("#p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm {job} Pop=Hirshfeld "
         "IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine")
HEAD = "%chk={chk}.chk\n%mem=24GB\n%nprocshared=12\n"
LADDER = (47, 45, 43)
DIRECT = (45, 43)
DIRECT_CONFIGS = ("c0_source", "c4_random")
HOURS = 24


def gjf(name: str, xyz: Path, mults: tuple[int, ...]) -> tuple[str, int]:
    atoms = xyz.read_text().splitlines()[2:15]
    links = []
    for m in mults:
        for job in ("Stable=Opt", "Force"):
            first = not links
            route = ROUTE.format(job=job) + ("" if first else " Geom=Checkpoint Guess=Read")
            if len(re.findall(r"\b(Stable=Opt|Force|Opt|SP|Freq)\b", route)) != 1:
                raise SystemExit(f"more than one job type in: {route}")
            body = HEAD.format(chk=name) + route + f"\n\n{name} M={m} {job}\n\n0 {m}\n"
            if first:
                body += "\n".join(f" {s:2s} {float(x):14.8f} {float(y):14.8f} {float(z):14.8f}"
                                  for s, x, y, z in (line.split() for line in atoms)) + "\n"
            links.append(body + "\n")
    return "--Link1--\n".join(links), len(links)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    setdir, template, out = Path(args.set), Path(args.template), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: never restage silently (the desk may have read it); remove it on purpose first")
    manifest = json.loads((setdir / "manifest.json").read_text())
    here = Path(__file__).resolve().parent
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=here, capture_output=True, text=True).stdout.strip()

    tasks = [(c["name"], "ladder", LADDER) for c in manifest["configurations"]]
    tasks += [(c, f"direct{m}", (m,)) for c in DIRECT_CONFIGS for m in DIRECT]
    (out / "inputs").mkdir(parents=True)
    (out / "logs").mkdir()
    frames, total = [], 0
    for i, (config, kind, mults) in enumerate(tasks):
        frame = f"frame_{i:04d}"
        name = f"fe13_{config}_{kind}"
        text, n = gjf(name, setdir / f"{config}.xyz", mults)
        (out / "inputs" / f"{frame}.gjf").write_text(text, encoding="ascii", newline="\n")
        total += n
        frames.append({"frame": frame, "job": name, "config": config, "task": kind, "charge": 0,
                       "multiplicities": " ".join(map(str, mults)), "links": n,
                       "geometry_sha256": next(c["sha256"] for c in manifest["configurations"] if c["name"] == config),
                       "input_sha256": hashlib.sha256(text.encode("ascii")).hexdigest()})
    with open(out / "inputs" / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frames[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(frames)

    shutil.copyfile(template / "gaussian_watchdog.sh", out / "gaussian_watchdog.sh")
    slurm = (template / "run_gaussian.slurm").read_text(encoding="utf-8")
    start = slurm.index("# Written by")
    end = slurm.index("# Memory comes from")
    slurm = (slurm[:start] + "# Written by ClusterMLIP experiments/tm13_benchmark/make_dft_package.py from package 112's script.\n"
             "# One array task per frame: fixed-geometry Fe13, Stable=Opt then Force per multiplicity, in\n"
             "# work/frame_NNNN, where its checkpoint stays; the log goes to outputs/frame_NNNN.log.\n"
             "# inputs/frames.csv names each frame.\n" + slurm[end:])
    for old, new in (("--job-name=fe16-coh-ref1", "--job-name=fe13-heldout"), ("--array=0-3", f"--array=0-{len(tasks) - 1}"),
                     ("--time=48:00:00", f"--time={HOURS}:00:00")):
        if old not in slurm:
            raise SystemExit(f"template changed: {old} not found")
        slurm = slurm.replace(old, new)
    (out / "run_gaussian.slurm").write_text(slurm, encoding="utf-8", newline="\n")
    (out / "export.list").write_text("outputs/*\nlogs/*\n", newline="\n")
    (out / ".gitignore").write_text("outputs/\nwork/\nlogs/*\n!logs/README.txt\n", newline="\n")
    (out / "logs" / "README.txt").write_text("Slurm logs land here.\n", newline="\n")
    table = "\n".join(f"| {f['frame']} | {f['config']} | {f['task']} | {f['multiplicities']} | {f['links']} |" for f in frames)
    (out / "README.md").write_text(f"""# {out.name}

Held-out Fe13 reference labels for the ClusterMLIP TM13 benchmark. **{len(tasks)} G09 D.01 tasks, {total} links.**
Owner: ClusterMLIP. Generator: `experiments/tm13_benchmark/make_dft_package.py` at commit `{commit}`
(branch `{branch}`). Regenerate there, not here. `package.json` has every file's SHA-256.

## The question

How well do the Fe16-trained models and the foundation models transfer to bare Fe13? The geometries are the
group's legacy UBPW91 Fe13 M = 45 structure ({manifest['source'].get('legacy_doc', '')}, state
`{manifest['source'].get('state_id', '')}`) and four small displacements of it (0.04 to 0.05 A). They are test
data only. The package gives forces and energies at M = 47, 45 and 43 on every geometry, and checks on two
geometries whether a fresh SCF at 45 or 43 finds the same state as the ladder.

## Frames

| frame | configuration | task | multiplicities | links |
|---|---|---|---|---|
{table}

Every input: fixed geometry, charge 0; per M one `Stable=Opt` link and one `Force` link, one checkpoint,
`Guess=Read` after the first link. Route as packages 91, 99, 112, 116: `UBPW91/6-311++G*`, with
`IOP(5/13=1,5/36=1,8/11=1)` in every link and one job type per link (checked by the generator).

## Run

`sbatch run_gaussian.slurm` from this directory. QB4 `single`, 12 cores, `%mem=24GB`, {HOURS} h cap per task,
per-link cap 12 h, at most {len(tasks) * 12 * HOURS} SU. `run_gaussian.slurm` is package 112's with the job
name, array range and wall time changed; `gaussian_watchdog.sh` is byte-identical to 112's.

Expected: Fe13 links should be faster than Fe16's 0.6 to 1 h (fewer basis functions); no Fe13 timing exists yet.

## Done

`outputs/frame_*.log`, `.seconds` and `.watchdog`. Report the watchdog reasons, the number of Normal
terminations per frame (= its link count for a full run) and the count of "Convergence criterion not met"
per link, Force links included.

An unconverged SCF or a watchdog stop is a result, not a package failure. Never resubmit with a longer limit
without asking.
""", encoding="utf-8", newline="\n")
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob("*")) if p.is_file()}
    package = {"package": out.name, "owner": "ClusterMLIP", "generator": "experiments/tm13_benchmark/make_dft_package.py",
               "commit": commit, "branch": branch, "set": manifest["set"], "set_source": manifest["source"],
               "template_package": template.name, "watchdog_sha256": files["gaussian_watchdog.sh"],
               "cluster": "LONI QB4", "partition": "single", "gaussian": "gaussian/g09-d01", "level": "UBPW91/6-311++G*",
               "cpus_per_task": 12, "memory_per_task": "24GB", "time_hours": HOURS, "link_limit_s": 43200,
               "max_su": len(tasks) * 12 * HOURS, "charge": 0, "links_total": total,
               "array": [f["frame"] for f in frames], "frames": frames,
               "outputs": ["outputs/frame_NNNN.log", "outputs/frame_NNNN.seconds", "outputs/frame_NNNN.watchdog"],
               "files_sha256": files, "submitted": False}
    (out / "package.json").write_text(json.dumps(package, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"staged {out}: {len(tasks)} tasks, {total} links, <= {package['max_su']} SU")


if __name__ == "__main__":
    main()
