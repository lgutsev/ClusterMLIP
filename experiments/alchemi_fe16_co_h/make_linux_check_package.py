"""Write the dispatch package for the Linux check of the batch backend.

    python make_linux_check_package.py DATA_DIR MODEL OUT_DIR --date YYYY-MM-DD [--number 111]

DATA_DIR holds pool/pool128.extxyz and pool/fixture12.extxyz (from build_pool.py). The
package carries the code it runs (``kit/``), the frozen pool and the fixture, and needs
no installed ClusterMLIP. One GPU job, no Gaussian.

Desk rules folded in after intake as package 111 (2026-11-10):
* no model binary in a package or in Git: the job reads the master copy on LONI and
  checks its sha256 before running (MODEL here is only used to confirm the hash);
* no environment inside the package and none built on a login node: setup_env.sh is a
  GPU batch job that builds ``/project/lgutsev/env/alchemi_check`` from Miniforge
  Python 3.12 (``ALCHEMI_ENV`` overrides the location);
* partition gpu2 with ``--gres=gpu:1`` is correct on QB4.
The desk's own copies of setup_env.sh and run_check.slurm in package 111 are the ones
that ran; the templates below follow the same rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = ["alchemi_backend.py", "alchemi_evaluator.py", "parity_single_point.py", "relax_bench.py", "inflight_bench.py",
           "compare_backends.py", "inflight_compare.py", "constraints.txt"]
MODEL_SHA = "75428afe"  # MACE-MPA-0 medium, as used by the pilot
MODEL_MASTER = "/ddnB/project/ramu/lgutsev/MLIP_PROJECT_STORAGE/MLIP_Foundational_Models/mace/mace-mpa-0-medium.model"
ENV_DEFAULT = "/project/lgutsev/env/alchemi_check"

SETUP = r"""#!/bin/bash
#SBATCH --job-name=alchemi-env
#SBATCH --account=loni_perovsk27
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Builds the Python environment OUTSIDE the package (desk rule: no venv in /work, none on a login node).
# Run as a batch job: sbatch setup_env.sh. Miniforge Python 3.12 is expected on PATH as `mamba`/`conda`.
set -eo pipefail
[ -n "${SLURM_JOB_ID:-}" ] || { echo "run inside an allocation: sbatch setup_env.sh"; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from the package directory}"
ENV=${ALCHEMI_ENV:-__ENV__}
mkdir -p outputs
curl -sSfI https://pypi.org/simple/pip/ > /dev/null || { echo "PyPI not reachable from this node"; exit 1; }
CONDA=$(command -v mamba || command -v conda) || { echo "no mamba/conda on PATH"; exit 1; }
[ -x "$ENV/bin/python" ] || "$CONDA" create -y -p "$ENV" --override-channels -c conda-forge python=3.12 pip
"$ENV/bin/python" -m pip install torch==2.11.0 mace-torch==0.3.16 nvalchemi-toolkit==0.3.0 ase==3.29.0 scipy
"$ENV/bin/python" -m pip freeze > outputs/pip_freeze.txt
"$ENV/bin/python" -c 'import torch, nvalchemi, mace; ok = torch.cuda.is_available(); print("torch", torch.__version__, "cuda build", torch.version.cuda, "cuda ok", ok); raise SystemExit(0 if ok else 1)' | tee outputs/setup_ok.txt
"""

SLURM = r"""#!/bin/bash
#SBATCH --job-name=alchemi-linux-check
#SBATCH --account=loni_perovsk27
#SBATCH --partition=__PARTITION__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Written by ClusterMLIP experiments/alchemi_fe16_co_h/make_linux_check_package.py.
# One GPU, one job. gpu2 with gres=gpu:1 is what the desk uses on QB4 (A100 80 GB).
# Run `sbatch setup_env.sh` first and wait for outputs/setup_ok.txt ("cuda ok True"). Nothing here needs the network.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the package directory}"
PKG=$PWD
PY=${ALCHEMI_ENV:-__ENV__}/bin/python
K=$PKG/kit/experiments/alchemi_fe16_co_h
POOL=$PKG/data/pool128.extxyz
FIX=$PKG/data/fixture12.extxyz
MODEL=${ALCHEMI_MODEL:-__MODEL__}
# desk rule: models are not copied into packages; use the master copy and check it is the pilot's model
case "$(sha256sum "$MODEL" | cut -c1-8)" in __SHA__) ;; *) echo "model sha256 mismatch: $MODEL"; exit 1 ;; esac
OUT=$PKG/outputs
mkdir -p "$OUT"
start=$(date +%s)
{ hostname; date; nvidia-smi; "$PY" -c 'import torch; print("torch", torch.__version__, torch.cuda.get_device_name(0))'; } \
  > "$OUT/node.txt" 2>&1
: > "$OUT/status.txt"

step() {  # step NAME LOG command... : never aborts the job; the exit code is the result
  local name=$1 log=$2; shift 2
  local rc=0
  "$@" > "$log" 2>&1 || rc=$?
  echo "$name exit=$rc" >> "$OUT/status.txt"
}

suite() {  # suite DIR NATIVE_REPEATS : the same measurements as on the laptop
  local d=$OUT/$1 nrep=$2
  mkdir -p "$d/bench" "$d/inflight"
  step "$1/parity64" "$d/parity64.log" "$PY" -W ignore "$K/parity_single_point.py" "$FIX" "$MODEL" --dtype float64 --out "$d"
  step "$1/parity32" "$d/parity32.log" "$PY" -W ignore "$K/parity_single_point.py" "$FIX" "$MODEL" --dtype float32 --out "$d"
  for b in 16 64; do
    step "$1/batch$b" "$d/bench_b$b.log" "$PY" -W ignore "$K/relax_bench.py" "$POOL" "$MODEL" --backend alchemi --batch-size $b --repeat 3 --out "$d/bench"
  done
  step "$1/batch1" "$d/bench_b1.log" "$PY" -W ignore "$K/relax_bench.py" "$POOL" "$MODEL" --backend alchemi --batch-size 1 --out "$d/bench"
  step "$1/native" "$d/bench_native.log" "$PY" -W ignore "$K/relax_bench.py" "$POOL" "$MODEL" --backend native --repeat "$nrep" --out "$d/bench"
  for s in 16 64; do for r in 1 2 3; do
    step "$1/inflight${s}_r$r" "$d/inflight_s${s}_r$r.log" "$PY" -W ignore "$K/inflight_bench.py" "$POOL" "$MODEL" --slots $s --out "$d/inflight/slots${s}_r$r"
  done; done
  step "$1/compare" "$d/compare.log" "$PY" -W ignore "$K/compare_backends.py" "$d/bench" --out "$d/compare.json"
  step "$1/inflight_compare" "$d/inflight_compare.log" "$PY" -W ignore "$K/inflight_compare.py" "$d/inflight" "$d/bench" --out "$d/inflight_compare.json"
}

# 1. the toolkit as shipped (torch.compile on; this path could not run on the Windows laptop)
suite compiled 3
# 2. the laptop's configuration, for a like-for-like comparison
export TORCHDYNAMO_DISABLE=1
suite nocompile 1
unset TORCHDYNAMO_DISABLE

echo "$(( $(date +%s) - start ))" > "$OUT/seconds"
cat "$OUT/status.txt"
"""

README = """# {name}

Linux check of the optional NVIDIA ALCHEMI batch backend (ClusterMLIP, {date}). **One GPU job, about
1 hour, no Gaussian, no VASP.** Background: `docs/alchemi_cluster_feasibility.md` at commit `{commit}`.

## Why

On the Windows laptop the backend matched native MACE to rounding and relaxed 128 Fe16 + CO + 2H
candidates 6 to 12 times faster. Two things could not be tested there: the toolkit's compiled path
(`torch.compile` needs Triton, absent on Windows) and Linux itself. This package repeats the same
measurements on a LONI GPU node.

## Steps

1. Once: `sbatch setup_env.sh`. It builds `{env}` (outside the package; `ALCHEMI_ENV` overrides) from
   Miniforge Python 3.12, needs PyPI, and ends with `outputs/setup_ok.txt` saying `cuda ok True`.
2. `sbatch run_check.slurm` from this directory (`--partition={partition}`, `--gres=gpu:1`).

## What it runs

Two passes over the same suite, `outputs/compiled` (toolkit as shipped) and `outputs/nocompile`
(`TORCHDYNAMO_DISABLE=1`, the laptop's configuration):

- single-point parity against native MACE, float64 and float32, 12 structures;
- relaxation of the frozen 128-candidate pool: toolkit fixed batches of 16 and 64 (3 repeats), batch 1,
  native ASE BFGS (3 repeats in `compiled`, 1 in `nocompile`), inflight batching with 16 and 64 slots (3 runs each);
- comparison of every run with native.

A step that fails does not stop the job. `outputs/status.txt` has one line per step with its exit code.

## Done

`outputs/` (about 20 MB): `status.txt`, `node.txt`, `seconds`, `pip_freeze.txt`, and the two suite
folders. Send back the whole `outputs/` folder and the Slurm log in `logs/`.

A non-zero exit code in `status.txt` is a result, not a package failure, in particular in `compiled`.
Do not resubmit with changes without asking.

## Contents

| Path | What |
|---|---|
| `kit/` | the code that runs: `src/cluster_mlip` and the two experiment folders, from commit `{commit}` |
| `data/pool128.extxyz`, `data/fixture12.extxyz` | frozen candidate pool and parity fixture |
| model | not in the package: `{model_master}` (MACE-MPA-0 medium, sha256 `{model_sha}…`, checked by the job) |
| `setup_env.sh`, `run_check.slurm` | environment and job |
| `package.json` | SHA-256 of every file |

All energies are from MACE-MPA-0, which failed the pilot's reference gate. This package measures the
backend, not chemistry.
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def copy_lf(src: Path, dst: Path) -> None:
    write_lf(dst, src.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--number", type=int, default=111)
    ap.add_argument("--partition", default="gpu2")
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    name = f"{args.number}_clustermlip_alchemi_linux_check"
    pkg = args.out / name
    if pkg.exists():
        raise SystemExit(f"{pkg} exists; a staged package is never overwritten (send the desk a revision note)")
    if not sha(args.model).startswith(MODEL_SHA):
        raise SystemExit("model is not MACE-MPA-0 medium (sha256 mismatch)")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "experiments"], cwd=REPO, capture_output=True,
                           text=True, check=True).stdout.strip()
    for src in sorted((REPO / "src" / "cluster_mlip").glob("*.py")):
        copy_lf(src, pkg / "kit" / "src" / "cluster_mlip" / src.name)
    acat = REPO / "experiments" / "acat_fe16_co_h"
    copy_lf(acat / "common.py", pkg / "kit" / "experiments" / "acat_fe16_co_h" / "common.py")
    copy_lf(acat / "catalyst" / "fe16_200_49_gf_m49.extxyz",
            pkg / "kit" / "experiments" / "acat_fe16_co_h" / "catalyst" / "fe16_200_49_gf_m49.extxyz")
    for s in SCRIPTS:
        copy_lf(HERE / s, pkg / "kit" / "experiments" / "alchemi_fe16_co_h" / s)
    for f in ("pool128.extxyz", "fixture12.extxyz"):
        copy_lf(args.data / "pool" / f, pkg / "data" / f)
    write_lf(pkg / "setup_env.sh", SETUP.replace("__ENV__", ENV_DEFAULT))
    write_lf(pkg / "run_check.slurm", SLURM.replace("__PARTITION__", args.partition).replace("__ENV__", ENV_DEFAULT)
             .replace("__MODEL__", MODEL_MASTER).replace("__SHA__", MODEL_SHA))
    write_lf(pkg / "logs" / "README.txt", "Slurm logs land here.\n")
    # desk rule (package 111): without an export.list nothing comes back from LONI
    write_lf(pkg / "export.list", "outputs/*" + chr(10) + "logs/*" + chr(10))
    write_lf(pkg / ".gitignore", "outputs/\nlogs/*.out\n")
    write_lf(pkg / "README.md", README.format(name=name, date=args.date, commit=commit, partition=args.partition,
                                              model_sha=MODEL_SHA, model_master=MODEL_MASTER, env=ENV_DEFAULT))
    files = sorted(p for p in pkg.rglob("*") if p.is_file())
    meta = {"package": name, "owner": "ClusterMLIP", "generator": "experiments/alchemi_fe16_co_h/make_linux_check_package.py",
            "commit": commit, "branch": branch, "uncommitted_changes_in_src_or_experiments": bool(dirty),
            "cluster": "LONI QB4", "partition": args.partition, "partition_verified": True, "gpus": 1,
            "model_master": MODEL_MASTER, "model_sha256_prefix": MODEL_SHA, "environment": ENV_DEFAULT,
            "cpus_per_task": 8, "time_hours": 2, "gaussian": None, "needs_network": "setup_env.sh only (batch job)",
            "sha256": {p.relative_to(pkg).as_posix(): sha(p) for p in files}}
    write_lf(pkg / "package.json", json.dumps(meta, indent=1) + "\n")
    print(pkg, len(files) + 1, "files", "dirty" if dirty else "clean", commit)


if __name__ == "__main__":
    main()
