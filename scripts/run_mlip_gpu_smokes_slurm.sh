#!/usr/bin/env bash
#SBATCH --job-name=mlip_gpu_smokes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu2
#SBATCH --account=loni_perovsk27
#SBATCH --output=mlip_gpu_smokes-%j.out
#SBATCH --error=mlip_gpu_smokes-%j.err
# GPU train -> save -> reload -> infer smokes for the local-spin experiment (short runs; they
# check the pipeline, not accuracy):
#   A  global charge/multiplicity MACE          (cluster-mlip train)
#   B  A + per-atom local moments               (cluster-mlip train --local-moment-key local_moment)
#   then smoke_local_moment.py on B and ab_eval.py on A and B;
#   C  PolarMACE from scratch, only if POLAR_PYTHON is set (an env with graph_longrange 0.4.0,
#      from github.com/WillBaldwin0/graph_electrostatics@v0.4.0; never install into a shared env).
# usage (from the ClusterMLIP checkout; AB_DATASET = output of build_ab_dataset.py):
#   MACE_PYTHON=/path/to/mace-0.3.16/python [POLAR_PYTHON=...] sbatch scripts/run_mlip_gpu_smokes_slurm.sh AB_DATASET OUT_DIR [EPOCHS]
set -euo pipefail
REPO=${CLUSTER_MLIP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
PY=${MACE_PYTHON:?set MACE_PYTHON to a python with mace-torch 0.3.16 and CUDA torch}
DATA=$1; OUT=$2; EPOCHS=${3:-5}
[[ -f "$DATA/train.extxyz" ]] || { echo "missing $DATA/train.extxyz" >&2; exit 1; }
[[ -e "$OUT" ]] && { echo "$OUT exists; choose a new OUT_DIR" >&2; exit 1; }
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"; git log --oneline -1; date -Is
"$PY" -c 'import mace, torch; print("mace", mace.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())'
gen() {  # gen NAME PYTHON [extra cluster-mlip train args...]
  local name=$1 runpy=$2; shift 2
  "$PY" -m cluster_mlip.cli train "$DATA" -o "$OUT/$name" --run-name "smoke_$name" --seed 1 \
    --max-num-epochs "$EPOCHS" --device cuda "$@"
  # run.sh calls `mace_run_train`; use the chosen interpreter so the right env is used
  sed -i "s#^mace_run_train #$runpy -m mace.cli.run_train #" "$OUT/$name/seed_1/run.sh"
}
gen A "$PY"
gen B "$PY" --local-moment-key local_moment
for m in A B; do (cd "$OUT/$m/seed_1" && bash run.sh > stdout.txt 2>&1) || { echo "$m failed" >&2; tail -20 "$OUT/$m/seed_1/stdout.txt" >&2; exit 1; }; done
MA=$(ls "$OUT"/A/seed_1/*.model | head -1); MB=$(ls "$OUT"/B/seed_1/*.model | head -1)
"$PY" scripts/diagnostics/local_spin_ab/smoke_local_moment.py "$MB" "$DATA/train.extxyz" "$OUT/B_smoke.json"
"$PY" scripts/diagnostics/local_spin_ab/ab_eval.py "$DATA" "$OUT/ab_eval.json" A="$MA" B="$MB:local_moment"
if [[ -n "${POLAR_PYTHON:-}" ]]; then
  "$POLAR_PYTHON" -c 'import graph_longrange; from importlib.metadata import version; print("graph_longrange", version("graph_longrange"))'
  gen C "$POLAR_PYTHON"
  R="$OUT/C/seed_1/run.sh"
  sed -i -e 's#--model=ScaleShiftMACE#--model=PolarMACE#' -e '/--embedding_specs=/d' -e '/--use_embedding_readout=/d' \
         -e 's#--batch_size=8#--batch_size=2#' -e 's#--valid_batch_size=8#--valid_batch_size=2#' "$R"
  (cd "$OUT/C/seed_1" && bash run.sh > stdout.txt 2>&1) || { echo "C failed" >&2; tail -20 "$OUT/C/seed_1/stdout.txt" >&2; }
fi
grep -hE "Epoch [0-9]+:" "$OUT"/*/seed_1/stdout.txt | tail -6
date -Is
