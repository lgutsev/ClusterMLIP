#!/usr/bin/env bash
#SBATCH --job-name=mlip_cpu_checks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --partition=single
#SBATCH --account=loni_perovsk27
#SBATCH --output=mlip_cpu_checks-%j.out
#SBATCH --error=mlip_cpu_checks-%j.err
# CPU data validation for the local-spin A/B experiment (no GPU, no mace-torch needed):
#   1. spin-table crosswalk on the raw Gaussian logs vs the spin audit and collect's provenance;
#   2. the population-labelled A/B dataset from a collect dataset.
# usage (from the ClusterMLIP checkout):
#   sbatch scripts/run_mlip_cpu_checks_slurm.sh RAW_BATCHES AUDIT_FRAMES_CSV COLLECT_DATASET OUT_DIR
set -euo pipefail
REPO=${CLUSTER_MLIP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
PY=${CLUSTER_MLIP_PYTHON:-/project/lgutsev/env/lgutsev_dev/bin/python}
RAW=$1; AUDIT=$2; DATASET=$3; OUT=$4
[[ -f "$REPO/scripts/diagnostics/spin_crosswalk/crosswalk.py" ]] || { echo "run from the ClusterMLIP checkout" >&2; exit 1; }
for p in "$RAW" "$AUDIT" "$DATASET/frame_provenance.csv"; do [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }; done
[[ -e "$OUT" ]] && { echo "$OUT exists; choose a new OUT_DIR" >&2; exit 1; }
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$REPO"; git log --oneline -1; "$PY" -V; date -Is
"$PY" scripts/diagnostics/spin_crosswalk/crosswalk.py "$RAW" "$AUDIT" "$DATASET/frame_provenance.csv" "$OUT/crosswalk"
"$PY" scripts/diagnostics/local_spin_ab/build_ab_dataset.py "$DATASET" "$OUT/ab_dataset"
date -Is
