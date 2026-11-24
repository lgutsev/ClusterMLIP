#!/usr/bin/env bash
#SBATCH --job-name=cluster_mlip_collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=single
#SBATCH --account=loni_perovsk27
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Read-only on the campaign: parse its Gaussian logs into the extxyz dataset.
# collect never touches checkpoints, inputs, inputs.txt or scheduler state.
#
#   sbatch scripts/run_collect_slurm.sh CAMPAIGN -o DATASET_DIR [collect args...]
#
# Every argument after CAMPAIGN is passed to `cluster-mlip collect`, e.g.
#   sbatch scripts/run_collect_slurm.sh "$W2" -o "$OUT/dataset_v3" \
#     --frames converged --allow-partial
#
# collect holds every labeled frame in memory before it writes the splits. If
# the job is killed for memory, resubmit with e.g. `sbatch --mem=48G ...`; any
# #SBATCH line above can be overridden that way.
#
# Logs land in the submission directory. Point CLUSTER_MLIP_REPO and
# CLUSTER_MLIP_PYTHON at the git checkout and its environment.
set -euo pipefail
[[ $# -ge 3 ]] || { echo "usage: $0 CAMPAIGN -o DATASET_DIR [collect args...]" >&2; exit 1; }
CAMPAIGN=$1
shift
REPO=${CLUSTER_MLIP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
PY=${CLUSTER_MLIP_PYTHON:-python3}
[[ -f "$REPO/src/cluster_mlip/cli.py" ]] || {
  echo "Set CLUSTER_MLIP_REPO to the ClusterMLIP checkout (got $REPO)" >&2; exit 1;
}
[[ -d "$CAMPAIGN" ]] || { echo "no such campaign directory: $CAMPAIGN" >&2; exit 1; }
# Refuse to write into an existing dataset directory: a rerun would mix two
# collections under one name.
destination=""
previous=""
for argument in "$@"; do
  case "$previous" in -o|--output) destination=$argument ;; esac
  case "$argument" in --output=*) destination=${argument#--output=} ;; esac
  previous=$argument
done
[[ -n "$destination" ]] || { echo "pass the dataset directory with -o" >&2; exit 1; }
[[ -e "$destination" ]] && { echo "refusing: $destination already exists" >&2; exit 1; }
export PYTHONPATH="$REPO/src"
echo "host $(hostname)  start $(date -Is)"
echo "code $(git -C "$REPO" rev-parse HEAD)  dirty=$(git -C "$REPO" status --porcelain | wc -l)"
"$PY" -c 'import sys, cluster_mlip; print("python", sys.version.split()[0], cluster_mlip.__file__)'
echo "collect $CAMPAIGN $*"
"$PY" -m cluster_mlip.cli collect "$CAMPAIGN" "$@"
echo "collect done $(date -Is)"
